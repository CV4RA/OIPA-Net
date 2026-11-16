"""Compare online PE LoRA + residual heads + MS with the frozen PE MS control."""
import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import image_path, records, training_batches, verify_formal
from ..io import digest, load_config, log, read_json, run_lock, save_torch, seed_all, sha256, write_json
from ..metrics import coverage
from ..models import HeadPair, raw_features
from ..oipa_features import CheckedImages, encode_feature_bank
from ..pe_core import PECore, checkpoint_identity, official_modules
from ..pe_lora import attach_pe_lora, restore_trainable, trainable_state
from ..training import StepCosine, objective, selection_key
from .oipa import immutable, protocol_config, runtime, summarize


METHOD = "PE_LoRA_Residual_MS"
DIRECTIONS = ("T2I", "I2T")


def make_model(config, recipe, device):
    backbone = PECore(config["checkpoint"]).to(device)
    attach_pe_lora(backbone, **recipe["lora"])
    return backbone, HeadPair(backbone.output_dim).to(device)


def encode(backbone, heads, inputs, modality, microbatch=None):
    batches = inputs.split(microbatch or len(inputs)) if torch.is_tensor(inputs) else (inputs,)
    device = next(heads.parameters()).device
    with torch.autocast(device.type, dtype=torch.bfloat16):
        features = torch.cat([raw_features(backbone, chunk, modality) for chunk in batches])
    with torch.autocast(device.type, enabled=False):
        if not isinstance(backbone, PECore):
            features = F.normalize(features.float(), dim=-1)
        return heads.encode(features, modality)


def score(backbone, heads, rows, config, device, processor=None):
    if processor is None:
        processor = backbone.processor()
    bank = encode_feature_bank(processor, rows, config, device,
                               lambda values, modality: {"global": encode(backbone, heads, values, modality)})
    return coverage({m: features["global"].to(device) for m, features in bank.items()}, rows)


def verify_inputs(config):
    for protocol in config["protocols"]:
        local = protocol_config(config, protocol)
        verify_formal(local["data"])
        missing = set()
        for split in ("train", "val", "test"):
            for row in records(local["data"], split)["image"]:
                path = image_path(row["path"], local["data"])
                if not path.is_file():
                    missing.add(path)
        if missing:
            raise FileNotFoundError(f"{protocol} missing {len(missing)} images, e.g. {min(missing)}")
        log({"stage": "inputs_verified", "protocol": protocol})


def control_results(recipe, config, prior):
    root = Path(recipe["control"])
    study = read_json(root / "run.json")["identity"]
    embedded_control = Path(recipe["reference"]).resolve() == root.resolve()
    reference_identity = study if embedded_control else study["reference_study"]
    if read_json(root / "completion.json")["status"] != "complete" or reference_identity != prior:
        raise ValueError("Expected the completed MS control from the same reference study")
    freeze = read_json(root / "test_freeze.json")
    if freeze["study"] != study:
        raise ValueError("Control freeze belongs to another study")
    gate = freeze["selections"]
    results = {}
    for protocol in config["protocols"]:
        for seed in config["seeds"]:
            directory = root / protocol / f"B1_MS_s{seed}"
            path = directory / "selection.json"
            selection = read_json(path)
            result = read_json(directory / "test_result.json")
            checksum = sha256(path)
            if (checksum != gate[str(path)] or checksum != result["selection_sha256"]
                    or selection["selected_on"] != "val"
                    or (not embedded_control and selection["loss"] != "ms")
                    or (embedded_control and selection.get("architecture") != "B1")
                    or (result["protocol"], result["seed"], result["method"], result["split"])
                    != (protocol, seed, "B1_MS", "test")):
                raise ValueError(f"Control provenance changed: {directory}")
            results[protocol, seed] = result
    return results


def smoke(config, recipe, device):
    seed_all(42)
    backbone, heads = make_model(config, recipe, device)
    pixels = torch.randn(32, 3, 336, 336, device=device)
    tokens = backbone.processor().tokenize([f"A street in district {i % 16}" for i in range(64)]).to(device)
    rows = [{"place_id": i % 16} for i in range(64)]
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(3):
        backbone.zero_grad(set_to_none=True)
        heads.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        images = encode(backbone, heads, pixels, "image", recipe["train_image_microbatch"])
        texts = encode(backbone, heads, tokens, "text")
        loss = objective(images, texts, rows[:32], rows, config["training"])
        loss.backward()
        torch.cuda.synchronize(device)
        durations.append(time.monotonic() - start)
    updates = {name: p.grad.norm().item() for name, p in backbone.named_parameters()
               if p.requires_grad and name.endswith("_up")}
    if len(updates) != 4 * recipe["lora"]["layers"] or not all(0 < norm < float("inf") for norm in updates.values()):
        raise ValueError("Full PE smoke run did not produce finite Q/V adapter gradients in both towers")
    log({"stage": "smoke_complete", "loss": loss.item(), "adapter_gradients": len(updates),
         "trainable_parameters": sum(p.numel() for module in (backbone, heads) for p in module.parameters()
                                     if p.requires_grad), "seconds_per_step": durations,
         "peak_cuda_allocated": torch.cuda.max_memory_allocated(device),
         "peak_cuda_reserved": torch.cuda.max_memory_reserved(device)})


def train(config, recipe, rows, identity, output, device, build=make_model):
    seed = identity["seed"]
    seed_all(seed)
    model = build(config, recipe, device)
    backbone, heads = model[:2]
    options = config["training"]
    schedule = [training_batches(rows["train"], options, seed + epoch) for epoch in range(options["epochs"])]
    adapters = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": heads.parameters(), "lr": options["head_lr"]},
                                   {"params": adapters, "lr": recipe["lora_lr"]}],
                                  weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, StepCosine(sum(map(len, schedule)), options["warmup_ratio"]))
    processor = model[2] if len(model) == 3 else backbone.processor()
    dataset = CheckedImages(rows["train"]["image"], config["data"], processor)
    texts = rows["train"]["text"]
    chunks = [processor.tokenize([r["text"] for r in texts[i:i + 1024]]) for i in range(0, len(texts), 1024)]
    token_bank = (torch.cat(chunks) if torch.is_tensor(chunks[0]) else
                  {name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]})
    last = output / "last.pt"
    epoch, step, total, best, history = 0, 0, 0.0, None, []
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError(f"Resume identity changed: {last}")
        epoch, step, total, best, history = (state[key] for key in ("epoch", "step", "total", "best", "history"))
        restore_trainable(backbone, state["weights"]["lora"])
        heads.load_state_dict(state["weights"]["heads"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    def weights():
        return {"lora": trainable_state(backbone), "heads": trainable_state(heads)}

    def save():
        save_torch(last, {"identity": identity, "epoch": epoch, "step": step, "total": total,
                          "history": history, "best": best, "weights": weights(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all()})

    workers = config["evaluation"]["workers"]
    while epoch < len(schedule):
        batches = schedule[epoch]
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        if step < len(batches):
            remaining = batches[step:]
            loader = DataLoader(dataset, batch_sampler=[indices["image"] for indices in remaining],
                                num_workers=workers, pin_memory=True,
                                generator=torch.Generator().manual_seed(seed + epoch),
                                **({"multiprocessing_context": "spawn", "prefetch_factor": 1} if workers else {}))
            for pixels, indices in zip(loader, remaining, strict=True):
                samples = {m: [rows["train"][m][i] for i in index] for m, index in indices.items()}
                optimizer.zero_grad(set_to_none=True)
                image = encode(backbone, heads, pixels.to(device, non_blocking=True), "image", recipe["train_image_microbatch"])
                text_indices = indices["text"]
                tokens = (token_bank[text_indices].to(device) if torch.is_tensor(token_bank) else
                          {key: value[text_indices].to(device) for key, value in token_bank.items()})
                text = encode(backbone, heads, tokens, "text")
                loss = objective(image, text, samples["image"], samples["text"], options)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite MS loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*heads.parameters(), *adapters], options["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                total += loss.item()
                step += 1
                if step == 1 or step % recipe["save_every"] == 0 or step == len(batches):
                    save()
                    log({"stage": "training", "protocol": identity["protocol"], "seed": seed, "epoch": epoch + 1,
                         "backbone": identity.get("backbone", "PE"), "step": step, "steps": len(batches), "mean_loss": total / step,
                         "peak_cuda_allocated": torch.cuda.max_memory_allocated(device)})
        validation = score(backbone, heads, rows["val"], config, device, processor)
        history.append({"epoch": epoch + 1, "loss": total / len(batches), "validation": validation,
                        "seconds_this_session": time.monotonic() - start,
                        "peak_cuda_allocated": torch.cuda.max_memory_allocated(device)})
        if best is None or selection_key(validation) > selection_key(best["validation"]):
            best = {"epoch": epoch + 1, "validation": validation, "weights": weights()}
        log({"stage": "epoch_complete", "protocol": identity["protocol"], "seed": seed, **history[-1]})
        epoch, step, total = epoch + 1, 0, 0.0
        save()
        write_json(output / "history.json", history)
    checkpoint = output / "best.pt"
    if not checkpoint.is_file():
        save_torch(checkpoint, {"identity": identity, **best})
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if selected["identity"] != identity or selected["epoch"] != best["epoch"] or any(
        not torch.equal(value, selected["weights"][group][name])
        for group, state in best["weights"].items() for name, value in state.items()
    ):
        raise ValueError(f"Best checkpoint differs from the validation choice: {checkpoint}")
    path = output / "selection.json"
    immutable(path, identity | {"selected_on": "val", "checkpoint": str(checkpoint),
                                "checkpoint_sha256": sha256(checkpoint), "epoch": best["epoch"],
                                "validation": best["validation"]})
    return path


def evaluate(config, recipe, path, rows, frozen, device, build=make_model):
    if sha256(path) != frozen:
        raise ValueError(f"Test selection changed: {path}")
    selection = read_json(path)
    checkpoint = Path(selection["checkpoint"])
    if selection["selected_on"] != "val" or sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError(f"Test checkpoint changed: {checkpoint}")
    result_path = path.with_name("test_result.json")
    if result_path.is_file():
        result = read_json(result_path)
        if result["selection_sha256"] != frozen:
            raise ValueError(f"Test result belongs to another selection: {result_path}")
        return result
    seed_all(selection["seed"])
    model = build(config, recipe, device)
    backbone, heads = model[:2]
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_trainable(backbone, selected["weights"]["lora"])
    heads.load_state_dict(selected["weights"]["heads"])
    metrics = score(backbone, heads, rows, config, device, model[2] if len(model) == 3 else None)
    result = {"split": "test", "metric_units": "fraction", "selection_sha256": frozen,
              "protocol": selection["protocol"], "seed": selection["seed"], "method": selection["method"],
              "coarse_only": metrics}
    if "backbone" in selection:
        result["backbone"] = selection["backbone"]
    immutable(result_path, result)
    log({"stage": "test_complete", **result})
    return result


def report(results, controls, protocols):
    groups = {}
    for protocol in protocols:
        pairs = [(controls[protocol, r["seed"]]["coarse_only"], r["coarse_only"])
                 for r in results if r["protocol"] == protocol]
        groups[protocol] = {direction: {metric: {
            "residual_ms": summarize([base[direction][metric] for base, _ in pairs]),
            "lora_residual_ms": summarize([lora[direction][metric] for _, lora in pairs]),
            "paired_delta": summarize([lora[direction][metric] - base[direction][metric] for base, lora in pairs]),
        } for metric in ("Recall@1", "Recall@5", "Recall@10")} for direction in DIRECTIONS}
    return {"metric_units": "fraction", "aggregate": groups, "per_seed": results,
            "control_per_seed": list(controls.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    recipe = load_config(args.config)
    source = read_json(Path(recipe["reference"]) / "run.json")
    config, prior = source["config"], source["identity"]
    if digest(config) != prior["config"]:
        raise ValueError("Reference configuration changed")
    config = config | {"training": config["training"] | {"loss": "ms", "ms_direction": "symmetric"},
                       "evaluation": config["evaluation"] | {"encode_batch_size": recipe["encode_batch_size"]}}
    if args.verify_inputs:
        verify_inputs(config)
        return
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    official = Path(official_modules()[0].__file__).parent
    pe = checkpoint_identity(config["checkpoint"])
    local_runtime = runtime(device)
    if args.smoke:
        smoke(config, recipe, device)
        return
    root = Path(recipe["output"]).resolve()
    with run_lock(root):
        verify_inputs(config)
        controls = control_results(recipe, config, prior)
        source_dir = Path(__file__).resolve().parents[1]
        identity = {"reference": prior, "config": digest(config), "recipe": recipe, "pe": pe, "runtime": local_runtime,
                    "official": digest({p.name: sha256(p) for p in sorted(official.glob("*.py"))}),
                    "code": digest({p.relative_to(source_dir).as_posix(): sha256(p) for p in sorted(source_dir.rglob("*.py"))})}
        immutable(root / "run.json", identity)
        paths = {}
        for protocol in config["protocols"]:
            local = protocol_config(config, protocol)
            rows = {split: records(local["data"], split) for split in ("train", "val")}
            for seed in config["seeds"]:
                run_id = {"study": digest(identity), "protocol": protocol, "seed": seed, "method": METHOD}
                output = root / protocol / f"s{seed}"
                paths[protocol, seed] = train(local, recipe, rows, run_id, output, device)
        frozen = {str(path): sha256(path) for path in paths.values()}
        immutable(root / "test_freeze.json", {"study": digest(identity), "selections": frozen})
        results = []
        for protocol in config["protocols"]:
            local = protocol_config(config, protocol)
            rows = records(local["data"], "test")
            for seed in config["seeds"]:
                path = paths[protocol, seed]
                results.append(evaluate(local, recipe, path, rows, frozen[str(path)], device))
        write_json(root / "coarse_test_report.json", report(results, controls, config["protocols"]))
        write_json(root / "completion.json", {"status": "complete", "test_runs": len(results)})
        log({"stage": "complete", "test_runs": len(results)})


if __name__ == "__main__":
    main()
