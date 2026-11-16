"""Run matched PE-Core ablations and freeze every validation choice before test."""
import argparse
import os
import statistics
import time
from pathlib import Path

import torch

from ..data import records, training_batches, verify_formal
from ..io import digest, load_config, log, read_json, run_lock, save_torch, seed_all, sha256, write_json
from ..metrics import coverage
from ..oipa import ABLATIONS, OIPAHeads, objective
from ..oipa_features import CheckedImages, encode_bank, native_features
from ..pe_core import PECore, checkpoint_identity, official_modules
from ..training import StepCosine, selection_key
from .pe_core import PROTOCOL_FINGERPRINTS


def immutable(path, value):
    if path.is_file() and read_json(path) != value:
        raise ValueError(f"Frozen identity differs: {path}")
    write_json(path, value)


def runtime(device):
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The study requires CUDA with BF16 support")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    q = torch.zeros((1, 16, 577, 64), device=device, dtype=torch.bfloat16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize(device)
    kernels = [event.key for event in trace.key_averages() if "_scaled_dot_product" in event.key]
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize(device)
        flash = True
    except RuntimeError:
        flash = False
    return {"gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "vision_sdpa_probe": kernels, "flash_available": flash,
            "feature_precision": "bf16_autocast", "adapter_precision": "float32"}


def bank_for(backbone, rows, config, device, path, identity):
    id_columns = {"image": ("image_id", "path", "sha256"), "text": ("text_id", "text")}
    signature = {m: digest([tuple(row[k] for k in id_columns[m]) for row in values])
                 for m, values in rows.items()}
    expected = {"study": identity, "rows": signature}
    if not path.is_file():
        log({"stage": "bank_started", "path": str(path), "rows": {m: len(v) for m, v in rows.items()}})
        save_torch(path, {"identity": expected, "features": encode_bank(backbone, rows, config, device)})
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if saved["identity"] != expected:
        raise ValueError(f"Feature bank identity differs: {path}")
    return saved["features"]


def embeddings(heads, bank, device):
    heads.eval()
    with torch.inference_mode():
        return {m: torch.cat([
            heads({k: v[start:start + 256].to(device) for k, v in features.items()
                   if k == "global" or heads.uses_tokens(m)}, m)[0].cpu()
            for start in range(0, len(features["global"]), 256)
        ]) for m, features in bank.items()}


def score(heads, bank, rows, device):
    features = embeddings(heads, bank, device)
    return coverage({m: value.to(device) for m, value in features.items()}, rows)


def train_one(heads, banks, rows, config, device, identity, output, seed):
    options = config["training"]
    last = output / "last.pt"
    schedule = [training_batches(rows["train"], options, seed + epoch) for epoch in range(options["epochs"])]
    optimizer = torch.optim.AdamW(heads.parameters(), lr=options["head_lr"], weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, StepCosine(sum(map(len, schedule)), options["warmup_ratio"])
    )
    history, best = [], None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError(f"Training identity differs: {last}")
        heads.load_state_dict(state["heads"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], state["best"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    for epoch in range(len(history), len(schedule)):
        heads.train()
        total = 0.0
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        for step, indices in enumerate(schedule[epoch], start=1):
            batch = {m: {k: v[index].to(device) for k, v in banks["train"][m].items()
                         if k == "global" or heads.uses_tokens(m)}
                     for m, index in indices.items()}
            samples = {m: [rows["train"][m][i] for i in index] for m, index in indices.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = objective(heads, batch, samples, options | config["oipa"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), options["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if step % 500 == 0 or step == len(schedule[epoch]):
                log({"stage": "training", "output": str(output), "epoch": epoch + 1,
                     "step": step, "steps": len(schedule[epoch]), "mean_loss": total / step})
        validation = score(heads, banks["val"], rows["val"], device)
        metrics = {"epoch": epoch + 1, "loss": total / len(schedule[epoch]), "validation": validation,
                   "seconds": time.monotonic() - start, "peak_cuda_allocated": torch.cuda.max_memory_allocated(device)}
        history.append(metrics)
        weights = {k: value.detach().cpu().clone() for k, value in heads.state_dict().items()}
        if best is None or selection_key(validation) > selection_key(best["validation"]):
            best = {"epoch": epoch + 1, "validation": validation, "heads": weights}
        save_torch(last, {"identity": identity, "heads": weights, "history": history, "best": best,
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
        write_json(output / "history.json", history)
        log({"stage": "epoch_complete", "output": str(output), **metrics})
    checkpoint = output / "best.pt"
    if not checkpoint.is_file():
        save_torch(checkpoint, {"identity": identity, **best})
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or any(not torch.equal(v, saved["heads"][k]) for k, v in best["heads"].items()):
            raise ValueError(f"Best checkpoint changed: {checkpoint}")
    return checkpoint, best["epoch"], best["validation"]


def fit(config, backbone, identity, root, device):
    selections = []
    for protocol in config["protocols"]:
        local = protocol_config(config, protocol)
        rows = {split: records(local["data"], split) for split in ("val", "train")}
        banks = {}
        for split in ("val", "train"):
            banks[split] = bank_for(backbone, rows[split], local, device,
                                    root / "banks" / f"{protocol}_{split}.pt", identity)
            if split == "val":
                if "B0" in config["methods"]:
                    output = root / protocol / "B0"
                    heads = OIPAHeads(backbone.output_dim, "B0", config["oipa"]["rank"]).to(device)
                    validation = score(heads, banks["val"], rows["val"], device)
                    selections.append(write_selection(output, identity, protocol, "B0", None, None, 0, validation))
        for method in [m for m in config["methods"] if m != "B0"]:
            for seed in config["seeds"]:
                output = root / protocol / f"{method}_s{seed}"
                seed_all(seed)
                recipe = config.get("recipes", {}).get(method, {"architecture": method, "loss": "place"})
                heads = OIPAHeads(backbone.output_dim, recipe["architecture"], config["oipa"]["rank"]).to(device)
                variant = {"study": identity, "protocol": protocol, "method": method, "seed": seed, **recipe}
                options = local | {"training": local["training"] | {"loss": recipe["loss"], "ms_direction": "symmetric"}}
                checkpoint, epoch, validation = train_one(heads, banks, rows, options, device, variant, output, seed)
                selections.append(write_selection(output, identity, protocol, method, seed, checkpoint, epoch,
                                                  validation, recipe.get("architecture") if method != recipe["architecture"] else None))
    return selections


def write_selection(output, identity, protocol, method, seed, checkpoint, epoch, validation, architecture=None):
    path = output / "selection.json"
    selection = {"study": identity, "protocol": protocol, "method": method, "seed": seed,
                 "selected_on": "val", "epoch": epoch, "checkpoint": str(checkpoint) if checkpoint else None,
                 "checkpoint_sha256": sha256(checkpoint) if checkpoint else None, "validation": validation}
    if architecture is not None:
        selection["architecture"] = architecture
    immutable(path, selection)
    log({"stage": "validated", "path": str(path), "validation": validation})
    return path


def protocol_config(config, protocol):
    data = config["data"] | {"formal": config["protocols"][protocol],
                             "fingerprint": config.get("protocol_fingerprints", PROTOCOL_FINGERPRINTS)[protocol]}
    return config | {"data": data}


def selected_heads(path, width, rank, device):
    selection = read_json(path)
    heads = OIPAHeads(width, selection.get("architecture", selection["method"]), rank).to(device)
    if selection["checkpoint"] is not None:
        if sha256(selection["checkpoint"]) != selection["checkpoint_sha256"]:
            raise ValueError(f"Validation checkpoint changed: {selection['checkpoint']}")
        state = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
        heads.load_state_dict(state["heads"])
    return selection, heads


@torch.inference_mode()
def profile(backbone, heads, rows, config, device):
    processor = backbone.processor()
    images = CheckedImages(rows["image"], config["data"], processor)
    inputs = {"image": images[0].unsqueeze(0).to(device),
              "text": processor.tokenize([rows["text"][0]["text"]]).to(device)}
    heads.eval()
    samples = {}
    for modality, value in inputs.items():
        timings = []
        for step in range(15):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            if step == 5:
                torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens, grid = heads.uses_tokens(modality), config["oipa"]["region_grid"]
                features = (native_features(backbone, value, modality, tokens, grid)
                            if isinstance(backbone, PECore) else backbone.encode(value, modality, tokens, grid))
            heads(features, modality)
            torch.cuda.synchronize(device)
            if step >= 5:
                timings.append((time.perf_counter() - start) * 1000)
        samples[modality] = {"median_ms": statistics.median(timings),
                             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    adapter_parameters = sum(p.numel() for p in heads.parameters())
    return {"batch_size": 1, "warmup": 5, "repetitions": 10, "model_forward_only": True,
            "total_parameters": sum(p.numel() for p in backbone.parameters()) + adapter_parameters,
            "trainable_parameters": adapter_parameters, "modalities": samples}


def finish(config, backbone, identity, paths, root, device):
    from .oipa_diagnostics import evaluate_diagnostics, prepare_diagnostics

    gate = {str(path): sha256(path) for path in paths}
    immutable(root / "test_freeze.json", {"study": identity, "selections": gate})
    reports = []
    for protocol in config["protocols"]:
        local = protocol_config(config, protocol)
        rows = records(local["data"], "test")
        validation = records(local["data"], "val")
        bank = bank_for(backbone, rows, local, device, root / "banks" / f"{protocol}_test.pt", identity)
        diag = prepare_diagnostics(config, protocol, rows, root)
        diag_bank = bank_for(backbone, diag["records"], local, device,
                             root / "banks" / f"{protocol}_diagnostic.pt", identity)
        for path in paths:
            if sha256(path) != gate[str(path)]:
                raise ValueError(f"Frozen selection changed: {path}")
            if read_json(path)["protocol"] != protocol:
                continue
            selection, heads = selected_heads(path, backbone.output_dim, config["oipa"]["rank"], device)
            report_path = path.parent / "test_result.json"
            if report_path.is_file():
                report = read_json(report_path)
                if report["selection_sha256"] != gate[str(path)]:
                    raise ValueError(f"Test result identity differs: {report_path}")
            else:
                report = {"split": "test", "metric_units": "fraction", "selection_sha256": gate[str(path)],
                          "protocol": protocol, "method": selection["method"], "seed": selection["seed"],
                          "coarse_only": score(heads, bank, rows, device), "hyperparameter_search_on_test": False,
                          "efficiency": profile(backbone, heads, validation, local, device)}
                write_json(report_path, report)
            reports.append(report)
            evaluate_diagnostics(diag, embeddings(heads, diag_bank, device), selection,
                                 gate[str(path)], root)
    for regime, methods in (("zero_shot", ["B0"]), ("fine_tuned", [m for m in config["methods"] if m != "B0"])):
        selected = [r for r in reports if r["method"] in methods]
        aggregate = {}
        for protocol in config["protocols"]:
            aggregate[protocol] = {}
            for method in methods:
                runs = [r["coarse_only"] for r in selected if r["method"] == method and r["protocol"] == protocol]
                if runs:
                    aggregate[protocol][method] = {d: {
                        f"Recall@{k}": summarize([r[d][f"Recall@{k}"] for r in runs]) for k in (1, 5, 10)
                    } for d in ("T2I", "I2T")}
        write_json(root / f"{regime}_test_report.json", {"aggregate": aggregate, "per_seed": selected})
    write_json(root / "completion.json", {"status": "complete", "study": identity, "test_runs": len(reports)})


def summarize(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if set(config["protocols"]) != set(config.get("protocol_fingerprints", PROTOCOL_FINGERPRINTS)) or set(config["methods"]) - (ABLATIONS.keys() | config.get("recipes", {}).keys()):
        raise ValueError("Expected P0/P1 and known ablations")
    root = Path(config["output"]).resolve()
    with run_lock(root):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        device = torch.device(config["device"])
        pe, _ = official_modules()
        source = Path(__file__).resolve().parents[1]
        code = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*.py"))}
        native = {p.name: sha256(p) for p in sorted(Path(pe.__file__).parent.iterdir())
                  if p.is_file() and p.suffix in (".py", ".gz")}
        info = runtime(device)
        log({"stage": "runtime", **info})
        from benchmark import __file__ as diagnostic_source
        diagnostics = {name: sha256(path) for name, path in config["diagnostics"].items() if name != "sizes"}
        diagnostics["code"] = sha256(diagnostic_source)
        identity = {"config": digest(config), "code": digest(code), "native": digest(native), "runtime": info,
                    "diagnostics": diagnostics,
                    "weights": checkpoint_identity(config["checkpoint"])}
        for protocol in config["protocols"]:
            local = protocol_config(config, protocol)
            verify_formal(local["data"])
        immutable(root / "run.json", {"identity": identity, "config": config})
        seed_all(42)
        backbone = PECore(config["checkpoint"]).to(device)
        paths = fit(config, backbone, identity, root, device)
        finish(config, backbone, identity, paths, root, device)


if __name__ == "__main__":
    main()
