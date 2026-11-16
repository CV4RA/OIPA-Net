"""Matched Q/V LoRA and symmetric MS runs for the released HF backbones."""
import argparse
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from torch import nn
from torch.nn.utils import parametrize

from ..data import records
from ..io import digest, load_config, log, model_identity, read_json, run_lock, seed_all, sha256, write_json
from ..models import HeadPair
from ..pe_lora import QVLoRA, checkpoint_forward
from ..training import objective
from .oipa import immutable, protocol_config, runtime
from .oipa_backbones import load_backbone
from .pe_lora import encode, evaluate, report, train, verify_inputs


class LinearLoRA(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.down = nn.Parameter(torch.randn(rank, in_dim, device=weight.device) * 0.02)
        self.up = nn.Parameter(torch.zeros(out_dim, rank, device=weight.device))
        self.scale = alpha / rank

    def forward(self, weight):
        with torch.autocast(weight.device.type, enabled=False):
            return weight + self.scale * (self.up @ self.down).to(weight.dtype)


def attach_qv(model, options):
    depth, rank, alpha = (options[k] for k in ("layers", "rank", "alpha"))
    if model.config.model_type == "blip":
        vision = model.vision_model.encoder.layers
        text = model.text_encoder.encoder.layer
        if not 1 <= depth <= min(len(vision), len(text)):
            raise ValueError("Invalid BLIP adaptation depth")
        for block in vision[-depth:]:
            qkv = block.self_attn.qkv
            width = qkv.in_features
            if qkv.out_features != 3 * width:
                raise ValueError("Unexpected BLIP fused QKV shape")
            parametrize.register_parametrization(qkv, "weight", QVLoRA(width, rank, alpha).to(qkv.weight.device))
        groups = [(vision[-depth:], ()), (text[-depth:], ("attention.self.query", "attention.self.value"))]
    else:
        towers = (model.vision_model.encoder.layers, model.text_model.encoder.layers)
        if not 1 <= depth <= min(map(len, towers)):
            raise ValueError("Invalid adaptation depth")
        groups = [(blocks[-depth:], ("self_attn.q_proj", "self_attn.v_proj")) for blocks in towers]
    for blocks, paths in groups:
        for block in blocks:
            for path in paths:
                layer = block.get_submodule(path)
                parametrize.register_parametrization(layer, "weight", LinearLoRA(layer.weight, rank, alpha))
            if options["checkpoint_blocks"]:
                block.forward = checkpoint_forward(block.forward)


def make_model(config, recipe, device):
    wrapper = load_backbone(recipe["model"], device)
    backbone = wrapper.model
    attach_qv(backbone, recipe["lora"])
    return backbone, HeadPair(wrapper.output_dim).to(device), wrapper.processor()


def smoke(config, recipe, device):
    seed_all(42)
    wrapper = load_backbone(recipe["model"], device)
    backbone, processor = wrapper.model, wrapper.processor()
    heads = HeadPair(wrapper.output_dim).to(device)
    shape = processor.preprocess_image(Image.new("RGB", (256, 256))).shape
    pixels = torch.randn(32, *shape, device=device)
    tokens = processor.tokenize([f"A street in district {i % 16}" for i in range(64)]).to(device)
    with torch.no_grad():
        expected = (encode(backbone, heads, pixels, "image"), encode(backbone, heads, tokens, "text"))
    attach_qv(backbone, recipe["lora"])
    with torch.no_grad():
        for actual, before in zip((encode(backbone, heads, pixels, "image"), encode(backbone, heads, tokens, "text")), expected):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
    rows = [{"place_id": i % 16} for i in range(64)]
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    images = encode(backbone, heads, pixels, "image", recipe["train_image_microbatch"])
    texts = encode(backbone, heads, tokens, "text")
    loss = objective(images, texts, rows[:32], rows, config["training"])
    loss.backward()
    gradients = [p.grad.norm().item() for name, p in backbone.named_parameters()
                 if p.requires_grad and name.endswith((".up", "_up"))]
    if len(gradients) != 4 * recipe["lora"]["layers"] or not all(0 < g < float("inf") for g in gradients):
        raise RuntimeError("Expected finite nonzero Q/V gradients in both towers")
    log({"stage": "smoke_complete", "backbone": recipe["name"], "loss": loss.item(),
         "gradient_tensors": len(gradients), "seconds": time.monotonic() - start,
         "peak_cuda_allocated": torch.cuda.max_memory_allocated(device)})


def controls_for(root, name, config, gate):
    results = {}
    for protocol in config["protocols"]:
        for seed in config["seeds"]:
            directory = root / name / protocol / f"B1_MS_s{seed}"
            path = directory / "selection.json"
            selection, result = read_json(path), read_json(directory / "test_result.json")
            if (sha256(path) != gate[str(path)] or result["selection_sha256"] != gate[str(path)]
                    or selection["selected_on"] != "val" or selection["loss"] != "ms"
                    or (result["protocol"], result["seed"], result["method"], result["split"])
                    != (protocol, seed, "B1_MS", "test")):
                raise ValueError(f"Invalid frozen MS control: {directory}")
            results[protocol, seed] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    study = load_config(args.config)
    reference = read_json(Path(study["reference"]) / "run.json")
    config, prior = reference["config"], reference["identity"]
    control_root = Path(study["control"]).resolve()
    control = read_json(control_root / "run.json")
    frozen = read_json(control_root / "test_freeze.json")
    if (digest(config) != prior["config"] or control["study"]["reference"] != prior
            or frozen["study"] != control["study"] or read_json(control_root / "completion.json")["status"] != "complete"):
        raise ValueError("Frozen control does not match the reference study")
    from pytorch_metric_learning import __version__ as metric_learning_version

    if (control["study"]["transformers"] != transformers.__version__
            or control["study"]["metric_learning"] != metric_learning_version):
        raise ValueError("Training dependencies differ from the control")
    config["training"].update(loss="ms", ms_direction="symmetric")
    config["evaluation"]["encode_batch_size"] = study["encode_batch_size"]
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    if runtime(device) != prior["runtime"]:
        raise ValueError("CUDA runtime differs from the matched control")
    recipes = {name: study | {"model": control["config"]["backbones"][name], "name": name}
               for name in study["backbones"]}
    for name, recipe in recipes.items():
        if model_identity(recipe["model"]["directory"]) != control["study"]["models"][name]:
            raise ValueError(f"Checkpoint files changed: {name}")
    if args.smoke:
        for recipe in recipes.values():
            smoke(config, recipe, device)
            torch.cuda.empty_cache()
        return
    root = Path(study["output"]).resolve()
    with run_lock(root):
        verify_inputs(config)
        source = Path(__file__).resolve().parents[1]
        identity = {"study": study, "config": digest(config), "control": control["study"],
                    "code": digest({p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*.py"))})}
        immutable(root / "run.json", identity)
        control_results = {name: controls_for(control_root, name, config, frozen["selections"]) for name in recipes}
        paths = {}
        for name, recipe in recipes.items():
            for protocol in config["protocols"]:
                local = protocol_config(config, protocol)
                rows = {split: records(local["data"], split) for split in ("train", "val")}
                for seed in config["seeds"]:
                    run_id = {"study": digest(identity), "protocol": protocol, "seed": seed,
                              "method": "OIPA-Net (LoRA)", "backbone": name}
                    output = root / name / protocol / f"s{seed}"
                    log({"stage": "training_started", **run_id})
                    paths[name, protocol, seed] = train(local, recipe, rows, run_id, output, device, build=make_model)
        gate = {str(path): sha256(path) for path in paths.values()}
        immutable(root / "test_freeze.json", {"study": digest(identity), "selections": gate})
        log({"stage": "test_frozen", "selections": len(gate)})
        all_reports = {}
        for name, recipe in recipes.items():
            results = []
            for protocol in config["protocols"]:
                local = protocol_config(config, protocol)
                rows = records(local["data"], "test")
                for seed in config["seeds"]:
                    path = paths[name, protocol, seed]
                    results.append(evaluate(local, recipe, path, rows, gate[str(path)], device, build=make_model))
            all_reports[name] = report(results, control_results[name], config["protocols"])
        write_json(root / "coarse_test_report.json", all_reports)
        immutable(root / "completion.json", {"status": "complete", "test_runs": len(paths)})
        log({"stage": "complete", "test_runs": len(paths)})


if __name__ == "__main__":
    main()
