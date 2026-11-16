"""Run PE-Core coarse baselines on the released P0/P1 protocols."""
import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch

from ..data import records, verify_formal
from ..encoding import encode_split
from ..io import digest, load_config, log, read_json, run_lock, seed_all, sha256, write_json
from ..metrics import coverage
from ..models import load_coarse, restore_coarse
from ..pe_core import CHECKPOINT_FORMAT, checkpoint_identity
from ..training import train

PROTOCOL_FINGERPRINTS = {
    "P0": "0aa126e367f28be0b4f63356081c55dc9c3525c99cd70ba81a5689dd9aa06d4a",
    "P1": "fa9e4fe583d0e364883aa4a9629caf287e31d50e4f6a52babbbe476d7e430378",
}
SELECTION_FORMAT = "pe-core-coarse-selection-v1"


def configure(base, baseline, seed):
    c = deepcopy(base)
    protocol = c["protocol"]
    if protocol not in PROTOCOL_FINGERPRINTS or c["data"]["fingerprint"] != PROTOCOL_FINGERPRINTS[protocol]:
        raise ValueError("Expected the published formal fingerprint for the selected P0/P1 protocol")
    if baseline not in ("B0", "B1"):
        raise ValueError(f"Unknown PE-Core baseline: {baseline}")
    c["baseline"] = baseline
    c["seed"] = base["seed"] if seed is None else seed
    c["training"].update(residual_heads=baseline == "B1", loss="place")
    variant = "b0" if baseline == "B0" else f"b1_s{c['seed']}"
    c["output"] = str(Path(c["output"]) / variant)
    return c


def _score(c, manifest, split, state=None):
    rows = records(c["data"], split)
    expected = manifest["splits"][split]
    if len(rows["image"]) != expected["gallery_images"] or len(rows["text"]) != expected["texts"]:
        raise ValueError("Split records disagree with the formal manifest counts")
    device = torch.device(c["device"])
    backbone, processor, heads = load_coarse(c, device)
    if state is not None:
        restore_coarse(backbone, heads, state)
    return coverage(encode_split(backbone, processor, heads, rows, c, device), rows)


def _selection(c, manifest, pretrained, output, checkpoint=None, epoch=0):
    return {
        "format": SELECTION_FORMAT,
        "selected_on": "val",
        "protocol": c["protocol"],
        "formal_fingerprint": manifest["fingerprint"],
        "config_digest": digest(c),
        "baseline": c["baseline"],
        "seed": c["seed"],
        "model": pretrained,
        "coarse_checkpoint": os.path.relpath(checkpoint, output) if checkpoint is not None else None,
        "coarse_sha256": sha256(checkpoint) if checkpoint is not None else None,
        "coarse_epoch": epoch,
    }


def _report(split, selection, metrics):
    return {
        "split": split,
        "regime": "zero_shot" if selection["baseline"] == "B0" else "fine_tuned",
        "metric_units": "fraction",
        "selection": selection,
        "coarse_only": metrics,
    }


def validate(c, resume=False):
    """Evaluate B0 on validation, or train and select B1 on validation."""
    output = Path(c["output"]) / "evaluation"
    selection_path = output / "selection.json"
    with run_lock(output):
        manifest = verify_formal(c["data"])
        if c["baseline"] == "B0":
            seed_all(c["seed"])
            pretrained = checkpoint_identity(c["models"]["pe_core"])
            selection = _selection(c, manifest, pretrained, output)
            if selection_path.is_file() and read_json(selection_path) == selection:
                return selection_path
            if selection_path.exists():
                raise ValueError("A different selection already exists. Use a new output directory.")
            metrics = _score(c, manifest, "val")
        else:
            checkpoint = train(c, resume=resume)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            selection = _selection(c, manifest, state["identity"]["pe_core"], output,
                                   checkpoint, state["epoch"])
            metrics = state["validation"]
        if selection_path.is_file() and read_json(selection_path) != selection:
            raise ValueError("A different selection already exists. Use a new output directory.")
        write_json(output / "val_result.json", _report("val", selection, metrics))
        write_json(selection_path, selection)
    return selection_path


def evaluate_test(c, selection_path):
    output = Path(c["output"]) / "evaluation"
    result_path = output / "test_result.json"
    with run_lock(output):
        manifest = verify_formal(c["data"])
        pretrained = checkpoint_identity(c["models"]["pe_core"])
        selection_path = Path(selection_path)
        selection = read_json(selection_path)
        checkpoint = Path(c["output"]) / "coarse/best.pt" if c["baseline"] == "B1" else None
        epoch = selection.get("coarse_epoch") if checkpoint is not None else 0
        expected = _selection(c, manifest, pretrained, selection_path.parent, checkpoint, epoch)
        if selection != expected:
            raise ValueError("Test requires the unchanged validation selection and checkpoint for this run")
        state = None
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            identity = {"formal_fingerprint": manifest["fingerprint"],
                        "config": digest(c), "pe_core": pretrained}
            if state.get("format") != CHECKPOINT_FORMAT or state["identity"] != identity or state["epoch"] != epoch:
                raise ValueError("The checkpoint belongs to a different PE-Core run")
        if result_path.is_file():
            report = read_json(result_path)
            if report["selection"] != selection:
                raise ValueError("A completed test used another selection")
            return report
        metrics = _score(c, manifest, "test", state)
        report = dict(_report("test", selection, metrics), hyperparameter_search_on_test=False)
        write_json(result_path, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("zero-shot", "train", "test"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name in ("train", "test"):
            command.add_argument("--seed", type=int)
        if name == "train":
            command.add_argument("--resume", action="store_true")
        if name == "test":
            command.add_argument("--baseline", required=True, choices=("B0", "B1"))
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline if args.command == "test" else ("B0" if args.command == "zero-shot" else "B1")
    c = configure(load_config(args.config), baseline, getattr(args, "seed", None))
    torch.set_num_threads(c.get("cpu_threads", 8))
    if args.command == "test":
        log(evaluate_test(c, args.selection))
    else:
        log({"selection": str(validate(c, resume=getattr(args, "resume", False)))})


if __name__ == "__main__":
    main()
