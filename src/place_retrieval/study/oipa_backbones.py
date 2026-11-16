"""Run the OIPA A+B and loss matrix with the released SigLIP2, BLIP and CLIP checkpoints."""
import argparse
import os
from pathlib import Path

import torch
import transformers

from ..data import records, verify_formal
from ..io import digest, load_config, log, model_identity, read_json, run_lock, save_torch, seed_all, sha256, write_json
from ..oipa import OIPAHeads
from ..oipa_features import encode_feature_bank
from ..oipa_hf import FrozenHF
from .oipa import (embeddings, immutable, profile, protocol_config, runtime, score,
                   selected_heads, train_one)
from .oipa_diagnostics import evaluate_diagnostics, prepare_diagnostics
from .oipa_loss import DIRECTIONS, factorial_report

RECIPES = {"B1_Place": ("B1", "place"), "B1_MS": ("B1", "ms"),
           "AB_Place": ("AB", "place"), "AB_MS": ("AB", "ms")}
COLUMNS = {"image": ("image_id", "path", "sha256"), "text": ("text_id", "text")}


class IndexedFeature:
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return self.value[self.index[index]]


def merge_rows(groups):
    """Encode each content identity once, preserving each group's order and membership."""
    unique = {m: [] for m in COLUMNS}
    lookup = {m: {} for m in COLUMNS}
    indices = {}
    for name, rows in groups.items():
        indices[name] = {}
        for modality, values in rows.items():
            order = []
            for row in values:
                key = tuple(row[k] for k in COLUMNS[modality])
                if key not in lookup[modality]:
                    lookup[modality][key] = len(unique[modality])
                    unique[modality].append(row)
                order.append(lookup[modality][key])
            indices[name][modality] = torch.tensor(order, dtype=torch.long)
    return unique, lookup, indices


def shared_features(backbone, groups, config, device, cache, identity):
    unique, lookup, indices = merge_rows(groups)
    expected = {"study": identity, "ordered_records": {m: digest(list(v)) for m, v in lookup.items()}}
    if not cache.exists():
        log({"stage": "bank_started", "path": str(cache), "rows": {m: len(v) for m, v in unique.items()}})
        bank = encode_feature_bank(
            backbone.processor(), unique, config, device,
            lambda inputs, modality: backbone.encode(inputs, modality, True, config["oipa"]["region_grid"]),
        )
        save_torch(cache, {"identity": expected, "features": bank})
        del bank
    saved = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    if saved["identity"] != expected:
        raise ValueError(f"Backbone bank identity changed: {cache}")
    return {name: {m: {k: IndexedFeature(v, order) for k, v in saved["features"][m].items()}
                   for m, order in views.items()} for name, views in indices.items()}


def validate_backbone(backbone, config, device, root, identity, cache):
    marker = root / "validation_complete.json"
    if marker.is_file():
        done = read_json(marker)
        if done["study"] != identity:
            raise ValueError("Backbone validation identity changed")
        for path, checksum in done["selections"].items():
            if sha256(path) != checksum:
                raise ValueError(f"Validation selection changed: {path}")
        cache.unlink(missing_ok=True)
        return [Path(p) for p in done["selections"]]

    configs = {p: protocol_config(config, p) for p in config["protocols"]}
    groups = {(p, split): records(c["data"], split) for p, c in configs.items() for split in ("train", "val")}
    banks = shared_features(backbone, groups, config, device, cache, identity)
    paths = []
    for protocol, local in configs.items():
        rows = {s: groups[protocol, s] for s in ("train", "val")}
        bank = {s: banks[protocol, s] for s in ("train", "val")}
        variants = [("B0", "B0", None, None)] + [
            (name, architecture, loss, seed) for name, (architecture, loss) in RECIPES.items()
            for seed in config["seeds"]
        ]
        for method, architecture, loss, seed in variants:
            output = root / protocol / (method if seed is None else f"{method}_s{seed}")
            path = output / "selection.json"
            seed_all(42 if seed is None else seed)
            heads = OIPAHeads(backbone.output_dim, architecture, config["oipa"]["rank"]).to(device)
            variant = {"study": identity, "protocol": protocol, "method": method, "seed": seed,
                       "architecture": architecture, "loss": loss}
            if seed is None:
                checkpoint, epoch = None, 0
                validation = score(heads, bank["val"], rows["val"], device)
            else:
                options = local["training"] | {"loss": loss, "ms_direction": "symmetric"}
                log({"stage": "training_started", "backbone": identity["backbone"], "protocol": protocol,
                     "method": method, "seed": seed})
                checkpoint, epoch, validation = train_one(
                    heads, bank, rows, local | {"training": options}, device, variant, output, seed,
                )
            immutable(path, variant | {"selected_on": "val", "epoch": epoch, "validation": validation,
                                       "checkpoint": str(checkpoint) if checkpoint else None,
                                       "checkpoint_sha256": sha256(checkpoint) if checkpoint else None})
            log({"stage": "validated", "backbone": identity["backbone"], "protocol": protocol,
                 "method": method, "seed": seed, "recall1": {d: validation[d]["Recall@1"] for d in DIRECTIONS}})
            paths.append(path)
    immutable(marker, {"study": identity, "selections": {str(path): sha256(path) for path in paths}})
    cache.unlink()
    return paths


def test_backbone(backbone, config, device, root, reference, paths, gate, identity, cache):
    configs = {p: protocol_config(config, p) for p in config["protocols"]}
    test_rows = {p: records(c["data"], "test") for p, c in configs.items()}
    diag = {p: prepare_diagnostics(config, p, values, root) for p, values in test_rows.items()}
    for p, value in diag.items():
        if value["spec"] != read_json(reference / "diagnostics" / "specs" / f"{p}.json"):
            raise ValueError(f"Diagnostic cohort differs from the PE run: {p}")
    groups = {(p, "test"): rows for p, rows in test_rows.items()} | {
        (p, "diagnostic"): value["records"] for p, value in diag.items()
    }
    banks = shared_features(backbone, groups, config, device, cache, identity)
    results = []
    for protocol, local in configs.items():
        validation = records(local["data"], "val")
        for path in paths:
            if path.parent.parent.name != protocol:
                continue
            if sha256(path) != gate[str(path)]:
                raise ValueError(f"Frozen selection changed: {path}")
            selection, heads = selected_heads(path, backbone.output_dim, config["oipa"]["rank"], device)
            result_path = path.parent / "test_result.json"
            if result_path.is_file():
                result = read_json(result_path)
                if result["selection_sha256"] != gate[str(path)]:
                    raise ValueError(f"Test selection identity changed: {result_path}")
            else:
                result = {"split": "test", "protocol": protocol, "method": selection["method"],
                          "seed": selection["seed"], "selection_sha256": gate[str(path)],
                          "metric_units": "fraction", "hyperparameter_search_on_test": False,
                          "coarse_only": score(heads, banks[protocol, "test"], test_rows[protocol], device),
                          "efficiency": profile(backbone, heads, validation, local, device)}
                write_json(result_path, result)
                log({"stage": "test_result", "backbone": identity["backbone"], "protocol": protocol,
                     "method": result["method"], "seed": result["seed"],
                     "recall1": {d: result["coarse_only"][d]["Recall@1"] for d in DIRECTIONS}})
            results.append(result)
            evaluate_diagnostics(diag[protocol], embeddings(heads, banks[protocol, "diagnostic"], device),
                                 selection, gate[str(path)], root)
    write_json(root / "zero_shot_test_report.json", {"per_seed": [r for r in results if r["seed"] is None]})
    adapted = [r for r in results if r["seed"] is not None]
    write_json(root / "factorial_test_report.json", {
        "aggregate": factorial_report(adapted, config["protocols"], config["seeds"]), "per_seed": adapted,
    })
    immutable(root / "completion.json", {"status": "complete", "study": identity, "test_runs": len(results)})
    cache.unlink()
    return len(results)


def load_backbone(recipe, device):
    model = FrozenHF(recipe["directory"], device, recipe)
    if model.kind != recipe["kind"]:
        raise ValueError(f"Unexpected model type {model.kind} for {recipe['directory']}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    study = load_config(args.config)
    reference, root = (Path(study[k]).resolve() for k in ("reference", "output"))
    original = read_json(reference / "run.json")
    prior, config = original["identity"], original["config"]
    if digest(config) != prior["config"] or read_json(reference / "completion.json")["study"] != prior:
        raise ValueError("Reference run identity changed")
    with run_lock(root):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        device = torch.device(config["device"])
        from benchmark import __file__ as diagnostic_source
        from pytorch_metric_learning import __version__ as metric_learning_version

        diagnostics = {name: sha256(path) for name, path in config["diagnostics"].items() if name != "sizes"}
        diagnostics["code"] = sha256(diagnostic_source)
        if runtime(device) != prior["runtime"] or diagnostics != prior["diagnostics"]:
            raise ValueError("Reference runtime or diagnostic implementation differs")
        models = {name: model_identity(recipe["directory"]) for name, recipe in study["backbones"].items()}
        source = Path(__file__).resolve().parents[1]
        code = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*.py"))}
        identity = {"config": digest(study), "reference": prior, "code": digest(code), "models": models,
                    "transformers": transformers.__version__, "metric_learning": metric_learning_version}
        immutable(root / "run.json", {"study": identity, "config": study, "recipes": RECIPES})
        for protocol in config["protocols"]:
            verify_formal(protocol_config(config, protocol)["data"])
        selected = {}
        for name, recipe in study["backbones"].items():
            log({"stage": "backbone_started", "backbone": name, "phase": "validation"})
            model = load_backbone(recipe, device)
            selected[name] = validate_backbone(model, config, device, root / name, identity | {"backbone": name},
                                               root / "scratch" / f"{name}_fit.pt")
            del model
            torch.cuda.empty_cache()
        gate = {str(path): sha256(path) for paths in selected.values() for path in paths}
        immutable(root / "test_freeze.json", {"study": identity, "selections": gate})
        log({"stage": "test_frozen", "selections": len(gate)})
        complete = {}
        for name, recipe in study["backbones"].items():
            cache = root / "scratch" / f"{name}_test.pt"
            marker = root / name / "completion.json"
            if marker.exists():
                done = read_json(marker)
                if done["study"] != identity | {"backbone": name}:
                    raise ValueError("Backbone completion identity changed")
                complete[name] = done["test_runs"]
                cache.unlink(missing_ok=True)
                continue
            log({"stage": "backbone_started", "backbone": name, "phase": "test"})
            model = load_backbone(recipe, device)
            complete[name] = test_backbone(model, config, device, root / name, reference, selected[name], gate,
                                           identity | {"backbone": name}, cache)
            del model
            torch.cuda.empty_cache()
        immutable(root / "completion.json", {"status": "complete", "study": identity, "test_runs": complete})
        log({"stage": "complete", "test_runs": complete})


if __name__ == "__main__":
    main()
