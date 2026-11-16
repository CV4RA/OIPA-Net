"""Compare Place and symmetric Multi-Similarity with and without OIPA A+B."""
import argparse
import os
from pathlib import Path

import torch

from ..data import records, verify_formal
from ..io import digest, load_config, log, read_json, run_lock, seed_all, sha256, write_json
from ..oipa import OIPAHeads
from ..pe_core import PECore, checkpoint_identity, official_modules
from .oipa import (bank_for, embeddings, immutable, profile, protocol_config, runtime, score,
                   selected_heads, summarize, train_one)
from .oipa_diagnostics import evaluate_diagnostics, prepare_diagnostics

DIRECTIONS = ("T2I", "I2T")
KS = (1, 5, 10)
CONTRASTS = {
    "ms_effect_B1": ("B1_MS", "B1_Place"),
    "ms_effect_AB": ("AB_MS", "AB_Place"),
    "AB_effect_with_place": ("AB_Place", "B1_Place"),
    "AB_effect_with_ms": ("AB_MS", "B1_MS"),
    "AB_MS_vs_B1_Place": ("AB_MS", "B1_Place"),
}


def shared_bank(reference, prior, protocol, split, rows, config, device):
    path = reference / "banks" / f"{protocol}_{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original frozen feature bank: {path}")
    return bank_for(None, rows, config, device, path, prior)


def reference_b1(reference, prior, config):
    freeze = read_json(reference / "test_freeze.json")
    if freeze["study"] != prior:
        raise ValueError("Original test freeze disagrees with its study")
    paths, results = [], []
    for protocol in config["protocols"]:
        for seed in config["seeds"]:
            run = reference / protocol / f"B1_s{seed}"
            path = run / "selection.json"
            selection, result = read_json(path), read_json(run / "test_result.json")
            recorded = freeze["selections"].get(str(path))
            if (sha256(path) != recorded or result["selection_sha256"] != recorded
                    or (result["split"], result["protocol"], result["seed"], result["method"])
                    != ("test", protocol, seed, "B1")
                    or selection["study"] != prior or selection["selected_on"] != "val"
                    or selection["protocol"] != protocol or selection["seed"] != seed
                    or selection["method"] != "B1"):
                raise ValueError(f"Original B1 test or selection changed: {run}")
            paths.append(path)
            results.append(result | {"method": "B1_Place", "reused_reference": True})
    return paths, results


def fit(recipes, config, backbone, device, root, reference, prior, identity):
    selections = []
    for protocol in config["protocols"]:
        local = protocol_config(config, protocol)
        rows = {split: records(local["data"], split) for split in ("train", "val")}
        banks = {split: shared_bank(reference, prior, protocol, split, values, local, device)
                 for split, values in rows.items()}
        for name, recipe in recipes.items():
            local_config = local | {"training": local["training"] | {"loss": recipe["loss"], "ms_direction": "symmetric"}}
            for seed in config["seeds"]:
                output = root / protocol / f"{name}_s{seed}"
                seed_all(seed)
                heads = OIPAHeads(backbone.output_dim, recipe["architecture"], config["oipa"]["rank"]).to(device)
                variant = {"study": identity, "protocol": protocol, "method": name, "seed": seed, **recipe}
                log({"stage": "training_started", "protocol": protocol, "method": name, "seed": seed})
                checkpoint, epoch, validation = train_one(heads, banks, rows, local_config, device, variant, output, seed)
                path = output / "selection.json"
                immutable(path, variant | {"selected_on": "val", "epoch": epoch, "validation": validation,
                                           "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)})
                log({"stage": "validated", "protocol": protocol, "method": name, "seed": seed, "epoch": epoch,
                     "recall1": {d: validation[d]["Recall@1"] for d in DIRECTIONS}})
                selections.append(path)
    return selections


def test(config, backbone, device, root, reference, prior, paths, gate):
    results = []
    for protocol in config["protocols"]:
        local = protocol_config(config, protocol)
        rows = records(local["data"], "test")
        validation = records(local["data"], "val")
        bank = shared_bank(reference, prior, protocol, "test", rows, local, device)
        diag = prepare_diagnostics(config, protocol, rows, root)
        if diag["spec"] != read_json(reference / "diagnostics" / "specs" / f"{protocol}.json"):
            raise ValueError(f"Diagnostic cohort changed for {protocol}")
        diag_bank = shared_bank(reference, prior, protocol, "diagnostic", diag["records"], local, device)
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
                    raise ValueError(f"Test identity changed: {result_path}")
            else:
                result = {"split": "test", "metric_units": "fraction", "selection_sha256": gate[str(path)],
                          "protocol": protocol, "method": selection["method"], "seed": selection["seed"],
                          "coarse_only": score(heads, bank, rows, device), "hyperparameter_search_on_test": False,
                          "efficiency": profile(backbone, heads, validation, local, device)}
                write_json(result_path, result)
                log({"stage": "test_result", "protocol": protocol, "method": result["method"], "seed": result["seed"],
                     "recall1": {d: result["coarse_only"][d]["Recall@1"] for d in DIRECTIONS}})
            results.append(result)
            evaluate_diagnostics(diag, embeddings(heads, diag_bank, device), selection, gate[str(path)], root)
    return results


def factorial_report(results, protocols, seeds):
    methods = ("B1_Place", "B1_MS", "AB_Place", "AB_MS")
    expected = {(p, m, s) for p in protocols for m in methods for s in seeds}
    by_key = {(r["protocol"], r["method"], r["seed"]): r["coarse_only"] for r in results}
    if len(results) != len(expected) or by_key.keys() != expected:
        raise ValueError("Incomplete or duplicated factorial runs")
    report = {}
    for p in protocols:
        means = {method: {d: {f"Recall@{k}": summarize([
            by_key[p, method, s][d][f"Recall@{k}"] for s in seeds
        ]) for k in KS} for d in DIRECTIONS} for method in methods}
        contrasts = {name: {d: {f"Recall@{k}": summarize([
            100 * (by_key[p, first, s][d][f"Recall@{k}"] - by_key[p, second, s][d][f"Recall@{k}"])
            for s in seeds
        ]) for k in KS} for d in DIRECTIONS} for name, (first, second) in CONTRASTS.items()}
        contrasts["interaction"] = {d: {f"Recall@{k}": summarize([
            100 * ((by_key[p, "AB_MS", s][d][f"Recall@{k}"] - by_key[p, "AB_Place", s][d][f"Recall@{k}"])
                   - (by_key[p, "B1_MS", s][d][f"Recall@{k}"] - by_key[p, "B1_Place", s][d][f"Recall@{k}"]))
            for s in seeds
        ]) for k in KS} for d in DIRECTIONS}
        report[p] = {"mean_and_sd_fractions": means, "paired_seed_differences_pp": contrasts}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    study = load_config(args.config)
    reference = Path(study["reference"]).resolve()
    root = Path(study["output"]).resolve()
    original = read_json(reference / "run.json")
    prior, config = original["identity"], original["config"]
    if digest(config) != prior["config"] or read_json(reference / "completion.json")["study"] != prior:
        raise ValueError("Reference run identity changed")
    if root == reference:
        raise ValueError("Use an independent output directory")
    with run_lock(root):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        device = torch.device(config["device"])
        pe, _ = official_modules()
        from benchmark import __file__ as diagnostic_source
        from pytorch_metric_learning import __version__ as metric_learning_version

        native = {p.name: sha256(p) for p in sorted(Path(pe.__file__).parent.iterdir())
                  if p.is_file() and p.suffix in (".py", ".gz")}
        diagnostics = {name: sha256(path) for name, path in config["diagnostics"].items()}
        diagnostics["code"] = sha256(diagnostic_source)
        if (runtime(device) != prior["runtime"] or diagnostics != prior["diagnostics"]
                or digest(native) != prior["native"] or checkpoint_identity(config["checkpoint"]) != prior["weights"]):
            raise ValueError("Reference runtime, PE checkpoint, source or diagnostics changed")
        source = Path(__file__).resolve().parents[1]
        code = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*.py"))}
        old_paths, old_results = reference_b1(reference, prior, config)
        identity = {"config": digest(study), "code": digest(code), "reference_study": prior,
                    "reference_results": digest(old_results),
                    "metric_learning_version": metric_learning_version}
        immutable(root / "run.json", {"identity": identity, "config": study})
        for protocol in config["protocols"]:
            verify_formal(protocol_config(config, protocol)["data"])
        seed_all(42)
        backbone = PECore(config["checkpoint"]).to(device)
        paths = fit(study["recipes"], config, backbone, device, root, reference, prior, identity)
        gate = {str(path): sha256(path) for path in [*old_paths, *paths]}
        validation = [selection | {"method": "B1_Place" if path in old_paths else selection["method"],
                                   "coarse_only": selection["validation"]}
                      for path in [*old_paths, *paths] for selection in [read_json(path)]]
        write_json(root / "factorial_validation_report.json", {
            "aggregate": factorial_report(validation, config["protocols"], config["seeds"])
        })
        immutable(root / "test_freeze.json", {"study": identity, "selections": gate})
        log({"stage": "test_frozen", "new_selections": len(paths), "reused_selections": len(old_paths)})
        new_results = test(config, backbone, device, root, reference, prior, paths, gate)
        results = [*old_results, *new_results]
        write_json(root / "factorial_test_report.json", {
            "per_seed": results, "aggregate": factorial_report(results, config["protocols"], config["seeds"]),
        })
        write_json(root / "completion.json", {"status": "complete", "study": identity,
                                             "new_runs": len(new_results), "reference_runs": len(old_results)})
        log({"stage": "complete", "new_runs": len(new_results), "reference_runs": len(old_results)})


if __name__ == "__main__":
    main()
