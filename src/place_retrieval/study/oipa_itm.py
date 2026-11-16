"""Apply validation-selected BLIP ITM fusion to frozen OIPA coarse checkpoints."""
import argparse
import os
import statistics
from pathlib import Path

import torch

from ..data import records, verify_formal
from ..encoding import score_blip
from ..io import digest, load_config, log, model_identity, read_json, run_lock, save_torch, sha256, write_json
from ..metrics import evaluate, select_fusion, shortlist
from ..training import selection_key
from .oipa import embeddings, immutable, protocol_config, runtime, selected_heads, summarize
from .oipa_backbones import RECIPES, load_backbone, shared_features
from .oipa_loss import DIRECTIONS, KS, shared_bank


def recipes_by_validation(groups, scope):
    if scope == "full":
        return tuple(groups)
    if scope == "pe_b1_ms":
        return ("B1_MS",)
    if scope != "validation_best":
        raise ValueError(f"Unknown reranking scope: {scope}")
    return (max(groups, key=lambda method: tuple(
        statistics.mean(values) for values in zip(*(selection_key(row["validation"]) for row in groups[method]))
    )),)


def read_frozen(root, protocol, dirname, freeze_root):
    path = root / protocol / dirname / "selection.json"
    recorded = read_json(freeze_root / "test_freeze.json")["selections"][str(path)]
    selection = read_json(path)
    if sha256(path) != recorded or selection["selected_on"] != "val" or selection["protocol"] != protocol:
        raise ValueError(f"Coarse selection changed: {path}")
    return path, recorded, selection


def make_plan(config, study):
    reference = Path(study["reference"]).resolve()
    pe = Path(study["pe_loss_study"]).resolve()
    other = Path(study["backbone_study"]).resolve()
    if read_json(pe / "completion.json")["status"] != "complete":
        raise ValueError("PE loss comparison must finish first")
    backbone_recipes = {}
    if study["scope"] != "pe_b1_ms":
        if read_json(other / "completion.json")["status"] != "complete":
            raise ValueError("Backbone comparison must finish first")
        run = read_json(other / "run.json")
        backbone_recipes = {name: recipe | {"identity": run["study"]["models"][name]}
                            for name, recipe in run["config"]["backbones"].items()}
    result = {}
    for protocol in config["protocols"]:
        jobs = []
        for name in ("PE", *backbone_recipes):
            groups = {}
            for method in RECIPES:
                roots = (reference if method == "B1_Place" else pe) if name == "PE" else other / name
                dirname = "B1" if name == "PE" and method == "B1_Place" else method
                freeze = roots if name == "PE" else other
                groups[method] = [read_frozen(roots, protocol, f"{dirname}_s{seed}", freeze)
                                  for seed in config["seeds"]]
            selected = recipes_by_validation({m: [item[2] for item in values] for m, values in groups.items()},
                                             study["scope"])
            for method in selected:
                for path, checksum, selection in groups[method]:
                    jobs.append({"backbone": name, "method": method, "seed": selection["seed"],
                                 "selection": str(path), "selection_sha256": checksum})
        result[protocol] = jobs
    return result, backbone_recipes


def build_shortlists(config, study, prior, rows, protocol, split, jobs, hf, root, device):
    cache = root / "shortlists" / protocol / split
    files = []
    for name in dict.fromkeys(job["backbone"] for job in jobs):
        group = [job for job in jobs if job["backbone"] == name]
        missing = [job for job in group if not candidate_path(cache, job).is_file()]
        if missing:
            if name == "PE":
                bank = shared_bank(Path(study["reference"]).resolve(), prior, protocol, split, rows, config, device)
                width = bank["image"]["global"].shape[-1]
            else:
                weights = model_identity(hf[name]["directory"])
                if weights != hf[name]["identity"]:
                    raise ValueError(f"Frozen backbone files changed: {name}")
                model = load_backbone(hf[name], device)
                scratch = root / "scratch" / f"{protocol}_{split}_{name}.pt"
                feature_identity = {"backbone": hf[name], "model": weights,
                                    "protocol": protocol, "split": split, "reference": prior}
                bank = shared_features(model, {split: rows}, config, device, scratch, feature_identity)[split]
                width = model.output_dim
            for job in missing:
                if sha256(job["selection"]) != job["selection_sha256"]:
                    raise ValueError("Frozen coarse selection changed")
                _, heads = selected_heads(job["selection"], width, config["oipa"]["rank"], device)
                candidate = shortlist(embeddings(heads, bank, device), rows, study["evaluation"]["candidates"])
                del candidate["records"]
                save_torch(candidate_path(cache, job), {"job": job, "candidates": candidate})
                del heads, candidate
            del bank
            if name != "PE":
                del model
                scratch.unlink()
                torch.cuda.empty_cache()
        files.extend((candidate_path(cache, job), job) for job in group)
    return files


def candidate_path(cache, job):
    return cache / f"{job['backbone']}_{job['method']}_s{job['seed']}.pt"


def candidates_from(path, job):
    if sha256(job["selection"]) != job["selection_sha256"]:
        raise ValueError("Frozen coarse selection changed")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if saved["job"] != job:
        raise ValueError(f"Reranking shortlist identity changed: {path}")
    return saved["candidates"]


def image_major_codes(pairs, n_text):
    return pairs[:, 1] * n_text + pairs[:, 0]


def score_union(files, rows, config, study, model_hashes, protocol, split, root, device):
    n_text = len(rows["text"])
    codes = torch.cat([image_major_codes(candidates_from(path, job)["pairs"], n_text)
                       for path, job in files]).unique(sorted=True)
    pairs = torch.stack((codes.remainder(n_text), codes.div(n_text, rounding_mode="floor")), dim=1)
    log({"stage": "itm_union_started", "protocol": protocol, "split": split, "runs": len(files),
         "unique_pairs": len(pairs)})
    source = Path(study["published_visual_cache"][protocol])
    vision_identity = {"model": model_hashes,
                       "images": digest([(r["image_id"], r["sha256"]) for r in rows["image"]]),
                       "device_compute": "bf16"}
    cache = source if (source / digest(vision_identity) / "manifest.json").is_file() else root / "visual_cache"
    rerank = config | {"models": {"blip": study["blip"]}, "output": str(root),
                       "visual_cache": str(cache), "evaluation": study["evaluation"]}
    return codes, score_blip({"pairs": pairs, "records": rows}, rerank, device,
                             root / "itm" / f"{protocol}_{split}.npy")


def protocol_split(config, study, prior, protocol, split, jobs, hf, root, device, model_hashes, frozen):
    local = protocol_config(config, protocol)
    rows = records(local["data"], split)
    files = build_shortlists(local, study, prior, rows, protocol, split, jobs, hf, root, device)
    codes, scores = score_union(files, rows, local, study, model_hashes, protocol, split, root, device)
    results = []
    for path, job in files:
        candidate = candidates_from(path, job)
        candidate["records"] = rows
        keys = image_major_codes(candidate["pairs"], len(rows["text"]))
        positions = torch.searchsorted(codes, keys)
        if not torch.equal(codes[positions], keys):
            raise ValueError("ITM union lost a coarse candidate")
        itm = scores[positions]
        out = root / "results" / protocol / path.stem
        coarse = evaluate(candidate, candidate["coarse_scores"], 0)
        if split == "val":
            chosen, grid = select_fusion(candidate, itm, study["evaluation"]["fusion_weights"], split)
            selection = {"job": job, "selected_on": "val", "weight": chosen["global_weight"],
                         "validation": chosen, "grid": grid}
            immutable(out / "selection.json", selection)
        else:
            selection_path = out / "selection.json"
            if sha256(selection_path) != frozen[str(selection_path)]:
                raise ValueError("ITM fusion selection changed after validation")
            selection = read_json(selection_path)
            if selection["job"] != job or selection["selected_on"] != "val":
                raise ValueError("ITM test requires a matching validation selection")
            result = {"job": job, "protocol": protocol, "split": "test", "metric_units": "fraction",
                      "coarse_only": coarse, "itm_only": evaluate(candidate, itm, 0),
                      "coarse_plus_itm": evaluate(candidate, itm, selection["weight"]),
                      "fusion_weight": selection["weight"], "selection_sha256": frozen[str(selection_path)]}
            immutable(out / "test_result.json", result)
            results.append(result)
        log({"stage": "itm_evaluated", "protocol": protocol, "split": split, **job})
    return results


def summary(results):
    groups = {}
    for r in results:
        job = r["job"]
        groups.setdefault((r["protocol"], job["backbone"], job["method"]), []).append(r)
    return [{"protocol": p, "backbone": b, "method": m, "metrics": {mode: {
        d: {f"Recall@{k}": summarize([r[mode][d][f"Recall@{k}"] for r in rows]) for k in KS}
        for d in DIRECTIONS
    } for mode in ("coarse_only", "itm_only", "coarse_plus_itm")}} for (p, b, m), rows in groups.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    study = load_config(args.config)
    root = Path(study["output"]).resolve()
    original = read_json(Path(study["reference"]) / "run.json")
    config, prior = original["config"], original["identity"]
    if digest(config) != prior["config"]:
        raise ValueError("Reference configuration changed")
    with run_lock(root):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        device = torch.device(config["device"])
        if runtime(device) != prior["runtime"]:
            raise ValueError("Reference runtime differs")
        plan, hf = make_plan(config, study)
        model_hashes = model_identity(study["blip"])
        source = Path(__file__).resolve().parents[1]
        code = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*.py"))}
        identity = {"study": study, "reference": prior, "code": digest(code), "blip": model_hashes, "jobs": plan}
        immutable(root / "run.json", identity)
        for protocol in plan:
            verify_formal(protocol_config(config, protocol)["data"])
        for protocol, jobs in plan.items():
            protocol_split(config, study, prior, protocol, "val", jobs, hf, root, device, model_hashes, {})
        selections = sorted((root / "results").glob("*/*/selection.json"))
        if len(selections) != sum(map(len, plan.values())):
            raise ValueError("Incomplete ITM validation selections")
        frozen = {str(path): sha256(path) for path in selections}
        immutable(root / "test_freeze.json", {"run": digest(identity), "selections": frozen})
        results = []
        for protocol, jobs in plan.items():
            results.extend(protocol_split(config, study, prior, protocol, "test", jobs, hf,
                                          root, device, model_hashes, frozen))
        write_json(root / "itm_test_report.json", {"aggregate": summary(results), "per_seed": results})
        write_json(root / "completion.json", {"status": "complete", "test_runs": len(results)})
        log({"stage": "complete", "test_runs": len(results)})


if __name__ == "__main__":
    main()
