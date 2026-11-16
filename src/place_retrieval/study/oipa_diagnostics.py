"""Apply the released AcrossPlace diagnostics to validation-frozen PE models."""
from pathlib import Path

import numpy as np
import torch

from ..io import digest, read_json, sha256, write_json


def prepare_diagnostics(config, protocol, test_rows, root):
    from benchmark import dataset_spec, feature_records, lines

    protocol_root = Path(config["protocols"][protocol])
    places = [p for p in lines(protocol_root / "valid.jsonl") if p["split"] == "test"]
    ids = {p["place_id"] for p in places}
    uncertain = {p["place_id"]: p["texts"] for p in lines(
        Path(config["protocols"]["P0"]) / "valid_plus_uncertain_same_places.jsonl"
    ) if p["place_id"] in ids}
    archive = read_json(config["diagnostics"]["archive_plan"])
    if archive["protocols"][protocol]["fingerprint"] != read_json(protocol_root / "manifest.json")["fingerprint"]:
        raise ValueError("Diagnostic protocol and published archive disagree")
    spec = dataset_spec(places, uncertain, archive["gallery_seeds"], **config["diagnostics"].get("sizes", {}))
    records = feature_records(spec)
    result = {"spec": spec, "records": records, "archive_sha256": sha256(config["diagnostics"]["archive_plan"])}
    if protocol == "P1" and "human_review" in config["diagnostics"]:
        review = read_json(config["diagnostics"]["human_review"])
        annotations = review["rows"]
        if len(annotations) != 240 or any(r["place_id"] not in ids for r in annotations):
            raise ValueError("The published P1 human review must align to its test cohort")
        images = {r["image_id"]: r for r in records["image"]}
        images.update({r["image_id"]: r for r in test_rows["image"]})
        texts = {r["text_id"]: r for r in records["text"]}
        background = []
        paired = {p["place_id"]: p["text"] for p in spec["pairs"]}
        for place in sorted(places, key=lambda p: p["place_id"]):
            row = paired.get(place["place_id"], min(place["texts"], key=lambda t: t["text_id"]))
            text = dict(row, place_id=place["place_id"], split="test")
            texts[text["text_id"]] = text
            background.append(text)
        for row in annotations:
            texts[row["text_id"]] = {k: row[k] for k in ("text", "text_id", "place_id")} | {"split": "test"}
        records["image"], records["text"] = list(images.values()), list(texts.values())
        result["human"] = {"annotations": annotations, "background": background, "gallery": test_rows["image"],
                           "sha256": sha256(config["diagnostics"]["human_review"])}
        result["canonical_prefix"] = config["data"]["canonical_prefix"]
    write_json(root / "diagnostics" / "specs" / f"{protocol}.json", spec)
    return result


def human_diagnostic(diag, features):
    from benchmark import retrieval_hits
    from human_retrieval import replacement_hits

    human = diag["human"]
    annotations = human["annotations"]
    background = human["background"]
    gallery = human["gallery"]
    text_index = {r["text_id"]: i for i, r in enumerate(diag["records"]["text"])}
    image_index = {r["path"]: i for i, r in enumerate(diag["records"]["image"])}
    place_index = {r["place_id"]: i for i, r in enumerate(background)}
    query_text = features["text"][[text_index[r["text_id"]] for r in annotations]].numpy()
    gallery_images = features["image"][[image_index[r["path"]] for r in gallery]].numpy()
    background_features = features["text"][[text_index[r["text_id"]] for r in background]].numpy()
    t2i = retrieval_hits(query_text, gallery_images, [r["place_id"] for r in annotations],
                         [r["place_id"] for r in gallery])
    per_item = []
    for i, row in enumerate(annotations):
        paths = [diag["canonical_prefix"] + "/" + p.split("GSVCities/", 1)[-1] for p in row["target_images"]]
        query_image = features["image"][[image_index[p] for p in paths]].numpy()
        scores = query_image @ background_features.T
        positive = query_image @ query_text[i]
        i2t = np.asarray([replacement_hits(score, hit, place_index[row["place_id"]])
                          for score, hit in zip(scores, positive)]).mean(0).tolist()
        per_item.append({"audit_id": row["audit_id"], "place_id": row["place_id"],
                         "T2I": t2i[i], "I2T_controlled": i2t})
    groups = {}
    for field in ("source_correctness", "target_visibility", "place_specificity", "rule_status"):
        groups[field] = {}
        for label in sorted({row[field] for row in annotations}):
            selected = [hits for row, hits in zip(annotations, per_item) if row[field] == label]
            groups[field][label] = {"n": len(selected), **{
                direction: {f"Recall@{k}": float(v) for k, v in zip(
                    (1, 5, 10), np.array([r[direction] for r in selected]).mean(0))}
                for direction in ("T2I", "I2T_controlled")
            }}
    return {"review_sha256": human["sha256"], "groups": groups, "per_item": per_item,
            "image_gallery": len(gallery), "controlled_text_gallery_places": len(background),
            "interpretation": "Descriptive subgroup associations from a single reviewer."}


def evaluate_diagnostics(diag, features, selection, selection_hash, root):
    from benchmark import evaluate_model

    seed = selection["seed"]
    name = selection["method"] if seed is None else f"{selection['method']}_s{seed}"
    protocol = selection["protocol"]
    mode = "zero_shot" if seed is None else "fine_tuned"
    output = root / "diagnostics" / mode
    path = output / f"{name}_{protocol}.json"
    identity = {"selection_sha256": selection_hash, "archive_sha256": diag["archive_sha256"],
                "spec_digest": digest(diag["spec"])}
    if path.exists():
        if read_json(path)["identity"] != identity:
            raise ValueError(f"Diagnostic identity changed: {path}")
        return
    bank = {"features": features, "records": diag["records"], "identity": identity}
    variant = {"id": name, "family": selection["method"], "training_seed": seed}
    archived = evaluate_model(variant, protocol, diag["spec"], bank, identity, output)
    report = {"identity": identity, "method": selection["method"], "protocol": protocol, "seed": seed,
              "D1": archived["A1"], "D2": {f"matched_{len(next(iter(diag['spec']['matched_1000'].values())))}": archived["A2"],
                                         "gallery_scale": archived["A3"]},
              "D3": {"rule_quality": archived["A4"]}}
    if "human" in diag:
        report["D3"]["human_evidence"] = human_diagnostic(diag, features)
    write_json(path, report)
