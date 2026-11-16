"""Coarse embeddings and the fixed pretrained BLIP cross-encoder."""
import hashlib
import time
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

from .data import ImageDataset
from .io import digest, log, model_identity, read_json, run_lock, sha256, write_json
from .models import load_blip, raw_features, tokenize_texts


@torch.inference_mode()
def encode_split(backbone, processor, heads, rows, c, device):
    backbone.eval()
    heads.eval()
    options = c["evaluation"]
    loader = DataLoader(ImageDataset(rows["image"], processor, c["data"]),
                        batch_size=options["encode_batch_size"], num_workers=options["workers"],
                        **({"multiprocessing_context": "spawn"} if options["workers"] else {}))
    images = []
    for pixels in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = raw_features(backbone, pixels.to(device), "image").float()
        images.append(raw.cpu())
    texts = []
    size = options["encode_batch_size"]
    for start in range(0, len(rows["text"]), size):
        values = [r["text"] for r in rows["text"][start:start + size]]
        tokens = tokenize_texts(backbone, processor, values, c["training"], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = raw_features(backbone, tokens, "text").float()
        texts.append(raw.cpu())
    # Match the archived inference path: project FP32 features in 2,048-sample chunks.
    features = {}
    for modality, chunks in (("image", images), ("text", texts)):
        raw = torch.cat(chunks)
        features[modality] = torch.cat([
            heads.encode(raw[start:start + 2048].to(device), modality).cpu()
            for start in range(0, len(raw), 2048)
        ])
    return features


@torch.inference_mode()
def score_blip(candidates, c, device, score_path=None):
    backbone, processor = load_blip(c["models"]["blip"], device)
    rows = candidates["records"]
    options = c["evaluation"]
    loader = DataLoader(ImageDataset(rows["image"], processor, c["data"]),
                        batch_size=options["visual_batch_size"], num_workers=options["workers"],
                        **({"multiprocessing_context": "spawn"} if options["workers"] else {}))
    visual = None
    bank_root = Path(c.get("visual_cache", str(Path(c["output"]) / "visual_cache")))
    identity = {"model": model_identity(c["models"]["blip"]),
                "images": digest([(r["image_id"], r.get("sha256")) for r in rows["image"]]),
                "device_compute": "bf16" if device.type == "cuda" else "float32"}
    bank_root = bank_root / digest(identity)
    with run_lock(bank_root):
        done = bank_root / "manifest.json"
        path = bank_root / "visual.npy"
        if done.exists():
            metadata = read_json(done)
            if metadata["identity"] != identity or sha256(path) != metadata["sha256"]:
                raise ValueError("Visual cache identity or hash changed")
        else:
            offset = 0
            for pixels in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    value = backbone.vision_model(pixel_values=pixels.to(device), return_dict=True).last_hidden_state
                if visual is None:
                    visual = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                                       shape=(len(rows["image"]), *value.shape[1:]))
                visual[offset:offset + len(value)] = value.half().cpu().numpy()
                offset += len(value)
            visual.flush()
            del visual
            write_json(done, {"identity": identity, "sha256": sha256(path)})
        visual = np.load(path, mmap_mode="r")
    size = options["pair_batch_size"]
    started = time.monotonic()
    pairs = candidates["pairs"]
    results = []
    offset = 0
    scores = None
    if score_path is not None:
        score_path = Path(score_path)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path = score_path.with_suffix(".json")
        identity = {"visual": identity, "pairs_sha256": hashlib.sha256(pairs.contiguous().numpy()).hexdigest(),
                    "texts": digest([row["text"] for row in rows["text"]]),
                    "length": options["text_length"], "batch_size": size}
        if progress_path.is_file():
            progress = read_json(progress_path)
            if progress["identity"] != identity:
                raise ValueError(f"BLIP pair score identity changed: {score_path}")
            offset = progress["scored"]
            scores = np.load(score_path, mmap_mode="r+")
            if scores.shape != (len(pairs),) or scores.dtype != np.float32 or not 0 <= offset <= len(pairs):
                raise ValueError("BLIP pair score cache is incomplete or malformed")
        else:
            scores = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(len(pairs),))
        if offset < len(pairs):
            chunks = [processor.tokenizer(
                [r["text"] for r in rows["text"][i:i + 1024]], padding="max_length",
                max_length=options["text_length"], truncation=True, return_tensors="pt",
            ) for i in range(0, len(rows["text"]), 1024)]
            token_bank = {k: torch.cat([chunk[k] for chunk in chunks]).to(device)
                          for k in ("input_ids", "attention_mask")}
    for start in range(offset, len(pairs), size):
        selected = pairs[start:start + size]
        if scores is None:
            texts = [rows["text"][i]["text"] for i in selected[:, 0].tolist()]
            tokens = processor.tokenizer(texts, padding="max_length", max_length=options["text_length"],
                                         truncation=True, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
            images = torch.from_numpy(np.array(visual[selected[:, 1].numpy()], copy=True)).to(device).float()
        else:
            tokens = {k: v.index_select(0, selected[:, 0].to(device)) for k, v in token_bank.items()}
            unique, inverse = selected[:, 1].unique(sorted=True, return_inverse=True)
            images = torch.from_numpy(np.array(visual[unique.numpy()], copy=True)).to(device).float()
            images = images.index_select(0, inverse.to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            attention = torch.ones(images.shape[:2], dtype=torch.long, device=device)
            hidden = backbone.text_encoder(**tokens, encoder_hidden_states=images,
                                           encoder_attention_mask=attention,
                                           return_dict=True).last_hidden_state[:, 0]
            logits = backbone.itm_head(hidden).float()
        result = (logits[:, 1] - logits[:, 0]).cpu()
        completed = start + len(result)
        if scores is None:
            results.append(result)
        else:
            scores[start:completed] = result.numpy()
        if start % (size * 250) == 0 or completed == len(pairs):
            if scores is not None:
                scores.flush()
                write_json(progress_path, {"identity": identity, "scored": completed})
            log({"stage": "rerank", "completed": completed,
                 "total": len(pairs), "seconds": time.monotonic() - started})
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(results) if scores is None else torch.from_numpy(np.array(scores, copy=True))
