"""Extract immutable PE global, regional visual, and contextual text features."""
import hashlib
import io

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import image_path
from .io import log


class CheckedImages(Dataset):
    def __init__(self, rows, data, processor):
        self.rows = rows
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = image_path(row["path"], self.data)
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
            raise ValueError(f"Frozen image hash mismatch: {path}")
        with Image.open(io.BytesIO(encoded)) as image:
            return self.processor.preprocess_image(image)


def native_features(backbone, inputs, modality, include_tokens, grid):
    model = backbone.model
    image = modality == "image"
    encode = backbone.encode_image if image else backbone.encode_text
    if not include_tokens:
        return {"global": encode(inputs).float()}
    captured = []

    def capture(module, arguments, output):
        captured.append(output)

    hook = model.visual.ln_post if image else model.ln_final
    with hook.register_forward_hook(capture):
        global_features = encode(inputs).float()
    hidden, = captured
    if image:
        vision = model.visual
        if vision.use_cls_token:
            hidden = hidden[:, 1:]
        patches = model.image_size // vision.patch_size
        regions = hidden.transpose(1, 2).reshape(len(inputs), hidden.shape[-1], patches, patches)
        regions = F.adaptive_avg_pool2d(regions, (grid, grid)).flatten(2).transpose(1, 2)
        projected = regions @ vision.proj
        mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=inputs.device)
    else:
        projected = hidden @ model.text_projection
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        mask = (positions > 0) & (positions < inputs.argmax(dim=-1)[:, None])
        mask[:, 0] = ~mask.any(dim=1)
    return {"global": global_features, "tokens": F.normalize(projected.float(), dim=-1).half(), "mask": mask}


@torch.inference_mode()
def encode_feature_bank(processor, rows, config, device, encode):
    options = config["evaluation"]
    size = options["encode_batch_size"]
    workers = options["workers"]
    loader = DataLoader(CheckedImages(rows["image"], config["data"], processor), batch_size=size,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        **({"multiprocessing_context": "spawn", "prefetch_factor": 1} if workers else {}))
    def collect(batches, modality):
        chunks = {}
        completed = 0
        for step, batch in enumerate(batches, start=1):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                values = encode(batch.to(device), modality)
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite {modality} features: {key}")
                chunks.setdefault(key, []).append(value.cpu())
            completed += len(values["global"])
            if step % 250 == 0 or completed == len(rows[modality]):
                log({"stage": "encode", "modality": modality, "completed": completed, "total": len(rows[modality])})
        return {key: torch.cat(values) for key, values in chunks.items()}

    texts = (processor.tokenize([row["text"] for row in rows["text"][offset:offset + size]])
             for offset in range(0, len(rows["text"]), size))
    return {"image": collect(loader, "image"), "text": collect(texts, "text")}


def encode_bank(backbone, rows, config, device):
    return encode_feature_bank(
        backbone.processor(), rows, config, device,
        lambda batch, modality: native_features(backbone, batch, modality, True, config["oipa"]["region_grid"]),
    )
