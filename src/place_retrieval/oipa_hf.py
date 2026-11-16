"""Native AcrossPlace HF backbone embeddings with spatial and text tokens for OIPA."""
import torch
from torch import nn
from torch.nn import functional as F

from .models import context_length, load_siglip, raw_features


class HFInputs:
    def __init__(self, processor, text_length):
        self.processor = processor
        self.text_length = text_length
        self.special_ids = tuple(getattr(processor.tokenizer, "all_special_ids", ()))

    def preprocess_image(self, image):
        return self.processor(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"][0]

    def tokenize(self, texts):
        return self.processor(text=texts, padding="max_length", max_length=self.text_length,
                              truncation=True, return_tensors="pt")


class FrozenHF(nn.Module):
    def __init__(self, directory, device, options):
        super().__init__()
        self.model, processor, _ = load_siglip(
            directory, {"adapt_towers": [], "residual_heads": False}, device,
        )
        self.model.eval()
        config = self.model.config
        self.kind = config.model_type
        if self.kind == "blip":
            self.output_dim = config.image_text_hidden_size
        elif self.kind == "clip":
            self.output_dim = config.projection_dim
        else:
            self.output_dim = config.text_config.hidden_size
        self.inputs = HFInputs(processor, context_length(self.model, options))

    def processor(self):
        return self.inputs

    def encode(self, inputs, modality, include_tokens, grid):
        if not include_tokens:
            return {"global": F.normalize(raw_features(self.model, inputs, modality).float(), dim=-1)}
        image = modality == "image"
        tower = self.model.vision_model if image else (
            self.model.text_encoder if self.kind == "blip" else self.model.text_model
        )
        captured = []

        def capture(module, args, output):
            captured.append(output.last_hidden_state)

        with tower.register_forward_hook(capture):
            global_features = F.normalize(raw_features(self.model, inputs, modality).float(), dim=-1)
        hidden, = captured
        if image:
            patch = self.model.config.vision_config.patch_size
            height, width = (n // patch for n in inputs.shape[-2:])
            if self.kind in ("clip", "blip"):
                hidden = hidden[:, 1:]
            if hidden.shape[1] != height * width:
                raise ValueError("Unexpected native image token layout")
            regions = hidden.transpose(1, 2).reshape(len(hidden), hidden.shape[-1], height, width)
            tokens = F.adaptive_avg_pool2d(regions, (grid, grid)).flatten(2).transpose(1, 2)
            if self.kind == "clip":
                tokens = self.model.visual_projection(tower.post_layernorm(tokens))
            elif self.kind == "blip":
                tokens = self.model.vision_proj(tokens)
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if self.kind == "clip":
                tokens = self.model.text_projection(hidden)
            elif self.kind == "blip":
                tokens = self.model.text_proj(hidden)
            else:
                tokens = tower.head(hidden)
            ids = inputs["input_ids"]
            mask = inputs["attention_mask"].bool() if "attention_mask" in inputs else torch.ones_like(ids, dtype=torch.bool)
            if self.inputs.special_ids:
                mask = mask & ~torch.isin(ids, ids.new_tensor(self.inputs.special_ids))
            mask[:, 0] |= ~mask.any(dim=1)
        if tokens.shape[-1] != global_features.shape[-1]:
            raise ValueError("Native token projection does not match the retrieval dimension")
        return {"global": global_features, "tokens": F.normalize(tokens.float(), dim=-1).half(), "mask": mask}
