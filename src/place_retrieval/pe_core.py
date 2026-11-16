"""Use Meta's native PE-Core model, checkpoint loader and input transforms."""
from functools import cached_property
from pathlib import Path

from torch import nn

from .io import sha256

MODEL_NAME = "PE-Core-L14-336"
WEIGHTS_SHA256 = "0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8"
CHECKPOINT_FORMAT = "pe-core-place-v1"


def checkpoint_identity(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PE-Core checkpoint missing: {path}. See docs/PE_CORE.md.")
    checksum = sha256(path)
    if checksum != WEIGHTS_SHA256:
        raise ValueError(f"{path} does not match the published {MODEL_NAME} checkpoint SHA256")
    return {"model": MODEL_NAME, "sha256": checksum}


def official_modules():
    try:
        from core.vision_encoder import pe, transforms
    except ImportError as error:
        raise ImportError("Install the optional Meta Perception Encoder source as in docs/PE_CORE.md") from error
    return pe, transforms


class PECoreProcessor:
    def __init__(self, image_size, context_length):
        self.image_size = image_size
        self.context_length = context_length

    @cached_property
    def image_transform(self):
        return official_modules()[1].get_image_transform(self.image_size)

    @cached_property
    def tokenizer(self):
        return official_modules()[1].get_text_tokenizer(self.context_length)

    def __getstate__(self):
        # The official transform includes a lambda. Rebuild it in spawned workers.
        return {"image_size": self.image_size, "context_length": self.context_length}

    def preprocess_image(self, image):
        return self.image_transform(image)

    def tokenize(self, texts):
        return self.tokenizer(texts)


class PECore(nn.Module):
    """A frozen pair of independent, officially projected PE towers."""

    def __init__(self, checkpoint_path=None):
        super().__init__()
        pe, _ = official_modules()
        self.model = pe.CLIP.from_config(
            MODEL_NAME, pretrained=True,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        )
        self.requires_grad_(False).eval()

    @property
    def output_dim(self):
        return self.model.output_dim

    def processor(self):
        return PECoreProcessor(self.model.image_size, self.model.context_length)

    def encode_image(self, pixels):
        return self.model.encode_image(pixels, normalize=True)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens, normalize=True)


class IdentityHeads(nn.Module):
    """Pass through PE's normalized embeddings for the zero-shot baseline."""

    def encode(self, value, modality):
        return value.float()
