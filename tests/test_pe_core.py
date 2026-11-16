"""CPU coverage for PE-Core input parity and the two coarse baseline workflows."""
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from place_retrieval import pe_core, training
from place_retrieval.io import read_json, sha256, write_json
from place_retrieval.oipa_features import native_features
from place_retrieval.study import pe_core as study


class TinyCLIP(nn.Module):
    image_size = 16
    context_length = 32
    output_dim = 3

    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 3, bias=False)
        self.text = nn.Embedding(128, 3)

    @classmethod
    def from_config(cls, name, pretrained=False, checkpoint_path=None):
        assert name == pe_core.MODEL_NAME and pretrained
        model = cls()
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return model

    def encode_image(self, image, normalize=False):
        assert normalize and not self.training
        return F.normalize(self.visual(image.mean(dim=(2, 3))), dim=-1)

    def encode_text(self, text, normalize=False):
        assert normalize and not self.training and text.shape[1] == self.context_length
        return F.normalize(self.text(text).mean(dim=1), dim=-1)


@pytest.fixture
def tiny_pe_config(tiny_config, tmp_path, monkeypatch):
    from torchvision.transforms import Compose, Resize, ToTensor

    def get_image_transform(size):
        return Compose([Resize((size, size)), ToTensor()])

    def get_text_tokenizer(length):
        return lambda texts: torch.tensor([
            [ord(char) % 128 for char in text[:length].ljust(length)] for text in texts
        ])

    monkeypatch.setattr(pe_core, "official_modules", lambda: (
        SimpleNamespace(CLIP=TinyCLIP),
        SimpleNamespace(get_image_transform=get_image_transform, get_text_tokenizer=get_text_tokenizer),
    ))
    weights = tmp_path / "PE-Core-L14-336.pt"
    torch.save(TinyCLIP().state_dict(), weights)
    monkeypatch.setattr(pe_core, "WEIGHTS_SHA256", sha256(weights))
    monkeypatch.setitem(study.PROTOCOL_FINGERPRINTS, "P0", tiny_config["data"]["fingerprint"])
    tiny_config.update(protocol="P0", models={"pe_core": str(weights)})
    # The native context length is authoritative even with inherited config values.
    tiny_config["training"]["text_length"] = 8
    return tiny_config


def test_official_processing_and_checkpoint(tmp_path, monkeypatch):
    pe = pytest.importorskip("core.vision_encoder.pe")
    from core.vision_encoder import transforms

    name = pe_core.MODEL_NAME
    assert pe.PE_VISION_CONFIG[name].image_size == 336
    assert pe.PE_TEXT_CONFIG[name].context_length == 32
    full_processor = pe_core.PECoreProcessor(336, 32)
    image = Image.new("P", (19, 33))
    expected_pixels = transforms.get_image_transform(336)(image)
    assert torch.equal(full_processor.preprocess_image(image), expected_pixels)
    texts = ["A stone arch", "blue door " * 40]
    expected_tokens = transforms.get_text_tokenizer(32)(texts)
    assert torch.equal(full_processor.tokenize(texts), expected_tokens)
    assert expected_tokens.shape == (2, 32)
    assert expected_tokens[1, -1] == full_processor.tokenizer.eot_token_id
    restored = pickle.loads(pickle.dumps(full_processor))
    assert torch.equal(restored.preprocess_image(image), expected_pixels)
    assert torch.equal(restored.tokenize(texts), expected_tokens)

    monkeypatch.setitem(pe.PE_VISION_CONFIG, name, replace(
        pe.PE_VISION_CONFIG[name], image_size=28, width=32, layers=1, heads=4, output_dim=16,
    ))
    monkeypatch.setitem(pe.PE_TEXT_CONFIG, name, replace(
        pe.PE_TEXT_CONFIG[name], width=32, layers=1, heads=4, output_dim=16,
    ))
    reference = pe.CLIP.from_config(name).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.1)
    checkpoint = tmp_path / "official_format.pt"
    torch.save({"state_dict": {"module." + k: v for k, v in reference.state_dict().items()}}, checkpoint)
    wrapper = pe_core.PECore(checkpoint)
    assert not any(parameter.requires_grad for parameter in wrapper.parameters())
    pixels = wrapper.processor().preprocess_image(Image.new("RGB", (19, 33), "tan")).unsqueeze(0)
    with torch.inference_mode():
        image_features, text_features, _ = reference(pixels, expected_tokens)
        torch.testing.assert_close(wrapper.encode_image(pixels), image_features)
        torch.testing.assert_close(wrapper.encode_text(expected_tokens), text_features)
        image_bank = native_features(wrapper, pixels, "image", True, 2)
        text_bank = native_features(wrapper, expected_tokens, "text", True, 2)
        torch.testing.assert_close(image_bank["global"], image_features)
        torch.testing.assert_close(text_bank["global"], text_features)
        assert image_bank["tokens"].shape == (1, 4, 16)
        assert text_bank["tokens"].shape == (2, 32, 16)
        assert not text_bank["mask"][:, 0].any()
        assert not text_bank["mask"].gather(1, expected_tokens.argmax(dim=1, keepdim=True)).any()


def test_config_guard_and_checkpoint_checksum(tiny_pe_config):
    with pytest.raises(ValueError, match="published formal fingerprint"):
        study.configure(dict(tiny_pe_config, protocol="P1"), "B0", None)
    path = Path(tiny_pe_config["models"]["pe_core"])
    path.write_bytes(b"other checkpoint")
    with pytest.raises(ValueError, match="published .* SHA256"):
        pe_core.checkpoint_identity(path)


@pytest.mark.parametrize("baseline", ["B0", "B1"])
def test_baseline_validation_and_guarded_test(tiny_pe_config, baseline, monkeypatch):
    c = study.configure(tiny_pe_config, baseline, None)
    accessed = []

    def track(load_records):
        def tracked(data, split):
            accessed.append(split)
            return load_records(data, split)
        return tracked

    monkeypatch.setattr(study, "records", track(study.records))
    monkeypatch.setattr(training, "records", track(training.records))
    if baseline == "B1":
        original_log = training.log

        def interrupt_after_epoch(event, path):
            if event.get("epoch") == 1 and "validation" in event:
                raise RuntimeError("simulated interruption after epoch one")
            original_log(event, path)

        with monkeypatch.context() as patch:
            patch.setattr(training, "log", interrupt_after_epoch)
            with pytest.raises(RuntimeError, match="simulated interruption"):
                study.validate(c)
        last = Path(c["output"]) / "coarse/last.pt"
        assert torch.load(last, weights_only=True)["epoch"] == 1
        path = study.validate(c, resume=True)
        assert torch.load(last, weights_only=True)["epoch"] == 2
    else:
        path = study.validate(c)
    assert set(accessed) == ({"val"} if baseline == "B0" else {"train", "val"})
    selection = read_json(path)
    report = read_json(path.parent / "val_result.json")
    assert report["selection"] == selection and report["split"] == "val"
    assert selection["selected_on"] == "val"
    if baseline == "B1":
        checkpoint = Path(c["output"]) / "coarse" / "best.pt"
        state = torch.load(checkpoint, weights_only=True)
        assert state["format"] == pe_core.CHECKPOINT_FORMAT and not state["lora"]
        assert all(value.abs().sum() > 0 for value in state["heads"].values())
        with checkpoint.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="unchanged validation selection"):
            study.evaluate_test(c, path)
        with checkpoint.open("r+b") as stream:
            stream.truncate(checkpoint.stat().st_size - len(b"changed"))
    else:
        assert selection["coarse_checkpoint"] is None and selection["coarse_epoch"] == 0
    write_json(path, dict(selection, selected_on="test"))
    with pytest.raises(ValueError, match="unchanged validation selection"):
        study.evaluate_test(c, path)
    assert "test" not in accessed
    write_json(path, selection)
    with pytest.raises(ValueError, match="unchanged validation selection"):
        study.evaluate_test(dict(c, evaluation=dict(c["evaluation"], encode_batch_size=1)), path)
    result = study.evaluate_test(c, path)
    assert result["split"] == "test"
    assert result["regime"] == ("zero_shot" if baseline == "B0" else "fine_tuned")
    assert accessed[-1] == "test"
    assert set(result["coarse_only"]) == {"T2I", "I2T"}
    assert all(0 <= values[f"Recall@{k}"] <= 1 for values in result["coarse_only"].values() for k in (1, 5, 10))
    assert study.evaluate_test(c, path) == result
    assert accessed.count("test") == 1
