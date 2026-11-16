import pytest
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel

from place_retrieval.oipa_hf import FrozenHF
from place_retrieval.study.oipa_backbones import IndexedFeature, merge_rows


@pytest.mark.parametrize("kind", ("siglip", "blip", "clip"))
def test_global_matches_native_retrieval_with_and_without_oipa_tokens(tiny_config, tmp_path, kind):
    if kind == "clip":
        config = CLIPConfig(
            text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                           "num_attention_heads": 2, "image_size": 16, "patch_size": 8},
            projection_dim=12,
        )
        directory = tmp_path / "clip"
        CLIPModel(config).save_pretrained(directory)
    else:
        directory = tiny_config["models"][kind]
    backbone = FrozenHF(directory, torch.device("cpu"), {"text_length": 8})
    processor = backbone.processor()
    image = processor.preprocess_image(Image.new("RGB", (16, 16), "tan")).unsqueeze(0)
    text = processor.tokenize(["independent view"])
    with torch.inference_mode():
        for modality, values, tokens in (("image", image, 4), ("text", text, 8)):
            base = backbone.encode(values, modality, False, 2)
            augmented = backbone.encode(values, modality, True, 2)
            torch.testing.assert_close(base["global"], augmented["global"], rtol=0, atol=0)
            assert augmented["tokens"].shape == (1, tokens, backbone.output_dim)
            assert augmented["mask"].any()


def test_shared_bank_rows_remain_in_protocol_order_and_split():
    image = {"image_id": "image", "path": "/image", "sha256": "hash"}
    a = {"text_id": "a", "text": "first"}
    b = {"text_id": "b", "text": "second"}
    groups = {
        ("P0", "train"): {"image": [image], "text": [a, b]},
        ("P1", "val"): {"image": [dict(image, split="val")], "text": [b]},
    }
    unique, _, views = merge_rows(groups)
    assert unique == {"image": [image], "text": [a, b]}
    view = IndexedFeature(torch.tensor([42, 43]), views["P1", "val"]["text"])
    assert len(view) == 1 and view[:].tolist() == [43]
