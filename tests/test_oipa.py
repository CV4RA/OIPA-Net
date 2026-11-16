import pytest
import torch
from torch.nn import functional as F

from place_retrieval.oipa import OIPAHeads, hard_negative_loss, objective
from place_retrieval.study.oipa_loss import factorial_report


def test_oipa_toggles_and_independent_encoding():
    bank = {"global": F.normalize(torch.rand(4, 8), dim=-1),
            "tokens": torch.randn(4, 3, 8), "mask": torch.ones(4, 3, dtype=torch.bool)}
    for method, vision, text in (("B0", False, False), ("B1", False, False), ("B2", True, False),
                                 ("B3", False, True), ("B4", False, False), ("AB", True, True),
                                 ("Ours", True, True)):
        heads = OIPAHeads(8, method, rank=4)
        assert heads.uses_tokens("image") == vision and heads.uses_tokens("text") == text
        if vision:
            torch.nn.init.normal_(heads.vision_pool.up.weight)
        if text:
            torch.nn.init.normal_(heads.text_pool.up.weight)
        for modality in ("image", "text"):
            embedding, _ = heads(bank, modality)
            standalone, _ = heads({k: v[1:2] for k, v in bank.items()}, modality)
            torch.testing.assert_close(embedding[1:2], standalone)
            if method == "B0":
                assert torch.equal(embedding, bank["global"])


@pytest.mark.parametrize("method, loss_name", (
    ("B1", "place"), ("B2", "place"), ("B3", "place"), ("B4", "place"),
    ("Ours", "place"), ("AB", "place"), ("B1", "ms"), ("AB", "ms"),
))
def test_adapted_methods_have_finite_gradients(method, loss_name):
    batch = {m: {"global": F.normalize(torch.randn(4, 8), dim=-1),
                 "tokens": torch.randn(4, 3, 8),
                 "mask": torch.tensor([[True, True, False]] * 4)} for m in ("image", "text")}
    rows = {m: [{"place_id": p} for p in (1, 1, 2, 2)] for m in batch}
    options = {"loss": loss_name, "ms_direction": "symmetric",
               "temperature": 0.07, "tolerance_m": 25, "view_weight": 0.1,
               "evidence_weight": 0.1, "negative_weight": 0.1, "hard_k": 4, "hard_margin": 0.05}
    heads = OIPAHeads(8, method, rank=4)
    loss = objective(heads, batch, rows, options)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())


def test_miner_never_uses_same_place_or_excluded_pairs():
    images = F.normalize(torch.randn(3, 8), dim=-1)
    texts = F.normalize(torch.randn(2, 8), dim=-1)
    all_positives = torch.ones(3, 2, dtype=torch.bool)
    assert hard_negative_loss(images, texts, all_positives, all_positives, 2, 0.05, 0.07) == 0
    positives = torch.tensor([[True, False], [True, False], [False, True]])
    assert hard_negative_loss(images, texts, positives, positives, 2, 0.05, 0.07) == 0
    allowed = torch.ones_like(positives)
    assert hard_negative_loss(images, texts, positives, allowed, 2, 0.05, 0.07) > 0


def test_factorial_contrasts_are_seed_paired():
    scores = {"B1_Place": (0.20, 0.30), "B1_MS": (0.30, 0.32),
              "AB_Place": (0.23, 0.34), "AB_MS": (0.37, 0.37)}
    results = [{"protocol": "P0", "method": method, "seed": seed,
                "coarse_only": {d: {f"Recall@{k}": value for k in (1, 5, 10)} for d in ("T2I", "I2T")}}
               for method, values in scores.items() for seed, value in zip((42, 43), values)]
    report = factorial_report(results, ("P0",), (42, 43))["P0"]["paired_seed_differences_pp"]
    assert report["ms_effect_B1"]["T2I"]["Recall@1"]["mean"] == pytest.approx(6)
    assert report["interaction"]["I2T"]["Recall@1"]["mean"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="Incomplete"):
        factorial_report(results[:-1], ("P0",), (42, 43))
