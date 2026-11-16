import torch

from place_retrieval.data import records
from place_retrieval.encoding import score_blip
from place_retrieval.io import read_json, write_json
from place_retrieval.study.oipa_itm import image_major_codes, recipes_by_validation


def test_resumable_blip_scores_match_released_scorer(tiny_config, tmp_path):
    rows = records(tiny_config["data"], "val")
    pairs = torch.tensor([[0, 0], [1, 0], [2, 1], [0, 2]])
    c = tiny_config
    c["evaluation"]["pair_batch_size"] = 2
    inputs = {"pairs": pairs, "records": rows}
    device = torch.device("cpu")
    expected = score_blip(inputs, c, device)
    path = tmp_path / "scores.npy"
    scores = score_blip(inputs, c, device, path)
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    progress = read_json(path.with_suffix(".json"))
    write_json(path.with_suffix(".json"), progress | {"scored": 2})
    torch.testing.assert_close(score_blip(inputs, c, device, path), expected, rtol=0, atol=0)
    assert image_major_codes(pairs, len(rows["text"])).tolist() == [0, 1, 6, 8]


def test_reranking_scope_uses_validation_coverage():
    def row(t2i, i2t, r1):
        return {"validation": {"T2I": {"Recall@100": t2i, "Recall@1": r1},
                               "I2T": {"Recall@100": i2t, "Recall@1": r1}}}

    groups = {"B1_Place": [row(0.9, 0.7, 0.3)], "B1_MS": [row(0.75, 0.76, 0.2)]}
    assert recipes_by_validation(groups, "validation_best") == ("B1_MS",)
    assert recipes_by_validation(groups, "full") == ("B1_Place", "B1_MS")
