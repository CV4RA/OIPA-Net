# PE-Core baselines

The Phase 1 recipes use the same PE-Core-L/14-336 weights, 336-pixel official transform, 32-position tokenizer and native image/text projections. B0 performs zero-shot cosine retrieval with the official normalized embeddings. B1 freezes both PE towers and learns two zero-initialized residual projections with the AcrossPlace bidirectional multi-positive place loss. B1 uses the existing sampler, temperature, AdamW schedule and validation checkpoint criterion. The highest weaker-direction validation Recall@100 wins, followed by weaker-direction Recall@1 and the earlier epoch.

Both recipes rank samples against the full opposite-modality gallery. They use the released rule-valid descriptions, target observations and same-place positives. CUDA encoding uses BF16 autocast and writes FP32 embeddings to the existing coarse metric. `val_result.json` and `test_result.json` report T2I and I2T Recall as fractions.

## Environment and weights

Run from the repository root with the PyTorch environment described in the [AcrossPlace reproduction guide](REPRODUCE.md). Install the optional PE dependencies and the official source at the tested commit. Installing the source with `--no-deps` leaves the AcrossPlace PyTorch and Transformers pins in effect.

```sh
python -m pip install -e ".[dev,pe]"
python -m pip install --no-deps "perception_models @ git+https://github.com/facebookresearch/perception_models.git@3e352cca660658d4b5c90f42a7808b11469e4c66"
hf download facebook/PE-Core-L14-336 PE-Core-L14-336.pt --revision bafb0f76541d399057e980a25947f67acec76575 --local-dir artifacts/PE-Core-L14-336
```

The checkpoint occupies 2.68 GB. The experiment entry point verifies its published SHA256 `0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8` before passing the file to Meta's loader. [Official source](https://github.com/facebookresearch/perception_models/tree/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder) · [Official checkpoint](https://huggingface.co/facebook/PE-Core-L14-336/blob/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt)

## Data and execution

Point `data.formal` and `data.image_root` at your exact metadata and local GSV-Cities image tree. See [DATA.md](DATA.md) for the directory structure. In the current workspace the verified P0 formal directory is `../outputs/gsv_full_62279/metadata/combined/formal`. The published `place_retrieval.study.protocol` command reproduces P1 from that P0 source. The configs use the original P0/P1 fingerprints and canonical image prefix.

```sh
place-retrieval verify-data --config configs/pe_core_p0.yaml --check-images
python -m place_retrieval.study.pe_core zero-shot --config configs/pe_core_p0.yaml
python -m place_retrieval.study.pe_core train --config configs/pe_core_p0.yaml --seed 42
```

Use `configs/pe_core_p1.yaml` for P1. B0 writes under `runs/pe_core/P0/b0` and B1 under `runs/pe_core/P0/b1_s42` by default. For the other B1 seeds use `--seed 43` and `--seed 44`. Add `--resume` to a B1 command to continue from the most recent completed epoch with the same config. B1 saves the trained heads separately from the official weights. Model, formal-data, config and checkpoint identities are recorded in `evaluation/selection.json`.

Test is an explicit command run after validation selection. In an approved benchmark run use the matching config, baseline, seed and selection file.

```sh
python -m place_retrieval.study.pe_core test --config configs/pe_core_p0.yaml --baseline B0 --selection runs/pe_core/P0/b0/evaluation/selection.json
python -m place_retrieval.study.pe_core test --config configs/pe_core_p0.yaml --baseline B1 --seed 42 --selection runs/pe_core/P0/b1_s42/evaluation/selection.json
```

## Verification

`python -m pytest -q tests/test_pe_core.py` exercises both baselines on synthetic formal splits, checks checkpoint and selection hashes, and verifies that validation and training access their intended splits. With the optional Meta source installed, it also checks the real preprocessing and tokenizer and compares the wrapper against Meta's forward path using a tiny checkpoint in native format.

Local verification used Windows, PyTorch 2.4.1+cu124 and the pinned Meta source and BPE vocabulary. The retrieval suite passed all 27 tests, including the native PE check and a B1 interruption and resume. The frozen diagnostics suite passed all 10 tests.

All files in the local P0 manifest passed hash verification. The release protocol generator reproduced P1 in `runs/P1/protocol_p1/formal` with the archived identity, counts and `valid.jsonl` SHA256 `342773ec22321a62171f69b193bb69341d6889ffde5444de1a13c885030540cf`. Reproduce the metadata check from the repository root after substituting the local P0 path.

```sh
python -m place_retrieval.study.protocol --source datasets/formal --destination runs/P1/protocol_p1/formal
```

Benchmark imagery and the full 2.68 GB PE checkpoint are needed for measured recall and efficiency. The Phase 1 verification used fixtures and metadata only.
