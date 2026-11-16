# OIPA-Net

**Observation-Informed Place Alignment for vision–language place retrieval**

[Project website](https://oipa-net.ornatesparrow2.chatgpt.site/) · [Dataset archives on Google Drive](https://drive.google.com/drive/folders/1AJGTxcmHjqnc90nchQL6EeYIPeEXKuDR) · [Results](results/oipa_net/) · [Data manifest](data-release/release-manifest.json)

OIPA-Net studies whether a description grounded in one street-level observation can retrieve an independent view of the same place. It pairs a modular dual encoder with a shared evaluation across GSV-Cities, MSLS-derived, and SF-XL-derived benchmarks. Image-gallery items and text queries are encoded independently. Place identities define all positive matches in both retrieval directions.

## 1. What is implemented

| Component | Implementation |
| --- | --- |
| PE-Core-L/14-336 baseline | [Official PE wrapper](src/place_retrieval/pe_core.py), native preprocessing and tokenizer, original projections and strict checkpoint verification |
| Observation modules | [View-stable visual aggregation, evidence-aware token weighting, and place-aware hard negatives](src/place_retrieval/oipa.py) |
| Feature extraction | [PE regional and token features](src/place_retrieval/oipa_features.py) and [matched Hugging Face backbones](src/place_retrieval/oipa_hf.py) |
| Place supervision | [Bidirectional multi-positive contrastive and symmetric Multi-Similarity objectives](src/place_retrieval/training.py) |
| Low-rank adaptation | [Q/V LoRA](src/place_retrieval/pe_lora.py), [PE-Core runner](src/place_retrieval/study/pe_lora.py), and [CLIP, BLIP, SigLIP 2 runner](src/place_retrieval/study/backbone_lora.py) |
| Evaluation | [Full-gallery Recall@1/5/10](src/place_retrieval/metrics.py), [D1–D3](src/place_retrieval/study/oipa_diagnostics.py), and [Top-100 BLIP ITM reranking](src/place_retrieval/study/oipa_itm.py) |

The visual branch pools local regions into a learned residual to the global image representation. The text branch learns token weights from agreement across same-place training observations. The hard-negative term selects visually or semantically similar different-place items while retaining all same-place positives. Inference uses one input per encoder and ranks ℓ₂-normalized descriptors by cosine similarity. Q/V LoRA with residual heads and symmetric Multi-Similarity is a separate matched adaptation experiment.

## 2. Environment

The archived studies use Linux, Python 3.11, PyTorch 2.5.1, CUDA with BF16 support, and seeds 42, 43, and 44. Use an NVIDIA driver compatible with the selected CUDA wheel. The PE study probes the PyTorch scaled-dot-product attention backend and records whether the fused Flash Attention backend is available. The recipes use PyTorch's built-in SDPA path.

```bash
git clone https://github.com/CV4RA/OIPA-Net.git
cd OIPA-Net
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev,pe,analysis]'
mkdir -p artifacts
git clone https://github.com/facebookresearch/perception_models.git artifacts/perception_models
git -C artifacts/perception_models checkout 3e352cca660658d4b5c90f42a7808b11469e4c66
hf download facebook/PE-Core-L14-336 PE-Core-L14-336.pt \
  --revision bafb0f76541d399057e980a25947f67acec76575 \
  --local-dir artifacts/PE-Core-L14-336
python scripts/download_models.py --model all --output artifacts
export PYTHONPATH="$PWD/src:$PWD/diagnostics:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Pretrained weights are downloaded from their publishers. The PE study verifies the checkpoint before encoding. The model-download script pins the Hugging Face revisions and verifies converted model tensors against their source. Budget separate storage for images, weights, feature banks, and checkpoints.

## 3. Prepare and verify the benchmarks

| Dataset | Formal protocols | Upstream imagery |
| --- | --- | --- |
| GSV-Cities | P0 place-disjoint, P1 city holdout | [GSV-Cities](https://github.com/amaralibey/gsv-cities) · [Original protocol bundle](https://drive.google.com/file/d/13639QpS-GX7ZeibqKaxd45dHxIo5Cf5W/view?usp=drive_link) |
| MSLS-derived | P0 place-disjoint, P1 city holdout | [Mapillary Street-Level Sequences](https://www.mapillary.com/dataset/places) |
| SF-XL-derived | P0 place-disjoint, P1-region buffered region holdout | [SF-XL from CosPlace](https://github.com/gmberton/CosPlace) |

Download the three metadata archives from the [release folder](https://drive.google.com/drive/folders/1AJGTxcmHjqnc90nchQL6EeYIPeEXKuDR). The folder uses the maintainer's Google Drive access settings. The archives contain frozen split manifests, source descriptions, image metadata, and bidirectional query and gallery lists. Street-view image bytes are acquired separately under the original dataset terms. Each image hash is tied to the experiment version.

```bash
download_dir="$HOME/Downloads/oipa-net"
(cd "$download_dir" && sha256sum -c /absolute/path/to/OIPA-Net/data-release/SHA256SUMS)
mkdir -p datasets
for dataset in gsv-cities msls-derived sf-xl-derived; do
  tar -xzf "$download_dir/oipa-net-$dataset-formal.tar.gz" -C datasets
done

dataset=gsv-cities                         # or msls-derived / sf-xl-derived
export RUN_ROOT="$PWD/runs/$dataset"
image_root="$PWD/datasets/$dataset/images" # use existing GSVCities root if available
python scripts/configure_oipa.py --dataset "$dataset" --data-root datasets \
  --image-root "$image_root" --model-root artifacts --run-root "$RUN_ROOT"
```

Each protocol resolves to `datasets/<dataset>/formal/<protocol>/manifest.json`. The configuration script takes the archived recipe and substitutes file paths. It keeps the protocol fingerprints, seeds, method settings, and validation rules. For an existing frozen image tree, verify it directly:

```bash
python scripts/stage_oipa_images.py --config "$RUN_ROOT/configs/oipa.yaml"
```

For an upstream archive whose member paths match the manifest, use `--archive /path/to/upstream.zip`. For an extracted upstream tree, use `--upstream-root /path/to/upstream`. The staging command uses `archive_member` when present, checks bytes against SHA-256, and writes to the per-dataset image tree in the generated configuration. For GSV-Cities, the original `GSVCities` directory can be used directly. Use the exact released image version if a member is absent or its bytes differ.

## 4. Run training and evaluation

From the repository root with the environment above, run the studies in dependency order. For a new dataset, repeat the configuration step before running these commands.

```bash
python -m place_retrieval.study.oipa --config "$RUN_ROOT/configs/oipa.yaml"
if [ -f "$RUN_ROOT/configs/oipa_loss.yaml" ]; then
  python -m place_retrieval.study.oipa_loss --config "$RUN_ROOT/configs/oipa_loss.yaml"
fi
python -m place_retrieval.study.oipa_backbones --config "$RUN_ROOT/configs/oipa_backbones.yaml"
python -m place_retrieval.study.pe_lora --config "$RUN_ROOT/configs/pe_lora.yaml" --verify-inputs
python -m place_retrieval.study.pe_lora --config "$RUN_ROOT/configs/pe_lora.yaml" --smoke
python -m place_retrieval.study.pe_lora --config "$RUN_ROOT/configs/pe_lora.yaml"
python -m place_retrieval.study.backbone_lora --config "$RUN_ROOT/configs/backbone_lora.yaml"
python -m place_retrieval.study.oipa_itm --config "$RUN_ROOT/configs/oipa_itm.yaml"
```

| Recipe | Definition |
| --- | --- |
| B0 | PE-Core zero-shot |
| B1 | Frozen PE-Core with residual heads and place contrastive loss |
| B2 | B1 + view-stable visual aggregation (A) |
| B3 | B1 + evidence-aware semantic weighting (B) |
| B4 | B1 + place-aware hard-negative learning (C) |
| B1-MS / AB-Place / AB-MS | Matched head-only and joint A+B loss controls |
| Q/V LoRA + heads + MS | Backbone adaptation under the matched MS objective |
| Coarse / +ITM | Frozen dual-encoder ranking and a separate shortlist reranking experiment |

The first stage runs B0–B4. GSV-Cities uses `oipa_loss` for additional loss controls. The two derived datasets include those controls in their main recipes. Baselines and adaptations share the formal image and text galleries. A test query is counted as a hit when at least one same-place item is returned among the first `k` ranks. Every training seed uses validation-only selection. The runs save configuration, source and checkpoint provenance, per-seed selection files, a test freeze, and aggregate results. To resume, re-run the same command with the same paths and recipe. A run with a different identity needs a new output directory.

D1 substitutes the source and independent target views in a paired evaluation. D2 expands the gallery with controlled candidate sets. D3 measures differences between evidence groups using archived diagnostic plans; GSV-Cities also includes an analysis-only human visibility audit. Report the paired source/target recall alongside the gap; they convey different information. Keep full-gallery zero-shot, adapted, and shortlist-reranked scores in separate result groups.

## 5. Inspect the released results

The [full-gallery table](results/oipa_net/matched_full_gallery_recall.csv) contains three datasets, both formal protocols, both retrieval directions, four backbones, and zero-shot, residual-head MS, and Q/V-LoRA MS rows. The [geographic D1 table](results/oipa_net/geographic_d1_recall.csv) provides detailed independent-target results for GSV-Cities, MSLS-derived, and SF-XL-derived. See the project site for an interactive view of the full-gallery table.

Example, PE-Core on **P0**, Recall@1 in percent (T2I / I2T):

| Dataset | Zero-shot | Q/V LoRA + residual heads + MS |
| --- | ---: | ---: |
| GSV-Cities | 7.24 / 9.21 | 19.27 / 24.02 |
| MSLS-derived | 3.46 / 2.50 | 6.05 / 5.51 |
| SF-XL-derived | 6.18 / 7.55 | 12.64 / 18.17 |

## 6. Tests and source

```bash
python -m pytest -q tests diagnostics/test_benchmark.py
python -m http.server 8000 --directory website
```

Open `http://localhost:8000` for a local copy of the project site. Synthetic tests cover place labels, exact ranking and tie handling, token aggregation, hard-negative masking, and reranking. The PE smoke command requires the official checkpoint and a GPU. See [third-party attribution and licenses](THIRD_PARTY_NOTICES.md) before using the datasets or weights.

