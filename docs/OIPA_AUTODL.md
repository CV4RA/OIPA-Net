# OIPA-Net AutoDL study

The study runs B0 once and B1–B4 with seeds 42, 43 and 44 on each of P0 and P1. B1–B4 share the frozen official PE-Core towers, place sampler, bidirectional multi-positive objective and optimizer schedule. B1 trains residual projection heads. B2 adds learned pooling of 4×4 visual regions and same-place cross-view consistency. B3 adds contextual token pooling and trains its weights from support across independent target views in the training split. B4 adds a different-place hard-negative term ranked by cross-modal score and same-modality prototype similarity. The `Ours` method enables all three modules and is available as a separate configuration choice.

The feature banks store official projected global embeddings in FP32 and auxiliary tokens in FP16 after BF16 autocast. Image bytes are checked against the formal records during encoding. The frozen backbone executes once per bank, independently for each observation or caption. The adapters train in FP32 on cached features. The validation criterion is the weaker-direction Recall@100, followed by weaker-direction Recall@1 and the earlier epoch. Epoch checkpoints include optimizer, scheduler and RNG state. Re-running the same command resumes the study.

```sh
export PYTHONPATH=src:diagnostics:artifacts/perception_models
/root/stet-r1-venv/bin/python -u -m place_retrieval.study.oipa --config configs/oipa_autodl.yaml
```

After all validation selections exist, `test_freeze.json` binds their hashes and the test stage runs the released full-gallery metric. The released diagnostics define the D1 source/target comparisons, D2 fixed and nested galleries, and D3 rule and human evidence groups. D3 human labels enter only the P1 test diagnostic. Results include per-seed and aggregate Recall, synchronized batch-one model-only latency and CUDA peak allocation. Zero-shot and adapted reports are separate. This study specifies coarse retrieval.

## Place and Multi-Similarity comparison

`configs/oipa_loss_autodl.yaml` specifies the PE-Core 2×2 comparison between projection heads alone and OIPA A+B, each trained with Place or symmetric Multi-Similarity. The original B1+Place runs and PE feature banks are reused. The C module is disabled in this matrix. Validation selection and the three-epoch schedule follow the reference run. The factorial report includes paired seed differences and the loss-by-module interaction in percentage points.

```sh
bash scripts/run_oipa_loss_autodl.sh
```

`configs/oipa_backbones_autodl.yaml` repeats the 2×2 comparison for the released SigLIP2 Base/16-384, BLIP ITM Base COCO and CLIP ViT-B/16 checkpoints. Each checkpoint supplies its native image transform, tokenizer and projected global features. A+B uses its projected contextual text tokens and a 4×4 grid of local image tokens. Training reads only the rows in the current protocol's training split. Content-identical items can share frozen features between P0 and P1. Each backbone has an additional fixed zero-shot reference. The runner freezes all 78 validation choices before testing and keeps the temporary banks on the AutoDL data disk.

```sh
bash scripts/run_oipa_backbones_autodl.sh
```

These comparisons reuse the released test cohorts after the original PE ablation study. Interpret their paired deltas as follow-up comparisons on the same benchmark.

## BLIP ITM reranking

`configs/oipa_itm_autodl.yaml` specifies which frozen coarse recipes enter reranking. The default picks one recipe per backbone and protocol using the mean of the three validation selection keys. All three seeds of each selected recipe are evaluated. Each direction retrieves its own Top-100 candidates. A fixed BLIP ITM head scores the union of distinct candidate pairs, with resumable score files shared across recipes. The release's validation grid selects the weight in `ITM logit difference + weight × cosine`. The grid includes zero for ITM-only. The runner freezes every validation weight before evaluating the test split and records coarse, ITM-only and validation-selected fusion separately.

## PE-Core Q/V LoRA with residual heads and MS

Run `bash scripts/run_pe_lora_autodl.sh --verify-inputs` to verify both formal manifests and locate their images. Run `bash scripts/run_pe_lora_autodl.sh --smoke` to measure a synthetic full-batch forward and backward pass with the official checkpoint. Run `bash scripts/run_pe_lora_autodl.sh` for the six training runs and full-gallery evaluation. The recipe uses BF16 tower computation, FP32 residual heads, and activation checkpointing in the last four blocks of each tower. The base weights are frozen and rank-8 LoRA updates the native fused Q/V projections in those blocks. The original P0/P1 cohorts, batch sampler, symmetric MS objective, three-epoch schedule and seeds match the existing `B1_MS` control. All six LoRA selections are frozen before test scoring.

The runner resumes from the last saved step in `study_pe_lora_v1`, including optimizer, scheduler and RNG state. The paired comparison against the completed `B1_MS` control is written to `coarse_test_report.json`. It records per-seed metrics and means, sample standard deviations and paired changes for both directions at R@1/5/10.

```sh
bash scripts/run_oipa_itm_autodl.sh
```

## SF-XL replication

The SF-XL study uses the supplied `formal/P0` and `formal/P1-region` manifests. P0 is place-disjoint. P1-region holds out a geographic band within San Francisco. Its protocol name is retained in every artifact. The test sets contain 1,100 places with 2,200 gallery images and 3,285 descriptions for P0, and 1,600 places with 3,200 images and 4,971 descriptions for P1-region.

Stage the formal directories under `/root/autodl-tmp/sfxl_acrossplace/formal` and extract the verified image bundle under `/root/autodl-tmp/sfxl_acrossplace/small`. The configurations bind the exact formal fingerprints. The matching recipes use the MSLS schedule, seeds 42–44, validation-only checkpoint and fusion selection, and full-gallery bidirectional recall. The D2 run uses 500 fixed query places and a matched gallery of 1,000 places under each protocol. D3 uses the rule-status diagnostic.

```sh
bash scripts/run_sfxl_autodl.sh
```

The queue runs PE-Core B0–B4 and the Place/MS factorial, PE-Core Q/V LoRA, the CLIP/BLIP/SigLIP 2 frozen controls and factorial, Q/V LoRA for those three backbones, and Top-100 BLIP ITM reranking for the validation-best frozen recipe per backbone and protocol. Outputs and logs use the `study_sfxl*` prefix under `/root/autodl-tmp/OIPA-Net`. Each successful stage writes a completion record and a zero exit marker. Rerunning the queue skips completed stages and resumes interrupted training using its existing checkpoints.
