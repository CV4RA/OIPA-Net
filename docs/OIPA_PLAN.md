# OIPA-Net implementation plan

## Repository audit

Audit target is AcrossPlace release 1.0.0 at commit `2a66c25f3a241c2af93eae9dbde658f3e19cde67`. The parent workspace contains older experimental snapshots and local copies of the formal metadata. This work uses the public release as its code base.

| Component | Existing path and behavior | Phase 1 integration |
| --- | --- | --- |
| Model | `models.py` loads CLIP, SigLIP2 and BLIP, with model-specific feature projections, optional LoRA and residual heads. | Add a native Meta PE loader and reuse the residual heads for B1. |
| Training | `training.py` uses place-grouped batches and symmetric multi-positive loss. The GPS mask needs coordinates in both modalities, while the released text rows omit GPS. The best checkpoint is selected by weaker-direction validation Recall@100, then Recall@1. | Reuse the sampler, objective and selection. Keep the PE towers frozen in eval mode and serialize the learned heads and continuation state. |
| Encoding | `encoding.py` encodes each modality separately and returns ordered features aligned to formal records. | Pass raw PIL inputs and text strings to the official PE transform and tokenizer. Use the projected features from the official towers. |
| Evaluation | `metrics.py` ranks all samples against the full opposite-modality gallery and defines positives by place ID. `evaluation.py` handles BLIP validation selection and guarded test evaluation. | Use the same full-gallery coarse metric for B0 and B1, with a separate validation selection for each run. |
| Data and study | `data.py` verifies the manifest and reads the original split and target-image rows. The published study planner, workers and test driver freeze recipe and selection identities. | Add separate PE run configs and a coarse-only entry point. Bind results to the published P0/P1 fingerprints without changing the released plans, data or test definitions. |
| Diagnostics | `diagnostics/` contains the fixed transfer, gallery-scale and language-evidence analyses. | In Phase 4, supply aligned PE/OIPA features to the existing diagnostics and preserve all cohort and sampling rules. |

Meta's native `PE-Core-L14-336` uses a 336-pixel bilinear squash with RGB conversion and normalization, a 32-position BPE tokenizer, and 1,024-dimensional projected image and text embeddings. Use its existing model configuration, EOT pooling, learned projections and checkpoint loader. B0 uses the official normalized outputs. B1 adds zero-initialized residual transformations to both projected embeddings and optimizes the published place loss. Temperature and GPS tolerance follow the existing study recipe.

## Phase 1 implementation order

1. Add the optional PE integration with an explicit checkpoint path, published weight checksum and local encoding interfaces.
2. Add P0/P1 recipes for B0 validation and B1 training, plus an explicit test command that consumes a validation-generated selection. Bind the selection to the model, checkpoint, data, inference settings and baseline. Report coarse metrics in fractions.
3. Verify transforms, native token length, independent encoding and checkpoint loading. Exercise both baselines, validation, restoration, continuation and split isolation on a small fixture. Run the existing regression suite and verify the available P0/P1 metadata identities.
4. Document reproducible commands and verification status. Full training and changes to frozen protocols require approval.

## Phases 2–5

1. **View-stable aggregation.** Expose projected global features and visual patch tokens from the same query image. Learn bounded pooling over patch features and fuse a residual into the global embedding. Use independently observed same-place images within training batches for consistency. Evaluate query invariance to gallery composition by comparing its embedding before and after gallery replacement.
2. **Evidence-aware weighting.** Expose contextual text tokens and mask padding and special tokens. Train a token pooling head with the place objective and cross-observation agreement on training data. Keep the D3 human audit and rule groups for post-selection diagnostics. Check for weight collapse and report text truncation under the fixed PE context window.
3. **Place-aware mining.** Build the same-place positive mask before mining. Mine high-scoring eligible different-place images/texts within a training batch, preserving the GPS exclusion policy. Validate loss and gradients when a batch has no eligible negatives and when several views share one place.
4. **Training and selection.** Add independent A/B/C switches and shared adapter/LoRA settings for all adapted methods. Select LoRA targets against native PE's fused attention and MLP implementation. Match batch composition, optimization budget, precision, seeds and checkpoint selection. Enable gradient checkpointing where trainable tower layers require it. Check deterministic replay across an epoch boundary.
5. **Experiments.** Freeze B0, B1, B2 (A), B3 (B), B4 (C) and the full model before testing. Use seeds 42, 43 and 44 for adapted runs and one result per protocol for the fixed B0. Report bidirectional Recall@1/5/10, D1 source-to-target transfer on the released matched sets, D2 on the released gallery sizes and draws, and D3 on the released evidence cohorts. Record total and trainable parameters, peak GPU allocation and synchronized latency with device, precision, batch size and warmup. Keep zero-shot, fine-tuned and any validation-selected reranked reports distinct. Record provenance and per-seed values before aggregate comparisons.

[AcrossPlace reproduction guide](https://github.com/CV4RA/AcrossPlace/blob/2a66c25f3a241c2af93eae9dbde658f3e19cde67/docs/REPRODUCE.md) · [Meta PE source](https://github.com/facebookresearch/perception_models/tree/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder) · [Official checkpoint](https://huggingface.co/facebook/PE-Core-L14-336/blob/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt)
