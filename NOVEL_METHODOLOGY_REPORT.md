# Novel Training Methodology Report — PlurVA Shared Task

Status as of 2026-07-31. Scores below are against the shared task's **hidden test set** (organizers hold gold labels; every number came from a Codabench submission, not a local eval — `data/test/*_test_without_gold.jsonl` has no gold field).

## 1. Background

The team led the Codabench leaderboard using Qwen3.5-4B with a hand-tuned sequential-curriculum LoRA recipe (0.8044 macro). That result works but isn't methodologically defensible as a research contribution — it's SFT with a curriculum order found by trial and error. Three concrete problems motivated a from-scratch redesign:

1. **Validation wastes irreplaceable training data.** Dev sets are tiny (zh=790, id=366, si=203 rows; 1,359 labeled examples total, vs. a 5,475-row test set). A fixed validation split permanently sacrifices data on an already data-starved leaderboard task.
2. **The training-time accuracy metric was meaningless** — teacher-forced argmax accuracy on a 2-example batch can only read 0, 0.5, or 1.0, and isn't even the real eval metric (real eval is generation + letter extraction).
3. **No genuine research novelty** — a hand-tuned curriculum order isn't a paper contribution.

## 2. Baseline model comparison (pre-fine-tuning)

Zero-shot, full dev set:

| Model | chinese | indonesian | srilankan | macro avg |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.2937 | 0.5710 | 0.3498 | 0.4048 |
| Qwen2.5-7B-Instruct | 0.4051 | 0.6175 | 0.5123 | 0.5116 |
| Gemma-3-4B-it | 0.4215 | 0.6421 | 0.6305 | 0.5647 |
| MERaLiON-LLaMA-3-8B-Instruct | 0.3171 | 0.6376 | 0.6424 | 0.5323 |
| Qwen3.5-4B | 0.5694 | 0.6668 | 0.8243 | 0.6869 |
| Gemma-4-E4B-it | 0.5016 | 0.6485 | 0.9235 | 0.6912 |

Qwen3.5-4B and Gemma-4-E4B-it were the strongest baselines; Qwen3.5-4B was selected for fine-tuning work. Gemma-4-E4B-it got *worse* on every language after LoRA fine-tuning (0.6912 → 0.6635) — unexplained, not yet root-caused.

## 3. Best-known strategy: sequential curriculum (0.8044)

Ad-hoc recipe, never implemented as a reusable local script (only ever run in Colab/Drive) until this session's curriculum-mode addition (§6):

| Stage | chinese | indonesian | srilankan | avg |
|---|---|---|---|---|
| chinese only | 0.7732 | 0.6668 | 0.8193 | 0.7531 |
| + indonesian | 0.7725 | 0.7084 | 0.8193 | 0.7667 |
| + aug-sinhala | 0.7725 | 0.7084 | 0.9284 | 0.8031 |
| + aug-chinese | 0.7763 | 0.7084 | 0.9284 | **0.8044** |

Each stage continues from the prior stage's checkpoint. "Augmented" stages used literal duplicated training rows with permuted option positions (proposed in this doc's original findings, §7 below), not a training-time loss term.

## 4. Novel methodology

Two mechanisms, grounded in this dataset's actual structure rather than generic architecture tweaks (`scripts/novel_data.py`, `scripts/train_novel.py`):

**Mechanism A — soft-label distillation.** Indonesian's `Gold_Answer` is always a 5-annotator vote list (e.g. `"A, A, A, A, C"`) for all 366/366 rows, not just the 72 with no strict majority. Training targets the full empirical vote distribution instead of a collapsed one-hot majority label. Degenerates to standard one-hot cross-entropy for zh/si (verified numerically identical via `torch.allclose`).

**Mechanism B — positional-invariance consistency regularization.** A Jensen-Shannon divergence penalty between the model's predicted answer distribution on an example and a position-permuted variant of the same content, weighted by `--consistency-lambda`. Targets a real, documented dev/test distribution mismatch (the team trained to strong dev scores on Chinese/Indonesian but scored notably lower on the hidden test set on those same languages) by discouraging the model from keying off which letter slot the answer landed in.

**CV + refit protocol** (`scripts/run_cv.py`, `scripts/train_novel.py`): k-fold cross-validation (k=5) trains k independent models fresh-from-base, each on (k-1)/k of the data, to find a validated iteration count (median `best_iter` across folds) without permanently sacrificing a validation split. A final refit then trains once more, fresh from base, on 100% of the data for that validated iteration count — that checkpoint, not any CV-round checkpoint, is the submission.

**Continuous metric**: `Σ target[letter] · p_model[letter]`, replacing the old 0/0.5/1 argmax accuracy — a free byproduct of Mechanism A's loss computation.

## 5. Novel-methodology results

| Config | chinese | indonesian | srilankan | avg | notes |
|---|---|---|---|---|---|
| Simultaneous macro training (pre-fix), `train_macro_lora_pt.py` | 0.7196 | 0.6512 | 0.8795 | 0.7501 | old hard-label pipeline, tied Indonesian rows dropped |
| **Soft-label + consistency (λ=0.5), refit_iters=25** | 0.7442 | 0.6941 | 0.9197 | **0.7860** | `refit_iters=25` is CV-validated (median `best_iter` across a real 5-fold sweep), not a guess — confirmed twice after initial mis-attribution |
| Soft-label + consistency (λ=0.6), refit_iters=50 | 0.7676 | 0.6975 | 0.8871 | 0.7841 | **confounded**: both λ and iters changed vs. the row above, so the per-language shift (zh +2.3, id +0.3, si -3.3) can't be attributed to λ alone |
| Soft-label + consistency (λ=0.6), refit_iters=CV-determined | 0.7489 | 0.7003 | 0.9097 | 0.7863 | clean, isolated comparison — same iters-selection method as the λ=0.5 row. Result: **λ barely matters in 0.5-0.6 range** (0.7860 vs 0.7863, within noise) |

**CV sweep detail (λ=0.5, k=5):** `val_metric_macro_mean=0.6419 ± 0.0531`; per-fold `best_iter` = [20, 25, 60, 15, 25] — a 4x spread, i.e. genuinely noisy at this dataset size, median=25 used for the refit.

## 6. Curriculum + novel mechanisms (in progress)

Every result in §5 tested the new mechanisms only under *simultaneous* training (all 3 languages every step, fresh from base). Curriculum ordering (§3) is independently the strongest lever found (+~1.8-2pts over every simultaneous variant) but was never combined with the new mechanisms — this is the most-supported untested hypothesis, and the cheapest real experiment left given limited remaining compute (no checkpoint exists to ensemble with, ruling that option out; a learning-rate sweep would cost 15+ full CV runs for a hypothesis with no direct evidence yet).

New script: `scripts/train_novel_curriculum.py`. Recipe: **zh → id → aug-si → aug-zh**, each stage continuing from the prior stage's adapter (`PeftModel.from_pretrained(..., is_trainable=True)`), training only on that stage's own language (not cumulative replay — matches the original recipe's low-forgetting behavior). Soft-label distillation + consistency regularization (λ=0.5) applied uniformly at every stage — replacing the original recipe's ad-hoc row-duplication "augmentation" with the actual novel mechanism. 25 fixed iterations/stage (reusing the CV-validated count from §5), no per-stage CV, one deliberate unswept run.

Checkpoint-continuation logic verified correct via a targeted unit check (perturbed a LoRA weight, saved, reloaded fresh, confirmed exact match rather than silent reinitialization) before committing to a real run. Result: **pending** (run in progress / not yet submitted).

## 7. Key findings

- **Indonesian is the macro-average's weak link across every strategy tried** — never above ~0.71, in fine-tuned or baseline form. The soft-label mechanism targets this directly since it's the only language with real annotator-disagreement signal to exploit.
- **Sequential curriculum's edge over simultaneous training is concentrated in Chinese** (+3.2pts vs. simultaneous macro, post-data-fix), not Indonesian (+1.4pts) or Sri Lankan (+0.9pts) — worth testing whether this is really about curriculum ordering, or just about Chinese getting double exposure (trained first and last in the original recipe).
- **consistency-lambda doesn't meaningfully move the macro average in the 0.5-0.6 range** — a clean, isolated comparison (§5, last two rows) came back within noise of each other.
- **CV fold-to-fold variance in `best_iter` is large (15-60, 4x spread)** on this small a dataset — the "right" iteration count is a genuinely noisy median, not a tight consensus. A learning-rate sweep is a plausible way to test whether this variance is reducible, but was deprioritized given limited compute and no direct evidence yet that LR (not just dataset size) is the cause.
- **A `prepare_training_data.py` fix** (tied-vote Indonesian rows kept instead of dropped) improved the *old* simultaneous-macro pipeline by +3.6pts (0.7501 → not yet re-measured with this exact combination, since the 0.786 result came from the novel methodology, not this pipeline) — this fix was never involved in any novel-methodology result, since `train_novel.py` builds targets directly via `resolve_gold_candidates`/`build_target_dist`, bypassing `prepare_training_data.py` entirely.
- **The original 0.8044 sequential-curriculum checkpoint no longer exists anywhere retrievable** — it was only ever run in Colab/Drive, never saved to this repo. This blocks ensembling as an option until/unless it's reproduced.

## 8. Infrastructure notes

- `.git` repo history was cleaned (8.9GB → 81MB) — orphaned temp pack files from an old interrupted operation, unrelated to tracked content.
- CV-round completion tracking had a real correctness bug (fixed): `round_is_done`/`aggregate()` previously inferred completion from `iter >= best_iter`, which is true after literally the first evaluation by construction — a crashed/interrupted round would be silently treated as complete. Fixed via an explicit `"finished"` boolean flag, verified against a real simulated crash-and-resume on this repo's actual CV harness.
- `--load-in-4bit` (QLoRA-style NF4 quantization) is available in `train_novel.py` for tight-VRAM cloud GPUs (e.g. T4, 16GB) — not needed on 24GB+ cards.

## 9. Open questions / next steps

1. Curriculum + novel-methodology result (§6) — pending, highest priority.
2. Chinese-oversampling test — does per-language sampling weight (not just curriculum order) explain sequential curriculum's Chinese-specific edge?
3. Learning-rate sweep — deprioritized given limited compute; worth revisiting if compute frees up, to test whether it's the source of the 4x CV fold-variance in `best_iter`.
4. Reproduce the original sequential-curriculum checkpoint (no new mechanisms) to enable ensembling with a novel-methodology checkpoint.
5. λ sweep beyond 0.5/0.6 (e.g. 0.25, 1.0) — low priority given 0.5 vs 0.6 showed no meaningful difference.
