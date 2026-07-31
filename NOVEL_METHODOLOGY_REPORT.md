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

## 6. Curriculum + novel mechanisms

Every result in §5 tested the new mechanisms only under *simultaneous* training (all 3 languages every step, fresh from base). Curriculum ordering (§3) is independently the strongest lever found (+~1.8-2pts over every simultaneous variant) but was never combined with the new mechanisms — this is the most-supported untested hypothesis, and the cheapest real experiment left given limited remaining compute (no checkpoint exists to ensemble with, ruling that option out; a learning-rate sweep would cost 15+ full CV runs for a hypothesis with no direct evidence yet).

New script: `scripts/train_novel_curriculum.py`. Recipe: **zh → id → aug-si → aug-zh**, each stage continuing from the prior stage's adapter (`PeftModel.from_pretrained(..., is_trainable=True)`), training only on that stage's own language (not cumulative replay — matches the original recipe's low-forgetting behavior). Soft-label distillation + consistency regularization (λ=0.5) applied uniformly at every stage — replacing the original recipe's ad-hoc row-duplication "augmentation" with the actual novel mechanism. 25 fixed iterations/stage (reusing the CV-validated count from §5), no per-stage CV, one deliberate unswept run.

Checkpoint-continuation logic verified correct via a targeted unit check (perturbed a LoRA weight, saved, reloaded fresh, confirmed exact match rather than silent reinitialization) before committing to a real run.

**Result: 0.7799 avg (chinese=0.7654, indonesian=0.6683, srilankan=0.9059) — underperforms both references.** The hypothesis did not pan out: this scores below the simultaneous novel-methodology results (0.7860-0.7863) *and* well below the original curriculum (0.8044). Confirmed via cloud training logs that all 4 stages genuinely completed all 25/25 iterations (not a truncated/broken run) before drawing this conclusion.

Notably, Indonesian scored its worst across every novel-methodology attempt (0.6683). Working hypothesis: in this new-language-only, no-cumulative-replay stage schedule (zh→id→si→zh), Indonesian is trained once in stage 1 and never revisited, while Chinese is trained first *and* last (stage 3). If the new soft-label+consistency loss causes even mild catastrophic forgetting between stages — unlike the original recipe's plain hard-label CE, described in the memory record as having "almost no forgetting" — Indonesian is the language most exposed to it, having two full subsequent stages with none of its own data before the run ends. Not yet confirmed; would need a 5th stage revisiting Indonesian, or per-stage validation tracking Indonesian's metric across stages 2-3, to test directly.

A real infrastructure bug was also found and fixed from this run's logs: recurring (non-fatal) CUDA OOM warnings throughout every stage — the allocator retried and recovered each time, but GPU free memory was repeatedly dropping under 1GB on a 31GB card. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` had only ever been added to the Colab notebook during earlier T4-OOM debugging, never ported to `train_novel.py`/`train_novel_curriculum.py` — fixed post-hoc, doesn't invalidate this result (training did complete correctly) but should prevent a harder failure on a future run.

### v2: Indonesian revisit + Chinese-weighted combined stage

Two follow-ups to v1's failure, both testable in one run:

1. **`--stages zh,id,si,zh,id`** — Indonesian revisited too, testing the forgetting hypothesis directly.
2. **`--final-combined-iters 25 --zh-weight 1.5`** — one more stage after the curriculum, switching back to *simultaneous* training across all 3 languages (same mechanism as §5) with zh's loss upweighted 1.5x before averaging — the curriculum-mode analog of "oversampling" Chinese (per-step batch weighting doesn't translate to sequential single-language stages the way it would in simultaneous training), testing whether sequential curriculum's Chinese-specific edge over simultaneous training is really about ordering, or just more effective Chinese exposure.

**Result: 0.8007 avg (chinese=0.7757, indonesian=0.7030, srilankan=0.9235) — 0.37 points behind the all-time best, by far the closest the novel methodology has gotten.** Both hypotheses were confirmed:
- Indonesian jumped from 0.6683 (v1) to 0.7030 — the best Indonesian score of any novel-methodology run, supporting the forgetting explanation.
- Chinese (0.7757) and Sri Lankan (0.9235) both recovered to within ~0.01-0.05 of the original curriculum's numbers (0.7763, 0.9284) — supporting the Chinese-exposure explanation over pure "ordering."

Notably, all three languages are now *uniformly* just slightly below the original curriculum's per-language scores, rather than one language dramatically behind (as in v1) — evidence the recipe direction is now right, and the remaining ~0.4pt gap is closer to "needs a bit more training/weight" than a structural problem. Untried next tweaks: more `--final-combined-iters` (currently an arbitrary 25) or a higher `--zh-weight` (currently 1.5).

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

1. Curriculum v2 (0.8007) is 0.37 points from the all-time best and the closest result yet — worth one more round of small tuning before moving on: more `--final-combined-iters` or a higher `--zh-weight`, given all three languages are now uniformly slightly behind rather than one being a structural outlier.
2. ~~Chinese-oversampling test~~ — addressed in v2's combined stage; confirmed Chinese exposure (not just ordering) explains part of the gap.
3. Learning-rate sweep — deprioritized given limited compute; worth revisiting if compute frees up, to test whether it's the source of the 4x CV fold-variance in `best_iter`.
4. Reproduce the original sequential-curriculum checkpoint (no new mechanisms) to enable ensembling with a novel-methodology checkpoint.
5. λ sweep beyond 0.5/0.6 (e.g. 0.25, 1.0) — low priority given 0.5 vs 0.6 showed no meaningful difference.
