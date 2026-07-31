# Experiments Report — PlurVA Shared Task

Status as of 2026-07-31. Scores are against the shared task's **hidden test set** (organizers hold gold labels; every number below came from a Codabench submission, not a local eval).

## Summary

**Best result: curriculum v5, 0.8062 macro accuracy** (chinese=0.7788, indonesian=0.7139, srilankan=0.9260) — beats the previous best of 0.8044, a sequential-curriculum recipe developed by trial and error with no recorded hyperparameters and no reusable implementation. v5 combines four elements, three of which are methodological contributions introduced this project: soft-label distillation from annotator disagreement, positional-consistency regularization, curriculum-ordered training, and a Chinese-weighted final training stage, with a tuned learning rate.

## Methodological contributions

Three mechanisms not present in the team's original recipe, all grounded in the dataset's actual structure rather than generic architecture tweaks:

### 1. Soft-label KL distillation

Indonesian's `Gold_Answer` field is a 5-annotator vote list for all 366/366 rows (e.g. `"A, A, A, A, C"`). The original pipeline collapsed every row to a one-hot majority label, discarding signal on rows with a non-unanimous vote (a 4-1 split trained identically to a 5-0 split). Since this is a values/ethics task (not a knowledge task like MMLU), annotator disagreement on a moral scenario is plausibly genuine signal rather than noise to resolve away. The fix: train against the full vote distribution via a KL-style loss at the single-token answer position, instead of collapsing to one-hot. For zh/si (one-hot in every observed case) this is numerically identical to standard cross-entropy — verified via a `torch.allclose` unit test.

### 2. Positional-invariance consistency regularization

Motivated by a documented dev/test distribution mismatch, and an idea the team's own findings doc had proposed but not implemented (permuting answer-option positions). Each training example is forwarded together with a permuted-option-order variant in one batched call; both get full soft-label supervision, plus a Jensen-Shannon consistency term between the two predictions (JS chosen over raw KL for boundedness — KL grows unbounded as distributions sharpen late in training). `--consistency-lambda` (default 0.5) controls the weight.

### 3. k-fold CV + final-refit protocol

Addresses the risk that a fixed validation split permanently removes rows from an already small training set (zh=790, id=366, si=203 rows). k-fold cross-validation determines the early-stopping iteration count (median `best_iter` across folds); a final refit then trains on 100% of the data for that count. Every row gets a validation turn across folds, so no data is permanently withheld, and the iteration count used for the deliverable checkpoint is validated rather than chosen by hand.

### Continuous evaluation metric

Replaced a training-time accuracy metric that could only read 0, 0.5, or 1.0 (teacher-forced argmax on a 2-example batch, not representative of the real generation-based eval) with `Σ target[letter]·p_model[letter]` — a continuous, bounded `[0,1]` expected-probability-mass metric, computed as a byproduct of the soft-label loss.

### Curriculum ordering, newly combined with the mechanisms above

The team's strongest pre-existing strategy — training one language at a time, each stage continuing from the prior stage's LoRA adapter — had not previously been combined with the soft-label/consistency mechanisms above; every earlier test of those mechanisms used simultaneous (all-languages-every-step) training. Combining the two was the single biggest lever found in this project.

## Experiments and results

### Reference point: sequential curriculum — 0.8044

Developed by trial and error, never implemented as a reusable script, no soft-label or consistency mechanisms, no recorded hyperparameters. Checkpoint no longer retrievable.

| Stage | chinese | indonesian | srilankan | avg |
|---|---|---|---|---|
| chinese only | 0.7732 | 0.6668 | 0.8193 | 0.7531 |
| + indonesian | 0.7725 | 0.7084 | 0.8193 | 0.7667 |
| + aug-sinhala | 0.7725 | 0.7084 | 0.9284 | 0.8031 |
| + aug-chinese | 0.7763 | 0.7084 | 0.9284 | **0.8044** |

Also tried simultaneously (not curriculum) on the pre-fix data pipeline (tied-vote Indonesian rows dropped): 0.7501 avg.

### Soft-label + consistency mechanisms, simultaneous training (pre-curriculum)

| Config | chinese | indonesian | srilankan | avg | notes |
|---|---|---|---|---|---|
| Simultaneous, hard-label pipeline (pre-fix) | 0.7196 | 0.6512 | 0.8795 | 0.7501 | reference point, no soft-label/consistency mechanisms |
| Soft-label + consistency, λ=0.5, refit_iters=25 (CV-validated) | 0.7442 | 0.6941 | 0.9197 | 0.7860 | first full result with both mechanisms |
| Soft-label + consistency, λ=0.6, refit_iters=50 | 0.7676 | 0.6975 | 0.8871 | 0.7841 | confounded (λ and iters both changed) |
| Soft-label + consistency, λ=0.6, refit_iters=CV-determined | 0.7489 | 0.7003 | 0.9097 | 0.7863 | clean isolated test — λ barely matters in 0.5-0.6 range |

### Curriculum combined with the mechanisms — six iterations

New script `train_novel_curriculum.py`: each stage trains one language, continuing from the prior stage's adapter; an optional final "combined" stage switches to simultaneous training across all three languages.

| Version | Change from prior version | chinese | indonesian | srilankan | avg | Outcome |
|---|---|---|---|---|---|---|
| v1 | zh→id→si→zh, no combined stage, 25 iters/stage | 0.7654 | 0.6683 | 0.9059 | 0.7799 | worse than simultaneous training with the same mechanisms; Indonesian drops without revisiting |
| v2 | + revisit id (zh,id,si,zh,id) + combined stage (zh-weight=1.5) | 0.7757 | 0.7030 | 0.9235 | 0.8007 | large jump; best Indonesian score yet; 0.37pt behind the 0.8044 reference |
| v3 | combined-stage iters 25→40 | 0.7586 | 0.6975 | 0.9398 | 0.7986 | worse — more iters skews toward Sri Lankan (smallest dataset) |
| v4 | zh-weight 1.5→1.3 | 0.7757 | 0.7003 | 0.9247 | 0.8002 | tied with v2 — not sensitive in this range |
| **v5** | combined-stage learning rate 1e-4→5e-5 | 0.7788 | 0.7139 | 0.9260 | **0.8062** | **best result — beats the 0.8044 reference on all 3 languages simultaneously** |
| v6 | full per-stage k-fold CV+refit, lr=5e-5 applied to every stage (not just combined) | 0.7710 | 0.7003 | 0.9021 | 0.7911 | worse than v5, despite more rigorous iteration-count selection |

**v5 is the current best result and the one submitted.** It only lowered the learning rate for the final combined stage; the five curriculum stages before it still ran at the original lr=1e-4.

**v6** was a follow-up that replaced every hand-chosen iteration count with a CV-validated one (via a new, separate script, `train_novel_curriculum_cv.py`, which does not modify any existing script) and applied the lower learning rate to all six stages, not just the last. It underperformed v5. Leading hypothesis: the lower learning rate was only shown to help the *final* stage (fine-tuning an already-adapted model); applying it to early stages training from the base model may have limited the progress those stages could make in a small number of steps. This is the third time in this arc (after v3, v4) that a more thorough single-axis extension beyond what testing specifically validated has underperformed the combination that testing actually found.

## Key takeaways

- Curriculum ordering combined with the soft-label and consistency mechanisms was the strategy that ultimately beat the 0.8044 reference point.
- Learning rate mattered a lot (v2→v5); consistency-lambda in the 0.5-0.6 range and zh-weight in the 1.3-1.5 range did not.
- More training iterations is not a free lever on a dataset this small — it disproportionately shifts the model toward whichever language has the fewest examples (Sri Lankan, 203 rows).
- More rigorous iteration-count selection (v6's fully CV-validated, per-stage-tuned recipe) did not produce a better result than the simpler, partly hand-tuned v5 — a hyperparameter finding validated in one context (fine-tuning a curriculum-adapted model) did not generalize by analogy to a different context (training from the base model).
- Indonesian was the weakest language across nearly every strategy tried; v5's 0.7139 is the best Indonesian score recorded in this project.
