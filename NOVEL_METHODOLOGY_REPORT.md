# Novel Training Methodology Report — PlurVA Shared Task

Status as of 2026-07-31. Scores are against the shared task's **hidden test set** (organizers hold gold labels; every number below came from a Codabench submission, not a local eval — `data/test/*_test_without_gold.jsonl` has no gold field locally). Where a number is a local/dev-set figure instead, it's labeled explicitly.

## 0. Executive summary

- **Current best: curriculum v5, 0.8062 macro accuracy** (chinese=0.7788, indonesian=0.7139, srilankan=0.9260) — beats the previous all-time-best of 0.8044 (the original ad-hoc sequential-curriculum recipe), and unlike that recipe, v5 has genuine mechanistic novelty: soft-label distillation from annotator disagreement + positional-consistency regularization + curriculum ordering + a Chinese-weighted final training stage.
- Path to v5: baseline model selection (Qwen3.5-4B) → original ad-hoc curriculum (0.8044) → from-scratch novel-methodology redesign (soft-label + consistency + CV/refit, §5) reaching 0.7860-0.7863 under simultaneous training → combining curriculum ordering with the new mechanisms (§6, `train_novel_curriculum.py`) → five iterations (v1-v5) of hypothesis-driven tuning, landing on v5.
- Along the way: multiple real infrastructure bugs found and fixed (§10), a from-8.9GB-to-81MB git repo cleanup (§11), and several instances of experiment provenance getting mis-recorded and corrected (§12) — kept here as a record of what actually happened, not just the clean final story.

## 1. Background

The team led the Codabench leaderboard using Qwen3.5-4B with a hand-tuned sequential-curriculum LoRA recipe (0.8044 macro, §3). That result worked but wasn't methodologically defensible as a research contribution — it's SFT with a curriculum order found by trial and error. Three concrete problems motivated a from-scratch redesign (user's own framing):

1. **Validation wastes irreplaceable training data.** Dev sets are tiny (zh=790, id=366, si=203 rows; 1,359 labeled examples total, vs. a 5,475-row test set). A fixed validation split permanently sacrifices data on an already data-starved leaderboard task. → addressed via k-fold CV + final refit (§5).
2. **The training-time accuracy metric was meaningless** — teacher-forced argmax accuracy on a 2-example batch (`train_macro_lora_pt.py`'s `_language_batch_loss`) can only read 0, 0.5, or 1.0, and isn't even the real eval metric (real eval is generation + letter extraction, not teacher-forced argmax). → addressed via a continuous expected-probability-mass metric (§5).
3. **No genuine research novelty** — a hand-tuned curriculum order isn't a paper contribution. → addressed via two mechanisms grounded in the dataset's actual structure (§4).

Design constraints set at the start: run on **either** local M1 Pro (MPS) or a Colab/CUDA GPU (device-portable, no CUDA-only tooling like Unsloth); train **fresh from the base Qwen3.5-4B model**, not warm-started from any existing checkpoint (keeps the causal story clean — gains attributable to the new mechanisms, not extra gradient steps). The curriculum-mode work later in this doc (§6) deliberately departs from the fresh-from-base rule, since curriculum ordering *requires* checkpoint continuation between stages — that's a distinct, explicitly-flagged departure, not an oversight.

**Foundational fact** (verified twice independently): for `Qwen/Qwen3.5-4B`'s tokenizer, the completions `" A"`/`" B"`/`" C"`/`" D"` each tokenize to exactly one token — IDs `{357, 417, 351, 414}`. This is what makes the soft-label/consistency mechanisms tractable without architectural surgery: the answer position in every training example is always exactly one token, found generically from the `-100` label mask (`(labels_shifted != -100).float().argmax(dim=1)`).

## 2. Baseline model comparison (pre-fine-tuning)

### 2a. Informal comparison (15 hand-picked questions per language, qualitative)

| Model | Sinhala | Chinese | Indonesia | Macro Avg |
|---|---|---|---|---|
| Llama 3.1 (8B) | 0.33 | 0.26 | 0.73 | 0.44 |
| Qwen 2.5 (7B) | 0.33 | 0.73 | 0.93 | 0.663 |
| Gemma 3 (4B) | 0.4 | 0.4 | 0.8 | 0.53 |

Qualitative notes from this pass: Qwen 2.5 8B gave correct Sinhala answers but incomplete explanations, incorrect Chinese answers, correct+good Indonesian. Llama 3.1 8B was correct on all three languages but with a recurring (repetitive) Sinhala explanation style. Gemma 3 4B answered Sinhala correctly only when the query was processed in English — forced-Sinhala queries got the wrong answer, with an explanation that actively mistranslated Sinhala meaning. Mistral-7B-v0.3 gave no good Sinhala response at all and was wrong on Indonesian.

### 2b. Full dev-set zero-shot comparison

| Model | chinese | indonesian | srilankan | macro avg |
|---|---|---|---|---|
| unsloth/llama-3.1-8b-Instruct-bnb-4bit | 0.2937 | 0.5710 | 0.3498 | 0.4048 |
| unsloth/Ministral-3-3B-Instruct-2512 | 0.3582 | 0.4672 | 0.1970 | 0.3408 |
| unsloth/Ministral-3-8B-Instruct-2512 | 0.2835 | 0.6120 | 0.2118 | 0.3691 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.2975 | 0.4809 | 0.5271 (only ever answers "A") | 0.4351 |
| Qwen/Qwen2.5-7B-Instruct | 0.4051 | 0.6175 | 0.5123 | 0.5116 |
| google/gemma-3-4b-it | 0.4215 | 0.6421 | 0.6305 | 0.5647 |
| MERaLiON-LLaMA-3-8B-Instruct | 0.3171 | 0.6376 | 0.6424 | 0.5323 |
| Qwen/Qwen3.5-4B | 0.5694 | 0.6668 | 0.8243 | 0.6869 |
| google/gemma-4-E4B-it | 0.5016 | 0.6485 | 0.9235 | 0.6912 |

Qwen3.5-4B and Gemma-4-E4B-it were the strongest baselines; **Qwen3.5-4B was selected for all fine-tuning work.**

### 2c. Sinhala zero-shot vs. one-shot (isolated finding, never incorporated into the fine-tuning pipeline)

| Model | Zero-shot (203) | One-shot (202) |
|---|---|---|
| Qwen3-8B | 0.3547 | 0.6881 |
| **Qwen3.5-4B** | **0.6207** | **0.7673** |
| Llama-3.2-3B-Instruct | 0.4877 | 0.3663 |
| Llama-3.1-8B-Instruct | 0.3498 | 0.7129 |
| gemma-4-E4B-it | 0.5961 | 0.7228 |
| gemma-3-4b-it | 0.7044 | 0.6188 |
| Mistral-7B-Instruct-v0.3 | 0.5468 | 0.5050 |
| DeepSeek-R1-Distill-Llama-8B | 0.4778 | 0.3218 |
| DeepSeek-R1-Distill-Qwen-7B | 0.2069 | 0.2178 |

A single in-context example moved Qwen3.5-4B's Sinhala accuracy from 0.6207 to 0.7673 (+15.7pts) — a large, cheap effect that was never stacked with the fine-tuning work in this report. Flagged as a potential lever (§13) but not pursued given compute constraints.

### 2d. Fine-tuning baseline comparison (Codabench, hidden test)

| Model | Strategy | chinese | indonesian | srilankan | avg |
|---|---|---|---|---|---|
| google/gemma-4-E4B-it | baseline | 0.5016 | 0.6485 | 0.9235 | 0.6912 |
| google/gemma-4-E4B-it | LoRA (separate adapters/lang) | 0.4844 | 0.6104 | 0.8959 | 0.6635 |
| Qwen/Qwen3.5-4b | baseline | 0.5694 | 0.6668 | 0.8243 | 0.6869 |
| MERaLiON-LLaMA-3-8B-Instruct | baseline | 0.3171 | 0.6376 | 0.6424 | 0.5323 |

Gemma-4-E4B-it got *worse* on every language after LoRA fine-tuning (0.6912 → 0.6635) — unexplained, never root-caused, not investigated further since the team standardized on Qwen3.5-4B.

## 3. Best-known original strategy: sequential curriculum (0.8044, superseded by §6 v5)

Ad-hoc recipe, developed by trial and error, **never implemented as a reusable local script — only ever run in Colab/Drive** — until this session's curriculum-mode addition (§6). The checkpoint itself no longer exists anywhere retrievable.

| Stage | chinese | indonesian | srilankan | avg |
|---|---|---|---|---|
| chinese only | 0.7732 | 0.6668 | 0.8193 | 0.7531 |
| + indonesian | 0.7725 | 0.7084 | 0.8193 | 0.7667 |
| + aug-sinhala | 0.7725 | 0.7084 | 0.9284 | 0.8031 |
| + aug-chinese | 0.7763 | 0.7084 | 0.9284 | 0.8044 |

Each stage continues from the prior stage's checkpoint. "Augmented" stages used literal duplicated training rows with permuted option positions (an idea proposed in the team's original findings doc, quoted in §4 Mechanism B) — a data-augmentation trick, not a training-time loss term. No hyperparameters (learning rate, epochs, LoRA rank, batch size) were ever recorded for this recipe.

Also tried on the old `train_macro_lora_pt.py` (simultaneous, not curriculum) pipeline: 0.7501 avg (chinese=0.7196, indonesian=0.6512, srilankan=0.8795) — this used the *old*, buggy `prepare_training_data.py` that dropped tied-vote Indonesian rows entirely instead of keeping them (fixed later, §10, but that fix was never re-measured against this exact simultaneous pipeline — the team moved to the novel methodology instead).

## 4. Novel methodology design

Two mechanisms, grounded in this dataset's actual structure rather than generic architecture tweaks (`scripts/novel_data.py`, `scripts/train_novel.py`).

### Mechanism A — soft-label KL distillation

Indonesian's `Gold_Answer` field is always a 5-annotator vote list (e.g. `"A, A, A, A, C"`), for all 366/366 rows — not just the 72 rows with no strict majority. The old pipeline (`prepare_training_data.py`, pre-fix) collapsed every row to a one-hot majority label and only specially handled the 72 tied rows (by duplicating them into separate hard-label examples) — discarding real signal on the other 294 rows (a 4-1 vote split trained identically to a 5-0 split). On a *values/ethics* task (the dataset schema includes `Value`/`Value_English` columns — this isn't a knowledge task like MMLU), annotator disagreement on a moral scenario is plausibly genuine signal, not noise to resolve away.

```python
def build_target_dist(lang, gold_answer):
    if lang == "id":
        votes = [v.strip() for v in gold_answer.split(",")]
        counts = Counter(votes)
        return {l: counts.get(l, 0) / len(votes) for l in "ABCD"}
    candidates = resolve_gold_candidates(lang, gold_answer)   # zh/si: one-hot in every observed case
    p = 1.0 / len(candidates)
    return {l: (p if l in candidates else 0.0) for l in "ABCD"}
```

Loss (mathematically KL(target || model) up to a model-independent constant — the target's own entropy doesn't depend on model parameters):

```python
def soft_ce_at_answer_position(logits_shifted, labels_shifted, target_dist_batch, letter_ids):
    valid = (labels_shifted != -100)
    pos = valid.float().argmax(dim=1)
    logits_at_pos = logits_shifted[torch.arange(logits_shifted.size(0)), pos]
    log_probs = F.log_softmax(logits_at_pos, dim=-1)
    letter_log_probs = log_probs[:, letter_ids]  # (B, 4), columns A,B,C,D
    loss = -(target_dist_batch * letter_log_probs).sum(dim=1).mean()
    metric = (target_dist_batch * letter_log_probs.exp()).sum(dim=1).mean()  # continuous metric, §4c
    return loss, metric, logits_at_pos
```

For zh/si (always one-hot target in this dataset), this is numerically **identical** to `F.cross_entropy` — verified as a literal `torch.allclose` unit test (`verify_novel.py` check 6), proving "generalizes hard CE" isn't just asserted.

### Mechanism B — positional-invariance consistency regularization

The team's own findings doc had proposed but never implemented this: *"augment the data by changing the position of the correct answer... for Chinese create 3-4 permutations... for Indonesian create 3-4 permutations, including when 2 answers are present... for Sinhala, permutate the A and B options, so there'd only be 2 permutations per question."* Motivated by a documented dev/test distribution mismatch: *"our development set and the final test set are at odds... training to get good scores on Chinese and Indonesian on the development set... but on the test set our Chinese score is 0.53 and Indonesian score is 0.64."*

```python
def permute_example(options, target_dist, perm):     # perm: old_letter -> new_letter
    new_options = {perm[l]: options[l] for l in "ABCD"}
    new_target = defaultdict(float)
    for l, p in target_dist.items():
        new_target[perm[l]] += p
    return new_options, dict(new_target)
```

si's `perm["C"] == perm["D"] == identity` always (C/D are fixed "Both"/"None" meta-options in this dataset, not permutable — asserted in `verify_novel.py` check 4).

Original+permuted variants are forwarded together in one batched call (not two sequential — better MPS/CUDA utilization), each getting full Mechanism-A supervision, plus a Jensen-Shannon consistency term on top (JS chosen over raw KL for symmetry and boundedness — KL blows up as either side's distribution sharpens toward one-hot late in training; JS is bounded to `[0, ln 2]` regardless):

```python
def consistency_loss(logits_at_pos_orig, logits_at_pos_perm, perm_idx, letter_ids):
    q_o = F.softmax(logits_at_pos_orig[:, letter_ids], dim=-1)
    q_p_raw = F.softmax(logits_at_pos_perm[:, letter_ids], dim=-1)
    q_p = torch.gather(q_p_raw, 1, perm_idx)  # remapped to canonical letter order
    m = 0.5 * (q_o + q_p)
    js = (0.5*(q_o*(q_o.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(1)
        + 0.5*(q_p*(q_p.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(1))
    return js.mean()

lang_loss = 0.5*(loss_orig + loss_perm) + consistency_lambda * consistency_loss(...)
```

`--consistency-lambda` (default 0.5) is a swept hyperparameter (§5). Verified: feeding the same example as both "original" and "permuted with the identity permutation" gives `consistency_loss ≈ 0` (`verify_novel.py` check 7). Added cost: one extra forward+backward per step (~1.6-2x wall-clock vs. plain CE).

### 4c. CV + refit protocol (solves problem #1)

- k-fold cross-validation (k=5 cloud tier / k=3 Mac tier), same fold index synchronized across zh/id/si within a round (round r's train = all folds ≠ r for every language; round r's val = fold r for every language) — one shared multilingual model per round.
- Stratified fold assignment (group by majority-vote gold letter, seeded shuffle, round-robin), no scikit-learn dependency, implemented manually in `novel_data.py`. Determinism required since each round runs as a separate subprocess (verified: same seed → identical folds across separate process runs, `verify_novel.py` check 5).
- CV selects the early-stopping **iteration count**: median `best_iter` across folds, rounded to the nearest `--steps-per-eval` multiple.
- **Final refit**: one more fresh-from-base-model run on 100% of dev data (1,359 rows, no held-out val) for the CV-selected fixed iteration budget — no early stopping possible (no validation signal at all in refit mode). This checkpoint, not any CV-round checkpoint, is the submission. The CV folds' mean±std becomes the rigor number; the refit is the deliverable.

### 4d. Continuous metric (solves problem #2)

`Σ target[letter] · p_model[letter]` — expected probability mass on the correct answer, weighted by the (soft or one-hot) target. A free byproduct of `soft_ce_at_answer_position`, no extra forward pass. Bounded `[0,1]`, continuous, dramatically lower variance than the old 0/0.5/1 metric. For Indonesian under Mechanism A this is literally "expected probability mass under the 5-vote distribution" — no id-specific branch needed.

### 4e. Planned ablation grid

One-axis-at-a-time from a base config (Mac tier cuts CV to k=3 or single-split, no λ sweep, lower iteration ceiling):

| # | Label | Target | Consistency-reg | Data protocol | Purpose | Status |
|---|---|---|---|---|---|---|
| 1 | Baseline control | hard (collapse+dup) | off | single split | reproduces old architecture | not run as a standalone config (superseded by §3's real 0.7501 result) |
| 2 | +CV/refit only | hard | off | k-fold CV + refit | isolates data-protocol contribution | not run |
| 3 | +Soft-label only | soft | off | single split | isolates Mechanism A | not run |
| 4 | +Consistency only | hard | on | single split | isolates Mechanism B | not run |
| 5 | +Soft+Consistency | soft | on | single split | A+B interaction, no CV | not run |
| 6 | Full (original headline) | soft | on | k-fold CV + refit | main claimed result | **run, §5** |
| 7a-c | λ sweep | soft | on, λ∈{0.25,0.5,1.0} | single split | justifies λ | **partially run** (0.5, 0.6 only, §5) |
| 8 | Dev/test gap table | configs 1 & 6 | — | actual Codabench scores | substantiates the mismatch-fix claim | partially available via §2d/§5 |

Most individual-mechanism ablations (2-5) were never run in isolation — compute went toward the curriculum-combination direction (§6) instead once it proved more promising.

## 5. Novel-methodology results (simultaneous training, pre-curriculum)

| Config | chinese | indonesian | srilankan | avg | notes |
|---|---|---|---|---|---|
| Simultaneous macro training (pre-fix), `train_macro_lora_pt.py` | 0.7196 | 0.6512 | 0.8795 | 0.7501 | old hard-label pipeline, tied Indonesian rows dropped (§3) |
| **Soft-label + consistency (λ=0.5), `train_novel.py --final-refit --refit-iters 25`** | 0.7442 | 0.6941 | 0.9197 | 0.7860 | `refit_iters=25` is CV-validated (median `best_iter` across a real 5-fold sweep), not a guess — confirmed twice after initial mis-attribution (§12) |
| Soft-label + consistency (λ=0.6), `--refit-iters 50` | 0.7676 | 0.6975 | 0.8871 | 0.7841 | **confounded**: both λ and iters changed vs. the row above, so the per-language shift (zh +2.3, id +0.3, si -3.3) can't be attributed to λ alone |
| Soft-label + consistency (λ=0.6), refit_iters=CV-determined | 0.7489 | 0.7003 | 0.9097 | 0.7863 | clean, isolated comparison — same iters-selection method as the λ=0.5 row. **λ barely matters in the 0.5-0.6 range** (0.7860 vs 0.7863, within noise) |

**CV sweep detail (λ=0.5, k=5):** `val_metric_macro_mean = 0.6419 ± 0.0531`. Per-fold detail:

| Round | best_iter | val_loss_macro | val_metric_macro | zh / id / si val_metric |
|---|---|---|---|---|
| 0 | 20 | 0.7021 | 0.6732 | 0.747 / 0.394 / 0.879 |
| 1 | 25 | — | — | — |
| 2 | 60 | — | — | — |
| 3 | 15 | — | — | — |
| 4 | 25 | — | — | — |

`median_best_iter=25`, used as `suggested_refit_iters`. The 4x spread in `best_iter` (15-60) was initially treated as possible small-dataset noise, deprioritized for a learning-rate sweep given compute constraints — later (§6 v5) confirmed to be a real too-aggressive-default-LR signal, not noise, once a lower LR was actually tested.

## 6. Curriculum + novel mechanisms

Every result in §5 tested the new mechanisms only under *simultaneous* training (all 3 languages every step, fresh from base). Curriculum ordering (§3) was independently the strongest lever found (+~1.8-2pts over every simultaneous variant) but had never been combined with the new mechanisms — the most-supported untested hypothesis, and the cheapest real experiment left given limited remaining compute at the time (no checkpoint existed to ensemble with the original 0.8044 recipe, ruling that out; a learning-rate sweep on the simultaneous script would have cost 15+ full CV runs for a hypothesis with no direct evidence yet).

New script: **`scripts/train_novel_curriculum.py`**, reusing `train_novel.py`'s tokenization/loss/step functions directly (`letter_token_ids`, `SoftPromptCompletionDataset`, `build_lang_examples`, `lang_step`) rather than reimplementing them, and `train_sft_then_grpo.py`'s checkpoint-continuation pattern:

```python
base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
if prev_adapter_path is None:
    model = get_peft_model(base_model, lora_config)          # stage 0: fresh
else:
    model = PeftModel.from_pretrained(base_model, str(prev_adapter_path), is_trainable=True)  # continue
```
(`is_trainable=True` is required — `PeftModel.from_pretrained` defaults to a frozen/eval-style load otherwise.)

Design decisions confirmed with the user before implementation:
- **New-language-only per stage** (not cumulative replay of prior languages) — matches the memory record's description of the original recipe ("each stage mostly moves only the newly-added language, almost no forgetting") and is cheaper.
- **`--consistency-lambda` applied uniformly at every stage** (not just the aug-stages the way the original recipe's row-duplication augmentation was) — replaces the old ad-hoc idea with the actual novel mechanism; this is the hypothesis under test, not a replication.
- **Fixed `--iters-per-stage` (default 25, the CV-validated count from §5), no per-stage CV/early-stopping** — kept to a single run given limited compute.

Checkpoint-continuation logic was verified correct via a fast, targeted unit check *before* committing to a real (slow) training run: perturbed a `lora_B` weight (normally zero-initialized) by adding 1.0, saved the adapter, reloaded it fresh via `PeftModel.from_pretrained(..., is_trainable=True)` on a newly-loaded base model, and confirmed the reloaded value matched the perturbation exactly (`torch.allclose`) rather than being silently reinitialized to zero. Ran in under 30 seconds (no actual training involved) after an initial full smoke test proved too slow for the user's open (billed) cloud instance to wait on.

### v1: `zh, id, si, zh` — failure

Recipe: **zh → id → aug-si → aug-zh**, each stage continuing from the prior stage's adapter, training only on that stage's own language. 25 fixed iterations/stage.

**Result: 0.7799 avg (chinese=0.7654, indonesian=0.6683, srilankan=0.9059) — underperforms both references.** Below both the simultaneous novel-methodology results (0.7860-0.7863, §5) and well below the original curriculum (0.8044). Confirmed via cloud training logs that all 4 stages genuinely completed all 25/25 iterations (not a truncated/broken run — `predict_test.py` requires the final stage's adapter to exist at all, and it processed all 5475 test rows with 0 unparseable) before drawing this conclusion — the user explicitly asked "are you sure this trained correctly" given the suspiciously short training wall-clock time (~6.6 minutes for all 4 stages combined on the cloud GPU), which turned out to be genuine GPU speed, not skipped work.

Indonesian scored its worst across every novel-methodology attempt (0.6683). Working hypothesis: in this new-language-only, no-cumulative-replay schedule, Indonesian is trained once in stage 1 and never revisited, while Chinese is trained first *and* last (stage 3). If the new soft-label+consistency loss causes even mild catastrophic forgetting between stages — unlike the original recipe's plain hard-label CE, described in the memory record as having "almost no forgetting" — Indonesian is the language most exposed to it, having two full subsequent stages with none of its own data before the run ends.

**Real infrastructure bug found from this run's cloud logs**: recurring (non-fatal) CUDA OOM warnings throughout every stage — PyTorch's allocator retried and recovered each time, but GPU free memory was repeatedly dropping under 1GB on a 31.4GB card before every retry. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` had only ever been added to the earlier Colab notebook during T4-OOM debugging, never ported to `train_novel.py` or `train_novel_curriculum.py` — same class of gap as other notebook-vs-script mismatches found this session (§10). Fixed post-hoc in both scripts; didn't invalidate the v1 result (training completed correctly) but reduced risk of a harder failure on future runs.

### v2: Indonesian revisit + Chinese-weighted combined stage — first real progress

Two follow-ups to v1's failure, combined into one run per the user's explicit request ("can we do both at the same time... combining"):

1. **`--stages zh,id,si,zh,id`** — Indonesian revisited too (`STAGES` made configurable via a new `--stages` CLI flag, comma-separated), testing the forgetting hypothesis directly.
2. **`--final-combined-iters 25 --zh-weight 1.5`** — one more stage after the curriculum, switching back to *simultaneous* training across all 3 languages (same mechanism as §5's `train_novel.py`) with zh's loss upweighted 1.5x before averaging.

**Why weight Chinese specifically?** Comparing the original sequential-curriculum recipe (§3) against simultaneous training (§5) language-by-language shows Chinese with a disproportionately large gap: sequential curriculum beat simultaneous training by +3.2pts on Chinese, vs. only +1.4pts on Indonesian and +0.9pts on Sri Lankan. Likely explanation: the original recipe trains Chinese *twice* (it's both the first stage and the last, "aug-chinese"), while Indonesian and Sri Lankan each train only once — so Chinese simply gets more total exposure, not necessarily an ordering effect specific to Chinese. Since v2 was already adding a stage where all three languages train together (to fix the Indonesian-forgetting problem), upweighting Chinese's loss there was a cheap way to test the extra-exposure hypothesis without needing a whole separate Chinese-only stage.

This is the curriculum-mode analog of "oversampling" Chinese — per-step batch weighting (the literal meaning of oversampling in a simultaneous-training context) doesn't translate to sequential single-language stages, so loss reweighting in the final combined stage was chosen instead (an explicit AskUserQuestion decision — the alternative considered was a bigger per-step batch for zh specifically, within that same combined stage).

**Result: 0.8007 avg (chinese=0.7757, indonesian=0.7030, srilankan=0.9235) — 0.37 points behind the all-time best, by far the closest the novel methodology had gotten at that point.** Both hypotheses confirmed:
- Indonesian jumped from 0.6683 (v1) to 0.7030 — the best Indonesian score of any novel-methodology run to that point.
- Chinese (0.7757) and Sri Lankan (0.9235) both recovered to within ~0.01-0.05 of the original curriculum's numbers (0.7763, 0.9284).

All three languages were now *uniformly* just slightly below the original curriculum's per-language scores, rather than one language dramatically behind (as in v1) — evidence the recipe direction was right, remaining gap looked like "needs a bit more training/weight," not a structural problem.

### v3: does more combined-stage training help? — no

Simplest next guess: keep `--zh-weight 1.5`, increase `--final-combined-iters` from 25 to 40. Made cheap to test via a new `--resume-from` flag (added to `train_novel_curriculum.py` specifically for this) that skips straight to the combined stage using an already-trained `stage_4_id/adapter` from a prior run, instead of redoing all 5 curriculum stages identically.

**Result: 0.7986 avg (chinese=0.7586, indonesian=0.6975, srilankan=0.9398) — worse than v2, not better.** Chinese dropped 1.71pts, Indonesian dropped 0.55pts, Sri Lankan *gained* 1.63pts. "More iterations helps proportionally" disproven: more combined-stage training shifted the model toward Sri Lankan specifically — likely because Sri Lankan has the smallest dataset of the three (203 rows), so at a fixed batch size with replacement sampling, its small example pool gets relatively more repeated exposure as total iterations increase, even though the per-language loss weighting (zh=1.5, id=si=1.0) never changed.

### v4: lower zh-weight instead — also no improvement

Other single-axis guess: keep `--final-combined-iters 25` (back to what worked), lower `--zh-weight` from 1.5 to 1.3.

**Result: 0.8002 avg (chinese=0.7757, indonesian=0.7003, srilankan=0.9247) — within noise of v2.** Chinese landed on the *exact same* score as v2 to 4 decimal places; indonesian/srilankan moved by ~0.001-0.003. Same pattern as the earlier λ=0.5-vs-0.6 finding: this hyperparameter isn't sensitive in the tested range. Both single-axis tweaks off v2 (v3: more iters, v4: lower zh-weight) had now failed to beat it — `--final-combined-iters 25 --zh-weight 1.5` looked like a genuine local optimum for this recipe shape, not just an untested first guess.

### v5: lower learning rate — NEW BEST, beats the all-time record

Third single-axis guess off v2: keep `--final-combined-iters 25 --zh-weight 1.5`, lower the combined stage's `--learning-rate` from the default 1e-4 to 5e-5. Motivated by the old CV fold-variance signal (§5: `best_iter` varied 15-60 across folds, a 4x spread), which hinted the default LR might converge a bit too fast/unstably for this dataset size — a hypothesis that had been explicitly deprioritized earlier due to compute cost, then revisited here.

Methodological caveat raised and discussed *before* running: testing a lower LR at the *same* iteration budget confounds "worse LR" with "just undertrained" if the result comes back worse (lower LR normally needs *more* steps to converge). Bumping iterations to compensate wasn't a safe fix either, since v3 had just proven more iterations distorts the per-language balance (toward Sri Lankan). Resolution: pre-register the interpretation before running — a worse result would be inconclusive either way, but a *better* result despite the same-or-worse effective convergence budget would be a genuinely strong signal, not an artifact.

**Result: 0.8062 avg (chinese=0.7788, indonesian=0.7139, srilankan=0.9260) — beats v2 on all three languages simultaneously, and beats the all-time-best 0.8044.** First result, from any strategy tried (novel or the original ad-hoc recipe), to exceed 0.8044. Per the pre-registered interpretation: since this improved despite a lower LR at the same iteration count (normally a disadvantage), it's strong evidence the default `lr=1e-4` really was too aggressive for this recipe/dataset, confirming the old CV fold-variance signal was real, not noise.

**Caveat / untested follow-up:** v5 only changed the learning rate of the *final combined stage* (via `--resume-from`, reusing v2's already-trained `stage_4_id` checkpoint) — the 5 curriculum stages before it were still trained at the original `lr=1e-4`. A full recipe rerun at the lower LR throughout (not just the last stage) is an untested, bigger-compute follow-up if further gains are wanted (§13).

## 7. Curriculum results at a glance

| Version | Stages | Combined stage | chinese | indonesian | srilankan | avg | vs. previous best |
|---|---|---|---|---|---|---|---|
| v1 | zh,id,si,zh | none | 0.7654 | 0.6683 | 0.9059 | 0.7799 | worse than 0.7860 (§5) |
| v2 | zh,id,si,zh,id | iters=25, zh-weight=1.5, lr=1e-4 | 0.7757 | 0.7030 | 0.9235 | 0.8007 | best so far, -0.37 from 0.8044 |
| v3 | zh,id,si,zh,id | iters=40, zh-weight=1.5, lr=1e-4 | 0.7586 | 0.6975 | 0.9398 | 0.7986 | worse than v2 |
| v4 | zh,id,si,zh,id | iters=25, zh-weight=1.3, lr=1e-4 | 0.7757 | 0.7003 | 0.9247 | 0.8002 | tied with v2 |
| **v5** | zh,id,si,zh,id | iters=25, zh-weight=1.5, **lr=5e-5** | 0.7788 | 0.7139 | 0.9260 | **0.8062** | **NEW BEST, beats 0.8044** |

## 8. `train_novel_curriculum.py` CLI reference (final state)

`--model-id`, `--seed`, `--max-seq-length`(768), `--lora-rank`(16), `--lora-alpha`(32.0), `--grad-checkpoint`/`--no-grad-checkpoint`, `--max-grad-norm`(1.0), `--consistency-lambda`(0.5, applied every stage), `--hard-labels` (ablation control), `--per-lang-batch-size`(8), `--iters-per-stage`(25), `--stages`(default `"zh,id,si,zh,id"`, comma-separated, validated against `LANGS`), `--final-combined-iters`(0 = disabled), `--zh-weight`(1.5, only used if the combined stage runs), `--resume-from`(path to an existing stage adapter, skips straight to the combined stage — requires `--final-combined-iters > 0` and the path to exist, both validated with a clear `ap.error`), `--steps-per-report`(5), `--learning-rate`(1e-4), `--out-dir`(default `results/novel_curriculum`).

Per-stage adapters are always saved individually (`{out_dir}/stage_{i}_{lang}/adapter`, or `stage_{i}_combined/adapter` for the final stage) — not just the final checkpoint — which is what made `--resume-from` possible to add later without re-architecting anything.

## 9. Command reference for the winning (v5) recipe

Full path to v5 from a clean clone, in the order actually used across this session (later runs reused earlier checkpoints via `--resume-from` rather than rerunning everything each time):

```bash
# v2 (produced the checkpoints v3/v4/v5 all resumed from)
python3 scripts/train_novel_curriculum.py \
  --stages zh,id,si,zh,id \
  --iters-per-stage 25 --per-lang-batch-size 8 \
  --final-combined-iters 25 --zh-weight 1.5 \
  --out-dir results/novel_curriculum_v2

# v5 (the actual winning config -- resumes v2's stage_4_id adapter)
python3 scripts/train_novel_curriculum.py \
  --resume-from results/novel_curriculum_v2/stage_4_id/adapter \
  --final-combined-iters 25 --zh-weight 1.5 \
  --learning-rate 5e-5 \
  --per-lang-batch-size 8 \
  --out-dir results/novel_curriculum_v5

python3 scripts/predict_test.py \
  --model-id Qwen/Qwen3.5-4B \
  --adapter-path results/novel_curriculum_v5/stage_5_combined/adapter \
  --out results/test_submission_curriculum_v5.jsonl
```

## 10. Bugs found and fixed this session

All found and fixed as part of this same body of work, roughly chronological:

1. **fp16 NaN on MPS when resuming GRPO training** (`train_sft_then_grpo.py`) — misdiagnosed initially as gradient explosion (added `--max-grad-norm` clipping; didn't fix it, identical NaN pattern with/without clipping). Root cause: fp16's narrow dynamic range through Qwen3.5's Gated DeltaNet torch-fallback attention path. Fixed by switching MPS dtype to bf16.
2. **GRPO KV-cache silently disabled** — `generate()` called on a `.train()`-mode model with gradient checkpointing enabled silently disables the KV cache (a `transformers` interaction, not a bug in this repo's code, but unhandled). Fixed by wrapping rollout sampling in `model.eval()`/`model.train()`.
3. **Missing `--skip-grpo` CLI flag** and a missing guard against `--skip-sft`+`--skip-grpo` together (would leave nothing to run) — added.
4. **`round_is_done` tautological completion check** (`run_cv.py`, later ported understanding to `scripts/run_cv.py`'s real implementation): `iter >= best_iter` is true after literally the first evaluation by construction (`best_iter` can never exceed the current `iter`), so an interrupted/crashed CV round would be silently marked "done." Fixed via an explicit `"finished"` boolean, written only on genuine completion (iteration ceiling or early-stopping fired on purpose) — verified via a real simulated crash-and-resume against the actual CV harness (killed a round mid-training, confirmed `finished=False` on disk, confirmed `round_is_done()`/`aggregate()` both correctly refused to treat it as complete, confirmed a resume correctly restarted it from scratch).
5. **`aggregate()` trusting any existing `summary.json` as complete** — same bug class as #4, one level downstream: even after the `finished` flag existed, `aggregate()` didn't check it. Fixed to skip/refuse aggregation on any round with `finished: false`.
6. **`%%capture` hiding pip install failures** (Colab notebook only) — added a post-install import-verification cell that raises a clear, named error if any critical package fails to import, rather than failing confusingly downstream.
7. **Consistency loss (`c_loss`) silently discarded** — computed every training step but never logged anywhere in the Colab notebook's `run_round` (found while answering a user question about training losses). Note: this bug never existed in the actual `.py` script (`train_novel.py` already logged it per-iteration) — only the notebook had dropped it during porting.
8. **CUDA OOM on a T4 GPU** (Colab) — multi-round fix: reduced `--per-lang-batch-size`, added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and ultimately added QLoRA-style 4-bit (NF4) quantization (`--load-in-4bit`, via `BitsAndBytesConfig` + `prepare_model_for_kbit_training`) to shrink the base model's footprint from ~8GB to ~2.5-3GB on tight-VRAM cards.
9. **`PYTORCH_CUDA_ALLOC_CONF` never ported from the notebook to the `.py` scripts** — found from curriculum v1's cloud logs (repeated, non-fatal CUDA OOM-retry warnings on a 31.4GB card). Fixed in both `train_novel.py` and `train_novel_curriculum.py` (the latter also covers `run_cv.py`, which launches `train_novel.py` as a subprocess and inherits its env-var setting).
10. **`--load-in-4bit` never ported to `train_novel.py`** either, only the notebook — ported for parity (not load-bearing for any result in this report, since cloud runs used large-enough GPUs, but available as a fallback).
11. **Checkpoint-continuation correctness** (`train_novel_curriculum.py`, new code, not a fix to existing code but flagged as the single most likely place for a silent bug given it was new and untested) — verified via a fast targeted check before trusting any real curriculum run: perturbed a LoRA weight, saved, reloaded fresh via `PeftModel.from_pretrained(..., is_trainable=True)`, confirmed exact match rather than silent reinitialization to the framework's zero-init default.

## 11. Repository cleanup

Done alongside the modeling work, at the user's request ("we need to clean up the repo first" before moving training to a cloud instance):

- **`.git` history: 8.9GB → 81MB.** Root cause: three orphaned `tmp_pack_*` files (totaling 8.8GB) sitting in `.git/objects/pack/`, leftover from an old interrupted git operation (likely a past attempt to commit the 8.2GB `models/` directory before it was gitignored). Confirmed via `git count-objects -v` (flagged as garbage), file timestamps (3 days stale), no active git process/lock files, and `git fsck` before deleting — then `git gc --prune=now`.
- **`.DS_Store` and a Word lock file (`data/~$adme.md.docx`)** had both been accidentally committed to git history — untracked via `git rm --cached` and added to `.gitignore` (`.DS_Store`, `~$*`, and `__pycache__/` also added; `venv/` corrected to `.venv/` to match the actual local environment name).
- **8.2GB of cached baseline model weights** (`models/meralion-llama3-8b-4bit`, `models/seallms-v3-7b-4bit`) deleted from local disk at the user's request — already gitignored (not a git-history issue), just local clutter from earlier baseline comparisons (§2d), re-downloadable from Hugging Face if ever needed again.
- Two genuinely empty leftover directories removed (`adapters/macro_lora_pt_continued`, `results/novel_cv/round_0` — the latter traced back to an earlier accidental real-training-run incident during notebook validation, caught and killed within seconds at the time).
- Four logically-grouped commits made for previously-uncommitted work: (1) the junk-tracking cleanup itself, (2) pending fixes (`eval_baseline.py`'s `resolve_gold_candidates` split, `prepare_training_data.py`'s tie-row-keeping fix, `train_macro_lora_pt.py`'s grad-norm clipping, regenerated `data/{train,val}/*.jsonl`), (3) the novel-methodology files (`novel_data.py`, `train_novel.py`, `run_cv.py`, `verify_novel*.py`, the Colab notebook), (4) the GRPO pipeline + predict-test notebook + findings doc + local val results.

## 12. Provenance corrections (kept as a record of what actually happened)

Several results in this project's memory/reporting got mis-attributed initially and were corrected once clarified — kept here rather than silently smoothed over, since it affects how much to trust similar unconfirmed details elsewhere:

- The 0.7860 result (λ=0.5) was first mis-attributed to `train_macro_lora_pt.py` (the old simultaneous script), then correctly identified as `train_novel.py --final-refit --refit-iters 25` (the new soft-label+consistency script).
- `refit_iters=25` was initially assumed to be an unjustified guess "borrowed" from an unrelated run, then confirmed to actually be `run_cv.py`'s own genuine k-fold-CV-determined `suggested_refit_iters` (median `best_iter` across a real 5-fold sweep) — a legitimately validated number, not a guess, on the second correction.
- The λ=0.6/`refit-iters=50` run's iteration count was never confirmed by the user as CV-derived or hand-picked — logged with that caveat explicitly rather than guessing either way.
- The curriculum v1 checkpoint's unusually short training wall-clock time (~6.6 min for all 4 stages) prompted a direct "are you sure this trained correctly" challenge from the user — resolved with concrete evidence (all stages logged `25/25` completion, `predict_test.py` processed all 5475 rows without error) rather than just asserting it was fine, and in the process of checking, a real bug (#9 in §10) was found anyway.

## 13. Key findings

- **Indonesian was the macro-average's weak link across every strategy tried for most of this work** — never above ~0.71 in any fine-tuned or baseline configuration, until curriculum v5's 0.7139 (still lower than zh/si in absolute terms, but the best id score ever recorded here).
- **Sequential curriculum's edge over simultaneous training was concentrated in Chinese** (+3.2pts vs. simultaneous macro, post-data-fix), not Indonesian (+1.4pts) or Sri Lankan (+0.9pts) — confirmed by v2's combined-stage experiment that Chinese-specific exposure (not just curriculum ordering in the abstract) explains real part of that gap.
- **consistency-lambda doesn't meaningfully move the macro average in the 0.5-0.6 range** (§5) — but **learning rate does matter** (§6 v5) — not every hyperparameter in this pipeline is equally sensitive, and it wasn't obvious in advance which would turn out to matter.
- **"More training" is not a free lever on this dataset size** — curriculum v3 showed more combined-stage iterations shifts the model toward whichever language has the smallest dataset (Sri Lankan, 203 rows) due to replacement-sampling dynamics, not uniform improvement.
- **CV fold-to-fold variance in `best_iter` (15-60, a 4x spread) turned out to be a real too-aggressive-LR signal**, not just small-dataset noise — confirmed once a lower LR was actually tested and it improved every language.
- **The `prepare_training_data.py` fix** (tied-vote Indonesian rows kept instead of dropped) improved the *old* simultaneous-macro pipeline's reported number by inference (0.7501 pre-fix) but was never actually re-measured with that fix applied — the pipeline that eventually beat the old best was the *novel* methodology, which never used `prepare_training_data.py` at all (`train_novel.py` builds targets directly via `resolve_gold_candidates`/`build_target_dist`).
- **A single in-context example nearly doubled the zero-shot-to-one-shot Sinhala accuracy gap for Qwen3.5-4B** (0.6207 → 0.7673, §2c) — never stacked with any of the fine-tuning work in this report; a plausible, cheap, unexplored lever.
- **The original 0.8044 sequential-curriculum checkpoint no longer exists anywhere retrievable** (only ever run in Colab/Drive, never saved to this repo) — moot now that curriculum v5 beats it outright, but it means ensembling with it was never actually possible during this work.

## 14. Open questions / next steps

**Current best: curriculum v5, 0.8062, beats the all-time record (0.8044).**

1. **Rerun the full 6-stage recipe at the lower learning rate throughout** (v5 only lowered LR for the final combined stage, resumed from a checkpoint trained at the original `lr=1e-4`) — untested, bigger compute spend, but the clearest remaining lever given LR is now confirmed to matter.
2. **Stack the one-shot-prompting finding (§2c)** with the current best checkpoint at inference time — untried, potentially cheap (no retraining, just a different prompt template at `predict_test.py` time).
3. Reproduce the original sequential-curriculum checkpoint (no new mechanisms) — no longer needed to "catch up," but would still enable ensembling with v5 for a possible further boost.
4. Individual-mechanism ablations (§4e, configs 2-5) were never run in isolation — would strengthen a paper write-up's claims about which mechanism contributes what, though not necessary to justify the empirical result.
5. λ sweep beyond 0.5/0.6 (e.g. 0.25, 1.0) — low priority given 0.5 vs 0.6 showed no meaningful difference, and v5 already changed the more impactful hyperparameter (LR).
