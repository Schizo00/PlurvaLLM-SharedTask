"""Data-transform library for the soft-label / positional-consistency training
methodology (scripts/train_novel.py). No torch/model code lives here -- this
module only builds target distributions, permuted option layouts, and CV fold
assignments from the raw PlurVA dev rows.

Reuses resolve_gold_candidates/resolve_options/build_prompt/SI_FIXED_OPTION_*/
SI_GOLD_MAP from eval_baseline.py so prompt formatting and gold resolution
never drift from the already-validated zero-shot/training pipeline.
"""
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import resolve_gold_candidates

LETTERS = "ABCD"


# ---------------------------------------------------------------------------
# Mechanism A: soft target distributions from annotator disagreement
# ---------------------------------------------------------------------------

def build_target_dist(lang: str, gold_answer: str) -> dict:
    """Per-example target distribution over {A,B,C,D}, summing to 1.0.

    id: the true empirical 5-annotator vote distribution, for EVERY row (not
    just the 72/366 with no strict majority) -- e.g. "A, A, A, A, C" ->
    {A:0.8, B:0.0, C:0.2, D:0.0}. This replaces prepare_training_data.py's
    tie-duplication hack entirely: that script only special-cased the ties
    and otherwise collapsed straight to a one-hot majority label, discarding
    the vote spread on the other 294 rows.

    zh/si: no natural per-example vote distribution exists in this dataset
    (see novel_data.py's caller-side data audit: zh/si Gold_Answer never
    contains a comma). resolve_gold_candidates degenerates to a single-letter
    list in every observed case for these two languages, so this returns a
    one-hot distribution -- but if a genuine zh/si tie ever appeared, mass
    would split evenly across the tied letters instead of being dropped
    (eval_baseline.resolve_gold) or arbitrarily duplicated
    (prepare_training_data.py's current behavior). Strict generalization,
    not a special case.
    """
    gold_answer = gold_answer.strip()
    if lang == "id" and "," in gold_answer:
        votes = [v.strip() for v in gold_answer.split(",")]
        n = len(votes)
        counts = Counter(votes)
        return {l: counts.get(l, 0) / n for l in LETTERS}
    candidates = resolve_gold_candidates(lang, gold_answer)
    p = 1.0 / len(candidates)
    return {l: (p if l in candidates else 0.0) for l in LETTERS}


# ---------------------------------------------------------------------------
# Mechanism B: answer-position permutation
# ---------------------------------------------------------------------------

# zh/id: a small, reviewable pool of permutations of A,B,C,D (identity +
# derangement-ish shuffles), matching the findings doc's "3 to 4 permutations
# per question" -- deliberately not sampled uniformly from all 24 possible
# permutations, so the actual permutation set used in training stays small
# enough to hand-inspect.
_FIXED_PERMUTATION_POOL = {
    "zh": ["ABCD", "BADC", "CDAB", "DCBA"],
    "id": ["ABCD", "BADC", "CDAB", "DCBA"],
}


def sample_permutation(lang: str, rng: random.Random, pool_size: int = 4) -> dict:
    """Returns old_letter -> new_letter. Deterministic given rng state.

    si: only ever swaps A/B (2 permutations total) -- C/D are the fixed
    "Both correct"/"neither correct" meta-options (SI_FIXED_OPTION_C/D), not
    real question-specific content, so permuting them would be meaningless
    (and the findings doc's own augmentation proposal explicitly only
    permutes A/B for Sinhala for this reason).
    """
    if lang == "si":
        if rng.random() < 0.5:
            return {"A": "A", "B": "B", "C": "C", "D": "D"}
        return {"A": "B", "B": "A", "C": "C", "D": "D"}
    perms = _FIXED_PERMUTATION_POOL[lang][:pool_size]
    chosen = rng.choice(perms)
    return dict(zip(LETTERS, chosen))


def permute_example(options: dict, target_dist: dict, perm: dict):
    """Apply perm (old_letter -> new_letter) to an options dict and a target
    distribution, returning the permuted (options, target_dist) pair. Option
    TEXT is preserved and relabeled; target probability mass moves with its
    letter's content to the new letter."""
    new_options = {perm[l]: options[l] for l in LETTERS}
    new_target = defaultdict(float)
    for l, p in target_dist.items():
        new_target[perm[l]] += p
    return new_options, dict(new_target)


# ---------------------------------------------------------------------------
# CV fold assignment
# ---------------------------------------------------------------------------

def assign_folds(rows: list, lang: str, k: int, seed: int) -> list:
    """Returns a list of fold indices (0..k-1), one per row, in the same
    order as `rows`. Stratified by majority-vote gold letter (via
    resolve_gold_candidates -- ties use the first sorted tied letter purely
    for stratification grouping, not for the actual training target) so each
    fold keeps roughly balanced label counts despite tiny per-fold sizes.
    Deterministic: same (rows, lang, k, seed) always produces the same
    assignment, required since each CV round runs as its own subprocess
    (scripts/run_cv.py)."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        candidates = resolve_gold_candidates(lang, row["Gold_Answer"])
        groups[candidates[0]].append(i)

    fold_of = [None] * len(rows)
    rng = random.Random(seed)
    for letter in sorted(groups):
        idxs = groups[letter]
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            fold_of[i] = j % k
    return fold_of
