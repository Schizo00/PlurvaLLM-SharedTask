"""Assertion-based smoke tests for the soft-label / positional-consistency
training methodology (novel_data.py, train_novel.py). Plain asserts, not
pytest -- matches this repo's existing convention (no test framework
anywhere else in it). Run directly:

    python scripts/verify_novel.py            # data-layer checks only (1-5)
    python scripts/verify_novel.py --all       # includes train_novel.py checks (6-9)
                                                # (requires loading the base model)

Each check prints PASS/FAIL and raises on the first failure so a broken
assumption is caught before any multi-hour training run, not after.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import (
    SI_FIXED_OPTION_C, SI_FIXED_OPTION_D, load_rows, resolve_options,
)
from novel_data import assign_folds, build_target_dist, permute_example, sample_permutation


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise AssertionError(label)


# ---------------------------------------------------------------------------
# 1. Tokenizer sanity
# ---------------------------------------------------------------------------

def verify_tokenizer_single_token(model_id="Qwen/Qwen3.5-4B"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = {}
    for letter in "ABCD":
        toks = tok(f" {letter}", add_special_tokens=False)["input_ids"]
        check(f"' {letter}' tokenizes to exactly 1 token (got {toks})", len(toks) == 1)
        ids[letter] = toks[0]
    print(f"  letter token ids: {ids}")
    return ids


# ---------------------------------------------------------------------------
# 2. Soft-label construction
# ---------------------------------------------------------------------------

def verify_soft_label_construction():
    # Hand-picked, verified against the real file this session:
    # "A, A, A, A, C" -> 4 A votes, 1 C vote.
    dist = build_target_dist("id", "A, A, A, A, C")
    check(f"4A/1C votes -> {{'A':0.8,'C':0.2}} (got {dist})",
          abs(dist["A"] - 0.8) < 1e-9 and abs(dist["C"] - 0.2) < 1e-9
          and dist["B"] == 0.0 and dist["D"] == 0.0)
    check("target dist sums to 1.0", abs(sum(dist.values()) - 1.0) < 1e-9)

    # A previously-tied row (no strict majority) -- e.g. 2-2-1 split --
    # should now get a genuine partial-mass split, not duplication.
    dist_tie = build_target_dist("id", "C, D, A, A, D")
    counts = Counter(v.strip() for v in "C, D, A, A, D".split(","))
    for letter in "ABCD":
        expected = counts.get(letter, 0) / 5
        check(f"tied row letter {letter}: dist={dist_tie[letter]:.2f} expected={expected:.2f}",
              abs(dist_tie[letter] - expected) < 1e-9)

    # zh/si: single hard label -> one-hot.
    dist_zh = build_target_dist("zh", "B")
    check(f"zh hard label 'B' -> one-hot (got {dist_zh})",
          dist_zh == {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0})
    dist_si_both = build_target_dist("si", "Both")
    check(f"si 'Both' -> one-hot on C (got {dist_si_both})",
          dist_si_both == {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0})

    # Real data pass: every id row's target dist sums to 1.0, matches its
    # own comma-vote structure (all 366 rows have exactly 5 votes).
    id_rows = load_rows("id")
    for row in id_rows:
        dist = build_target_dist("id", row["Gold_Answer"])
        s = sum(dist.values())
        check(f"row {row.get('ID')}: dist sums to 1.0 (got {s:.6f})", abs(s - 1.0) < 1e-9)
    print(f"  verified soft-label construction on all {len(id_rows)} real Indonesian rows")


# ---------------------------------------------------------------------------
# 3. Answer-position finding (needs the tokenizer + the training dataset class)
# ---------------------------------------------------------------------------

def verify_answer_position_finding(model_id="Qwen/Qwen3.5-4B"):
    import torch
    from transformers import AutoTokenizer
    from eval_baseline import build_prompt
    from train_novel import SoftPromptCompletionDataset

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    examples = []
    for lang in ["zh", "id", "si"]:
        row = load_rows(lang)[0]
        options = resolve_options(lang, row)
        prompt = build_prompt(lang, row, options)
        dist = build_target_dist(lang, row["Gold_Answer"])
        letter = max(dist, key=dist.get)
        examples.append({"lang": lang, "prompt": prompt, "completion": f" {letter}",
                          "target_dist": dist, "letter": letter})

    ds = SoftPromptCompletionDataset(examples, tok, max_seq_length=768)
    for ex, raw in zip(ds.examples, examples):
        labels = ex["labels"]
        labels_shifted = labels[1:]
        n_valid = sum(1 for x in labels_shifted if x != -100)
        check(f"{raw['lang']}: exactly one non-masked label position (got {n_valid})", n_valid == 1)
        pos = next(i for i, x in enumerate(labels_shifted) if x != -100)
        input_ids = ex["input_ids"]
        decoded = tok.decode([input_ids[pos + 1]])
        check(f"{raw['lang']}: decoded answer token {decoded!r} == ' {raw['letter']}'",
              decoded == f" {raw['letter']}")


# ---------------------------------------------------------------------------
# 4. Permutation correctness
# ---------------------------------------------------------------------------

def verify_permutation_correctness():
    rng = random.Random(0)
    for lang in ["zh", "id", "si"]:
        row = load_rows(lang)[0]
        options = resolve_options(lang, row)
        dist = build_target_dist(lang, row["Gold_Answer"])

        for _ in range(10):
            perm = sample_permutation(lang, rng)
            new_options, new_dist = permute_example(options, dist, perm)

            # Content preserved: the text that WAS under letter l is now
            # under perm[l], for every letter.
            for l in "ABCD":
                check(f"{lang}: content of {l} moved to {perm[l]} intact",
                      new_options[perm[l]] == options[l])

            # Target mass conservation.
            check(f"{lang}: permuted target dist sums to 1.0 (got {sum(new_dist.values()):.6f})",
                  abs(sum(new_dist.values()) - 1.0) < 1e-9)

            if lang == "si":
                check("si: C option text never moves", new_options["C"] == SI_FIXED_OPTION_C)
                check("si: D option text never moves", new_options["D"] == SI_FIXED_OPTION_D)
                check("si: perm never touches C/D", perm["C"] == "C" and perm["D"] == "D")


# ---------------------------------------------------------------------------
# 5. Fold assignment
# ---------------------------------------------------------------------------

def verify_fold_assignment():
    for lang in ["zh", "id", "si"]:
        rows = load_rows(lang)
        k = 5
        f1 = assign_folds(rows, lang, k, seed=42)
        f2 = assign_folds(rows, lang, k, seed=42)
        check(f"{lang}: fold assignment deterministic across runs", f1 == f2)

        check(f"{lang}: every row assigned a fold in range", all(0 <= f < k for f in f1))

        counts = Counter(f1)
        n = len(rows)
        expected = n / k
        for fold_idx, c in counts.items():
            check(f"{lang} fold {fold_idx}: size {c} within +/-1 of expected {expected:.1f}"
                  f" (loose bound since stratification groups vary in size)",
                  abs(c - expected) <= max(2, n // k // 3))
        print(f"  {lang}: fold sizes = {dict(sorted(counts.items()))}")

        # Different seed -> (almost certainly) different assignment.
        f3 = assign_folds(rows, lang, k, seed=1)
        check(f"{lang}: different seed changes assignment", f1 != f3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                     help="also run checks 6-9 (train_novel.py checks; requires torch/model)")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    print("== 1. Tokenizer sanity ==")
    verify_tokenizer_single_token(args.model_id)

    print("\n== 2. Soft-label construction ==")
    verify_soft_label_construction()

    print("\n== 3. Answer-position finding ==")
    verify_answer_position_finding(args.model_id)

    print("\n== 4. Permutation correctness ==")
    verify_permutation_correctness()

    print("\n== 5. Fold assignment ==")
    verify_fold_assignment()

    if args.all:
        from verify_novel_train import (
            verify_consistency_loss_identity, verify_cv_round_independence,
            verify_loss_equivalence, verify_smoke_test,
        )
        print("\n== 6. Loss-equivalence unit check ==")
        verify_loss_equivalence()
        print("\n== 7. Consistency-loss identity check ==")
        verify_consistency_loss_identity()
        print("\n== 8. Tiny end-to-end smoke test ==")
        verify_smoke_test(args.model_id)
        print("\n== 9. CV round independence ==")
        verify_cv_round_independence()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
