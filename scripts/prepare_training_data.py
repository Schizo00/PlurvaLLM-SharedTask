"""Build train/val splits from the PlurVA dev sets for the training-pipeline
smoke test (per-project decision: this burns the dev sets as a held-out
discriminator -- NOT the final production data source, see conversation).

Reuses resolve_gold_candidates/resolve_options/build_prompt from eval_baseline.py
so the training prompt format is identical to what was already validated in the
zero-shot eval (scenario-inclusion fix, Sinhala Both/0 -> C/D mapping, etc.).

Indonesian rows with no strict-majority annotator vote (a tie between two
letters, e.g. "C, D, A, A, D") aren't dropped here the way eval_baseline.py's
scoring does -- each tied row becomes one training example per tied letter,
sharing the same prompt, so the model sees both plausible answers instead of
losing the row's signal entirely.

Output schema: {"prompt": <raw un-templated prompt>, "completion": " <letter>"}
-- chat-template application happens later, in train_macro_lora_pt.py's
dataset class, not here.
"""
import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import load_rows, resolve_gold_candidates, resolve_options, build_prompt

LANGS = ["zh", "id", "si"]


def build_example_groups(lang):
    """Return one group per row (a list of examples sharing that row's prompt)
    so tied rows -- which expand to 2 examples -- can be kept together on the
    same side of the train/val split rather than shuffled independently."""
    rows = load_rows(lang)
    groups = []
    tied_rows = 0
    for row in rows:
        candidates = resolve_gold_candidates(lang, row["Gold_Answer"])
        if len(candidates) > 1:
            tied_rows += 1
        options = resolve_options(lang, row)
        prompt = build_prompt(lang, row, options)
        groups.append([{"prompt": prompt, "completion": f" {letter}"} for letter in candidates])
    return groups, tied_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/train", type=Path)
    ap.add_argument("--val-dir", default="data/val", type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for lang in LANGS:
        groups, tied_rows = build_example_groups(lang)
        rng.shuffle(groups)
        n_val_groups = max(1, int(len(groups) * args.val_fraction))
        val_groups = groups[:n_val_groups]
        train_groups = groups[n_val_groups:]

        train_examples = [ex for group in train_groups for ex in group]
        val_examples = [ex for group in val_groups for ex in group]

        with open(args.train_dir / f"train_{lang}.jsonl", "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(args.val_dir / f"val_{lang}.jsonl", "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"{lang}: {len(train_examples)} train examples ({len(train_groups)} rows), "
              f"{len(val_examples)} val examples ({len(val_groups)} rows), "
              f"{tied_rows} tied rows duplicated")


if __name__ == "__main__":
    main()
