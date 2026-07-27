"""Build train/val splits from the PlurVA dev sets for the training-pipeline
smoke test (per-project decision: this burns the dev sets as a held-out
discriminator -- NOT the final production data source, see conversation).

Reuses resolve_gold/resolve_options/build_prompt from eval_baseline.py so the
training prompt format is identical to what was already validated in the
zero-shot eval (scenario-inclusion fix, Sinhala Both/0 -> C/D mapping, etc.).

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
from eval_baseline import load_rows, resolve_gold, resolve_options, build_prompt

LANGS = ["zh", "id", "si"]


def build_examples(lang):
    rows = load_rows(lang)
    examples = []
    dropped = 0
    for row in rows:
        gold = resolve_gold(lang, row["Gold_Answer"])
        if gold is None:
            dropped += 1
            continue
        options = resolve_options(lang, row)
        prompt = build_prompt(lang, row, options)
        examples.append({"prompt": prompt, "completion": f" {gold}"})
    return examples, dropped


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
        examples, dropped = build_examples(lang)
        rng.shuffle(examples)
        n_val = max(1, int(len(examples) * args.val_fraction))
        val_examples = examples[:n_val]
        train_examples = examples[n_val:]

        with open(args.train_dir / f"train_{lang}.jsonl", "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(args.val_dir / f"val_{lang}.jsonl", "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"{lang}: {len(train_examples)} train, {len(val_examples)} val, "
              f"{dropped} dropped (no majority)")


if __name__ == "__main__":
    main()
