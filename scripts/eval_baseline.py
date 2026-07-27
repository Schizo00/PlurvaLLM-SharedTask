"""Zero-shot MCQ baseline eval for candidate base models against PlurVA dev sets.

Usage:
    python eval_baseline.py <mlx_model_path> <lang: zh|id> [--n N] [--seed S] [--out path.jsonl]
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LANG_FILES = {
    "zh": DATA_DIR / "chinese_dev.jsonl",
    "id": DATA_DIR / "indonesian_dev.jsonl",
    "si": DATA_DIR / "sri_lankan_dev.jsonl",
}

LETTER_RE = re.compile(r"\b([ABCD])\b")
THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)

SI_FIXED_OPTION_C = "පිළිතුරු දෙකම නිවැරදියි."
SI_FIXED_OPTION_D = "පිළිතුරු දෙකම නිවැරදි නොවේ."

PROMPT_TEMPLATES = {
    "zh": """You are a Simplified Chinese Expert for answering multiple-choice questions.
    You need to always evaluate these questions based on the Chinese context and provide the most accurate answer.
    You should only respond with the correct option text, without any additional explanation or commentary.
{scenario_line}Question: {question}
Option A: {option_a}
Option B: {option_b}
Option C: {option_c}
Option D: {option_d}
Return only the correct option text. Expected output is either the text of 'A', 'B', 'C', or 'D'.""",
    "id": """You are an Indonesian Expert for answering multiple-choice questions.
    You need to always evaluate these questions based on the Indonesian context and provide the most accurate answer.
    You should only respond with the correct option text, without any additional explanation or commentary.
{scenario_line}Question: {question}
Option A: {option_a}
Option B: {option_b}
Option C: {option_c}
Option D: {option_d}
Return only the correct option text. Expected output is either the text of 'A', 'B', 'C', or 'D'.""",
    "si": """You are a Sinhala Expert for answering multiple-choice questions.
    You need to always evaluate these questions based on the Sri Lankan context and provide the most accurate answer.
    You should only respond with the correct option text, without any additional explanation or commentary.
{scenario_line}Question: {question}
Option A: {option_a}
Option B: {option_b}
Option C: {option_c}
Option D: {option_d}
Return only the correct option text. Expected output is either the text of 'A', 'B', 'C', or 'D'.""",
}


def load_rows(lang: str):
    rows = []
    with open(LANG_FILES[lang], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


SI_GOLD_MAP = {"Both": "C", "0": "D"}


def resolve_gold(lang: str, gold_answer: str):
    """Handle single-letter gold answers, Indonesian's comma-separated
    annotator votes (majority vote; drop rows with no strict majority), and
    Sri Lankan's 'Both'/'0' labels (map to the fixed C/D meta-options)."""
    gold_answer = gold_answer.strip()
    if lang == "si" and gold_answer in SI_GOLD_MAP:
        return SI_GOLD_MAP[gold_answer]
    if "," in gold_answer:
        votes = [v.strip() for v in gold_answer.split(",")]
        counts = Counter(votes)
        top, top_n = counts.most_common(1)[0]
        # strict majority required (ties -> drop)
        if list(counts.values()).count(top_n) > 1:
            return None
        return top
    return gold_answer


def resolve_options(lang, row):
    """Return {letter: option_text} actually presented in the prompt."""
    options = {
        "A": row.get("Option_A", ""),
        "B": row.get("Option_B", ""),
        "C": row.get("Option_C", ""),
        "D": row.get("Option_D", ""),
    }
    if lang == "si":
        options["C"] = SI_FIXED_OPTION_C
        options["D"] = SI_FIXED_OPTION_D
    return options


def build_prompt(lang, row, options):
    scenario = row.get("Scenario", "").strip()
    scenario_line = f"Scenario: {scenario}\n" if scenario else ""
    return PROMPT_TEMPLATES[lang].format(
        scenario_line=scenario_line,
        question=row["Question"],
        option_a=options["A"],
        option_b=options["B"],
        option_c=options["C"],
        option_d=options["D"],
    )


def strip_thinking(text: str) -> str:
    """Drop a leading <think>...</think> reasoning block if the model emitted
    one despite enable_thinking=False (mirrors the Gemma-4 gotcha)."""
    return THINK_RE.sub("", text, count=1).strip()


def extract_letter(text: str, valid_letters, options=None):
    text = strip_thinking(text)
    # Prefer the first standalone valid letter found.
    for m in LETTER_RE.finditer(text):
        if m.group(1) in valid_letters:
            return m.group(1)
    # Fallback: model followed the "respond with option text" instruction
    # literally instead of returning the letter. Match against option text.
    if options is not None:
        norm = text.strip().lower()
        if norm:
            for letter in valid_letters:
                opt_text = options.get(letter, "").strip().lower()
                if opt_text and (opt_text in norm or norm in opt_text):
                    return letter
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("lang", choices=["zh", "id", "si"])
    ap.add_argument("--backend", choices=["lm", "vlm"], default="lm",
                     help="mlx_lm (text-only models) or mlx_vlm (natively multimodal, e.g. Qwen3.5)")
    ap.add_argument("--n", type=int, default=None, help="sample size (default: all rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--enable-thinking", action="store_true",
                     help="leave the model's reasoning/thinking mode on (default: try to disable it)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.lang)
    if args.n is not None and args.n < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]

    print(f"Loading model from {args.model_path} (backend={args.backend}) ...", file=sys.stderr)
    if args.backend == "vlm":
        from mlx_vlm import load, generate as vlm_generate
        model, tok_or_proc = load(args.model_path)
        model_config = model.config
    else:
        from mlx_lm import load, generate as lm_generate
        model, tok_or_proc = load(args.model_path)

    out_path = args.out or f"results_{Path(args.model_path).name}_{args.lang}.jsonl"
    out_f = open(out_path, "w", encoding="utf-8")

    correct = 0
    scored = 0
    dropped_no_majority = 0
    unparseable = 0
    t0 = time.time()

    for i, row in enumerate(tqdm(rows, desc="Evaluating")):
        gold = resolve_gold(args.lang, row["Gold_Answer"])
        if gold is None:
            dropped_no_majority += 1
            continue

        options = resolve_options(args.lang, row)
        valid_letters = {l for l in "ABCD" if options[l]}
        prompt = build_prompt(args.lang, row, options)

        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {} if args.enable_thinking else {"enable_thinking": False}

        if args.backend == "vlm":
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
            try:
                chat_prompt = vlm_apply_chat_template(
                    tok_or_proc, model_config, messages,
                    add_generation_prompt=True, num_images=0, **template_kwargs,
                )
            except Exception:
                chat_prompt = prompt
            response = vlm_generate(
                model, tok_or_proc, prompt=chat_prompt, image=None,
                max_tokens=args.max_tokens, verbose=False,
            )
            response = response.text if hasattr(response, "text") else str(response)
        else:
            try:
                chat_prompt = tok_or_proc.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **template_kwargs
                )
            except TypeError:
                chat_prompt = tok_or_proc.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                chat_prompt = prompt
            response = lm_generate(
                model, tok_or_proc, prompt=chat_prompt, max_tokens=args.max_tokens, verbose=False
            )

        pred = extract_letter(response, valid_letters, options)

        is_correct = pred == gold
        scored += 1
        if pred is None:
            unparseable += 1
        if is_correct:
            correct += 1

        out_f.write(
            json.dumps(
                {
                    "ID": row["ID"],
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "raw_response": response,
                }
            )
            + "\n"
        )
        out_f.flush()

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc_so_far = correct / scored if scored else 0.0
            print(
                f"[{i+1}/{len(rows)}] acc_so_far={acc_so_far:.3f} "
                f"elapsed={elapsed:.0f}s",
                file=sys.stderr,
            )

    out_f.close()

    acc = correct / scored if scored else 0.0
    print(
        json.dumps(
            {
                "model": args.model_path,
                "lang": args.lang,
                "n_rows": len(rows),
                "scored": scored,
                "dropped_no_majority": dropped_no_majority,
                "unparseable": unparseable,
                "correct": correct,
                "accuracy": acc,
                "elapsed_sec": time.time() - t0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
