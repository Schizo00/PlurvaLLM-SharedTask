"""Generate shared-task submission predictions for the (unlabeled) test sets,
using a base model + a PEFT LoRA adapter (e.g. adapters/macro_lora_pt/best).

Reuses the exact prompt-building and letter-extraction logic from
scripts/eval_baseline.py so test-time prompting matches training/eval exactly.
Test rows (data/test/*_test_without_gold.jsonl) have no Gold_Answer field, so
this script only builds prompts and extracts predictions -- no scoring.

Output: one JSON object per line, in the shared-task submission format:
    {"dataset": "chinese"|"indonesian"|"sri_lankan", "id": <ID>, "LLM_Output": "A"|"B"|"C"|"D"|"Both"|"0"}

Usage:
    python scripts/predict_test.py \\
        --model-id Qwen/Qwen3.5-4B --adapter-path adapters/macro_lora_pt/best \\
        --out results/test_submission.jsonl

    # quick smoke test on a handful of rows
    python scripts/predict_test.py --langs zh --n 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import resolve_options, build_prompt, extract_letter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
TEST_FILES = {
    "zh": DATA_DIR / "chinese_test_without_gold.jsonl",
    "id": DATA_DIR / "indonesian_test_without_gold.jsonl",
    "si": DATA_DIR / "sri_lankan_test_without_gold.jsonl",
}
DATASET_NAMES = {"zh": "chinese", "id": "indonesian", "si": "sri_lankan"}
LANGS = ["zh", "id", "si"]

# Inverse of eval_baseline.SI_GOLD_MAP: internal C/D letters (the fixed
# "both correct" / "neither correct" meta-options used during training/eval)
# map back to the shared task's expected "Both"/"0" strings for Sri Lankan.
SI_OUTPUT_MAP = {"C": "Both", "D": "0"}


def load_test_rows(lang):
    rows = []
    with open(TEST_FILES[lang], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_submission_output(lang, letter):
    if lang == "si":
        return SI_OUTPUT_MAP.get(letter, letter)
    return letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter-path", default="adapters/macro_lora_pt/best",
                     help="PEFT LoRA adapter dir, or '' to eval the base model zero-shot")
    ap.add_argument("--langs", nargs="+", choices=LANGS, default=LANGS)
    ap.add_argument("--n", type=int, default=None, help="rows per language (default: all)")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--enable-thinking", action="store_true",
                     help="leave the model's reasoning/thinking mode on (default: try to disable it)")
    ap.add_argument("--out", default="results/test_submission.jsonl")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device == "cuda" and not torch.cuda.is_bf16_supported() else torch.bfloat16

    print(f"Loading {args.model_id} on {device} ({dtype}) ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    if args.adapter_path:
        from peft import PeftModel
        print(f"Applying LoRA adapter from {args.adapter_path} ...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")

    total = 0
    unparseable = 0
    t0 = time.time()

    for lang in args.langs:
        rows = load_test_rows(lang)
        if args.n is not None:
            rows = rows[: args.n]

        for row in tqdm(rows, desc=f"Predicting {lang}"):
            options = resolve_options(lang, row)
            valid_letters = {l for l in "ABCD" if options[l]}
            prompt = build_prompt(lang, row, options)

            messages = [{"role": "user", "content": prompt}]
            template_kwargs = {} if args.enable_thinking else {"enable_thinking": False}
            try:
                chat_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **template_kwargs
                )
            except TypeError:
                chat_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                chat_prompt = prompt

            inputs = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            pred = extract_letter(response, valid_letters, options)
            total += 1
            if pred is None:
                # Submission format has no "unknown" value -- default to "A"
                # rather than emit an invalid row; unparseable count below
                # tells you how often this fallback fired.
                unparseable += 1
                pred = "A"

            out_f.write(
                json.dumps(
                    {
                        "dataset": DATASET_NAMES[lang],
                        "id": row["ID"],
                        "LLM_Output": to_submission_output(lang, pred),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

    out_f.close()
    elapsed = time.time() - t0
    print(
        f"Done: {total} rows written to {out_path}, "
        f"{unparseable} unparseable (defaulted to 'A'), elapsed={elapsed:.0f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
