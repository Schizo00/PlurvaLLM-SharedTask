"""Per-stage, CV-validated version of the winning curriculum recipe
(scripts/train_novel_curriculum.py's v5: zh -> id -> si -> zh -> id -> a
final "combined" simultaneous stage with zh's loss upweighted, at the
lower learning rate that produced the current best hidden-test result,
0.8062, see NOVEL_METHODOLOGY_REPORT.md).

Every iteration count used to produce v5 was a single number *borrowed*
from an unrelated CV sweep (a different learning rate, a different
training regime -- simultaneous, not curriculum), never actually
validated for curriculum mode's own stage structure. This script fixes
that: for each of the 6 stages ("slots"), in order, it runs real k-fold
CV -- continuing from the FIXED adapter the previous slot's refit
produced -- to find that slot's own validated best_iter, then refits on
100% of that slot's data for that many iterations before moving to the
next slot. No data is permanently sacrificed to validation (every row
gets a turn across the k folds, same principle as the project's original
CV+refit protocol) and no iteration count here is borrowed from anywhere
else.

Per explicit instruction: does NOT modify train_novel.py,
train_novel_curriculum.py, run_cv.py, or novel_data.py -- train_novel_curriculum.py
already produced the best submittable result and must stay runnable
as-is. Everything here is new, reusing those scripts only via import.
"""
import argparse
import gc
import json
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import load_rows
from novel_data import LETTERS, assign_folds
from train_macro_lora_pt import LANGS
from train_novel import (
    MAX_GRAD_NORM_DEFAULT,
    SoftPromptCompletionDataset,
    build_lang_examples,
    lang_step,
    letter_token_ids,
    per_language_soft_validate,
)

SLOTS = ["zh", "id", "si", "zh", "id", "combined"]  # matches curriculum v5's winning structure


def slot_dir(out_dir: Path, slot_idx: int) -> Path:
    return out_dir / f"slot_{slot_idx}_{SLOTS[slot_idx]}"


def build_datasets(langs, rows_by_lang, tokenizer, args):
    raw_examples_by_lang = {
        lang: build_lang_examples(lang, rows_by_lang[lang], hard_labels=args.hard_labels)
        for lang in langs
    }
    datasets = {
        lang: SoftPromptCompletionDataset(raw_examples_by_lang[lang], tokenizer, args.max_seq_length)
        for lang in langs
    }
    return raw_examples_by_lang, datasets


def load_model(args, device, dtype, prev_adapter_path):
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    if prev_adapter_path is None:
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    else:
        model = PeftModel.from_pretrained(base_model, str(prev_adapter_path), is_trainable=True)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    model.train()
    return model, base_model


def free_model(model, base_model, optimizer, device):
    del model, base_model, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def one_train_step(model, tokenizer, langs, datasets, raw_examples_by_lang, letter_ids, device, rng, args):
    """One optimizer step: a single lang_step for a single-language slot, or
    a simultaneous step across all LANGS (zh upweighted) for the combined
    slot -- mirrors train_novel_curriculum.py's run_combined_stage inner
    loop, which is intentionally NOT imported since that script is left
    untouched per the plan; kept minimal and only duplicated here."""
    if len(langs) == 1:
        lang = langs[0]
        lang_loss, metric, c_loss_value = lang_step(
            model, tokenizer, datasets[lang], raw_examples_by_lang[lang], args.per_lang_batch_size,
            letter_ids, device, rng, args.max_seq_length, tokenizer.pad_token_id, args.consistency_lambda,
        )
        lang_loss.backward()
        return lang_loss.item(), metric, c_loss_value
    total_loss, total_metric, total_closs = 0.0, 0.0, 0.0
    weights = {lang: (args.zh_weight if lang == "zh" else 1.0) for lang in langs}
    for lang in langs:
        lang_loss, metric, c_loss_value = lang_step(
            model, tokenizer, datasets[lang], raw_examples_by_lang[lang], args.per_lang_batch_size,
            letter_ids, device, rng, args.max_seq_length, tokenizer.pad_token_id, args.consistency_lambda,
        )
        (lang_loss * weights[lang] / len(langs)).backward()
        total_loss += lang_loss.item()
        total_metric += metric
        total_closs += c_loss_value
    return total_loss / len(langs), total_metric / len(langs), total_closs / len(langs)


def run_cv_fold(fold_idx, slot_idx, langs, train_rows_by_lang, val_rows_by_lang, tokenizer, letter_ids,
                 device, dtype, args, prev_adapter_path):
    """Trains one CV fold from the fixed prev_adapter_path, early-stopping
    on this fold's held-out validation. Returns best_iter (diagnostic only
    -- this fold's checkpoint is discarded afterward, never reused)."""
    train_raw, train_datasets = build_datasets(langs, train_rows_by_lang, tokenizer, args)
    _, val_datasets = build_datasets(langs, val_rows_by_lang, tokenizer, args)
    model, base_model = load_model(args, device, dtype, prev_adapter_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed + slot_idx * 100 + fold_idx)

    best_val_loss, best_iter, stale_evals = float("inf"), 0, 0
    for it in range(1, args.cv_iters_ceiling + 1):
        optimizer.zero_grad()
        one_train_step(model, tokenizer, langs, train_datasets, train_raw, letter_ids, device, rng, args)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if it % args.steps_per_eval == 0:
            val_loss, val_metric, _, _ = per_language_soft_validate(
                model, val_datasets, args.per_lang_batch_size, args.val_batches, letter_ids,
                tokenizer.pad_token_id, device, rng,
            )
            tqdm.write(f"[slot {slot_idx}/{SLOTS[slot_idx]} fold {fold_idx}][{it}] "
                       f"val_loss={val_loss:.4f} val_metric={val_metric:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, best_iter, stale_evals = val_loss, it, 0
            else:
                stale_evals += 1
                if args.patience > 0 and stale_evals >= args.patience:
                    tqdm.write(f"[slot {slot_idx}/{SLOTS[slot_idx]} fold {fold_idx}] early stop at {it}")
                    break

    free_model(model, base_model, optimizer, device)
    return best_iter, best_val_loss


def run_refit(slot_idx, langs, rows_by_lang, tokenizer, letter_ids, device, dtype, args,
              prev_adapter_path, iters, out_dir):
    raw_examples_by_lang, datasets = build_datasets(langs, rows_by_lang, tokenizer, args)
    model, base_model = load_model(args, device, dtype, prev_adapter_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed + slot_idx * 100 + 999)

    for it in range(1, iters + 1):
        optimizer.zero_grad()
        loss, metric, closs = one_train_step(
            model, tokenizer, langs, datasets, raw_examples_by_lang, letter_ids, device, rng, args,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if it % args.steps_per_report == 0 or it == iters:
            tqdm.write(f"[slot {slot_idx}/{SLOTS[slot_idx]} refit][{it}/{iters}] "
                       f"loss={loss:.4f} metric={metric:.4f} consistency={closs:.4f}")

    adapter_path = slot_dir(out_dir, slot_idx) / "adapter"
    model.save_pretrained(str(adapter_path))
    free_model(model, base_model, optimizer, device)
    return adapter_path


def run_slot(slot_idx, tokenizer, letter_ids, device, dtype, args, prev_adapter_path, out_dir):
    slot_lang = SLOTS[slot_idx]
    langs = LANGS if slot_lang == "combined" else [slot_lang]
    rows_by_lang = {lang: load_rows(lang) for lang in langs}
    folds_by_lang = {lang: assign_folds(rows_by_lang[lang], lang, args.cv_folds, args.fold_seed) for lang in langs}

    best_iters = []
    for fold_idx in range(args.cv_folds):
        train_rows = {lang: [r for r, f in zip(rows_by_lang[lang], folds_by_lang[lang]) if f != fold_idx]
                      for lang in langs}
        val_rows = {lang: [r for r, f in zip(rows_by_lang[lang], folds_by_lang[lang]) if f == fold_idx]
                    for lang in langs}
        best_iter, best_val_loss = run_cv_fold(
            fold_idx, slot_idx, langs, train_rows, val_rows, tokenizer, letter_ids, device, dtype,
            args, prev_adapter_path,
        )
        best_iters.append(max(best_iter, 1))  # guard against a fold that never improved past iter 0
        print(f"[slot {slot_idx}/{slot_lang}] fold {fold_idx}: best_iter={best_iter} "
              f"best_val_loss={best_val_loss:.4f}")

    median_iter = int(statistics.median(best_iters))
    print(f"[slot {slot_idx}/{slot_lang}] CV done: best_iters={best_iters} -> refit_iters={median_iter}")

    adapter_path = run_refit(
        slot_idx, langs, rows_by_lang, tokenizer, letter_ids, device, dtype, args,
        prev_adapter_path, median_iter, out_dir,
    )
    summary = {
        "slot": slot_idx, "lang": slot_lang, "cv_folds": args.cv_folds,
        "best_iters": best_iters, "refit_iters": median_iter,
        "consistency_lambda": args.consistency_lambda,
        "zh_weight": args.zh_weight if slot_lang == "combined" else None,
        "learning_rate": args.learning_rate, "adapter_path": str(adapter_path),
        "finished": True,
    }
    (slot_dir(out_dir, slot_idx) / "summary.json").write_text(json.dumps(summary, indent=2))
    return adapter_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM_DEFAULT)
    ap.add_argument("--consistency-lambda", type=float, default=0.5)
    ap.add_argument("--zh-weight", type=float, default=1.5, help="combined slot only")
    ap.add_argument("--hard-labels", action="store_true")
    ap.add_argument("--per-lang-batch-size", type=int, default=8)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-iters-ceiling", type=int, default=300)
    ap.add_argument("--steps-per-eval", type=int, default=5)
    ap.add_argument("--steps-per-report", type=int, default=5)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=5e-5,
                     help="defaults to the lower LR that produced curriculum v5 (0.8062), "
                          "not train_novel_curriculum.py's 1e-4 default.")
    ap.add_argument("--resume-from-slot", type=int, default=0,
                     help="skip slots before this index, resuming from "
                          "{out-dir}/slot_{N-1}_*/adapter (must already exist).")
    ap.add_argument("--out-dir", default="results/novel_curriculum_cv")
    args = ap.parse_args()

    if not (0 <= args.resume_from_slot <= len(SLOTS)):
        ap.error(f"--resume-from-slot must be between 0 and {len(SLOTS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_adapter_path = None
    if args.resume_from_slot > 0:
        prev_adapter_path = slot_dir(out_dir, args.resume_from_slot - 1) / "adapter"
        if not prev_adapter_path.exists():
            ap.error(f"--resume-from-slot {args.resume_from_slot} requires "
                     f"{prev_adapter_path} to already exist")
        print(f"Resuming from slot {args.resume_from_slot}, using {prev_adapter_path}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device == "cuda" and not torch.cuda.is_bf16_supported() else torch.bfloat16
    print(f"Loading {args.model_id} on {device} ({dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = torch.tensor([letter_token_ids(tokenizer)[l] for l in LETTERS], device=device)

    for slot_idx in range(args.resume_from_slot, len(SLOTS)):
        prev_adapter_path = run_slot(
            slot_idx, tokenizer, letter_ids, device, dtype, args, prev_adapter_path, out_dir,
        )

    print(f"Done. Final adapter: {prev_adapter_path}")


if __name__ == "__main__":
    main()
