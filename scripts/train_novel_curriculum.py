"""Curriculum variant of the novel methodology: combines the team's known-good
sequential-curriculum recipe (zh -> id -> aug-si -> aug-zh, continuing from
each prior stage's adapter, ~0.8044 macro on hidden test) with the two
mechanisms from scripts/train_novel.py (soft-label distillation from
annotator disagreement + positional-invariance consistency regularization).
Neither has been tested together before -- every prior soft-label/consistency
run trained all three languages simultaneously from a fresh base model.

Design choices (confirmed with the user, see the plan's curriculum-mode
addendum):
  - Each stage trains ONLY on that stage's language (not cumulative replay of
    prior languages) -- matches the memory record's description of the
    original recipe ("each stage mostly moves only the newly-added language,
    almost no forgetting") and is cheaper.
  - --consistency-lambda is applied uniformly at every stage, replacing the
    original recipe's ad-hoc literal-permuted-row-duplication idea of
    "augmentation" with the actual novel mechanism -- this is the hypothesis
    under test, not just a replication of the old recipe.
  - Fixed --iters-per-stage (default 25, the CV-validated iters from the
    lambda=0.5 sweep in scripts/run_cv.py), no per-stage CV/early-stopping --
    kept to a single run given limited compute.

Reuses scripts/train_novel.py's tokenization/loss/step machinery directly
(letter_token_ids, SoftPromptCompletionDataset, build_lang_examples,
lang_step) and scripts/train_sft_then_grpo.py's checkpoint-continuation
pattern (PeftModel.from_pretrained(..., is_trainable=True)).
"""
import argparse
import json
import os
import random
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
from novel_data import LETTERS
from train_novel import (
    MAX_GRAD_NORM_DEFAULT,
    SoftPromptCompletionDataset,
    build_lang_examples,
    lang_step,
    letter_token_ids,
)

STAGES = ["zh", "id", "si", "zh"]  # zh revisited in stage 3 ("aug-zh")


def stage_dir(out_dir: Path, stage_idx: int, lang: str) -> Path:
    return out_dir / f"stage_{stage_idx}_{lang}"


def run_stage(stage_idx, lang, tokenizer, letter_ids, device, dtype, args,
              prev_adapter_path, out_dir):
    raw_examples = build_lang_examples(lang, load_rows(lang), hard_labels=args.hard_labels)
    dataset = SoftPromptCompletionDataset(raw_examples, tokenizer, args.max_seq_length)
    print(f"[stage {stage_idx}] lang={lang}: {len(dataset)} examples "
          f"({'fresh from base' if prev_adapter_path is None else f'continuing from {prev_adapter_path}'})")

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
    model.print_trainable_parameters()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed + stage_idx)

    losses, closses, metrics, steps, t0 = 0.0, 0.0, 0.0, 0, time.time()
    for it in range(1, args.iters_per_stage + 1):
        optimizer.zero_grad()
        lang_loss, metric, c_loss_value = lang_step(
            model, tokenizer, dataset, raw_examples, args.per_lang_batch_size,
            letter_ids, device, rng, args.max_seq_length, tokenizer.pad_token_id,
            args.consistency_lambda,
        )
        lang_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        losses += lang_loss.item()
        closses += c_loss_value
        metrics += metric
        steps += 1
        if it % args.steps_per_report == 0 or it == args.iters_per_stage:
            elapsed = time.time() - t0
            tqdm.write(f"[stage {stage_idx}/{lang}][{it}/{args.iters_per_stage}] "
                       f"loss={losses/steps:.4f} metric={metrics/steps:.4f} "
                       f"consistency={closses/steps:.4f} elapsed={elapsed:.0f}s")
            losses, closses, metrics, steps = 0.0, 0.0, 0.0, 0

    adapter_path = stage_dir(out_dir, stage_idx, lang) / "adapter"
    model.save_pretrained(str(adapter_path))
    final_metric = metrics / steps if steps else None
    print(f"[stage {stage_idx}] lang={lang} done, adapter saved to {adapter_path}")

    del model, base_model, optimizer
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return adapter_path, final_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM_DEFAULT)
    ap.add_argument("--consistency-lambda", type=float, default=0.5,
                     help="applied uniformly at every stage (not just the aug-si/aug-zh stages "
                          "the way the original recipe's row-duplication augmentation was) -- "
                          "this is the mechanism under test, see module docstring.")
    ap.add_argument("--hard-labels", action="store_true",
                     help="collapse id's vote distribution to one-hot (ablation control).")
    ap.add_argument("--per-lang-batch-size", type=int, default=8)
    ap.add_argument("--iters-per-stage", type=int, default=25,
                     help="fixed budget per stage, no per-stage CV/early-stopping -- reuses the "
                          "CV-validated iters from the lambda=0.5 simultaneous-training sweep.")
    ap.add_argument("--steps-per-report", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--out-dir", default="results/novel_curriculum")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    stages_summary = []
    prev_adapter_path = None
    for stage_idx, lang in enumerate(STAGES):
        adapter_path, final_metric = run_stage(
            stage_idx, lang, tokenizer, letter_ids, device, dtype, args,
            prev_adapter_path, out_dir,
        )
        stages_summary.append({
            "stage": stage_idx, "lang": lang, "iters": args.iters_per_stage,
            "consistency_lambda": args.consistency_lambda,
            "final_train_metric": final_metric, "adapter_path": str(adapter_path),
        })
        (out_dir / "summary.json").write_text(json.dumps({
            "stages": STAGES, "hard_labels": args.hard_labels,
            "consistency_lambda": args.consistency_lambda,
            "iters_per_stage": args.iters_per_stage,
            "completed_stages": stages_summary,
            "finished": stage_idx == len(STAGES) - 1,
        }, indent=2))
        prev_adapter_path = adapter_path

    print(f"Curriculum done. Final adapter: {prev_adapter_path}")


if __name__ == "__main__":
    main()
