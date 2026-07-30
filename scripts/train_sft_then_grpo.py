"""Combined SFT + GRPO training pipeline for PlurVA zh/id/si on Qwen3.5-4B.

Runs sequentially:
  1. LoRA SFT (simultaneous zh/id/si, macro-averaged loss) on data/train --
     either a fresh LoRA (default) or continuing from an existing adapter via
     --resume-sft-adapter, e.g. to pick up newly-added/expanded training data
     (like the Indonesian tie-duplication fix in prepare_training_data.py)
     without spending the full training budget again from scratch. Reuses the
     SFT mechanics from train_macro_lora_pt.py directly (imported, not
     duplicated).
  2. GRPO fine-tuning on top of whatever adapter phase 1 produced (or the
     --resume-sft-adapter checkpoint if phase 1 was skipped). Samples G
     completions per prompt from the raw dev-set rows, rewards them via
     resolve_gold_candidates/extract_letter (same logic eval_baseline.py and
     the SFT tie-fix use), and does a group-relative policy-gradient update.
     No KL-to-reference term (would need a second 4B-param model copy in
     memory) -- if the policy drifts or reward-hacks, that's the first thing
     to add back.

Skip SFT with --skip-sft to run GRPO directly on an existing checkpoint (in
which case --resume-sft-adapter is required, since GRPO needs a starting
point from somewhere).

Usage:
    # fresh SFT from scratch, then GRPO on top
    python scripts/train_sft_then_grpo.py

    # continue SFT from an existing checkpoint (e.g. after regenerating
    # data/train with new/expanded examples), then GRPO
    python scripts/train_sft_then_grpo.py \\
        --resume-sft-adapter adapters/macro_lora_pt/best

    # skip SFT entirely, run GRPO directly on an existing checkpoint
    python scripts/train_sft_then_grpo.py --skip-sft \\
        --resume-sft-adapter adapters/macro_lora_pt/best
"""
import argparse
import os
import random
import time
from pathlib import Path

# Must be set before torch touches the MPS backend: any op transformers/peft
# calls that MPS doesn't implement yet falls back to CPU instead of raising,
# which matters here since Qwen3.5's Gated DeltaNet hybrid-attention layers
# already need transformers' plain-torch fallback path (see
# train_macro_lora_pt.py's module docstring) rather than a fused kernel.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_baseline import build_prompt, extract_letter, load_rows, resolve_gold_candidates, resolve_options
from train_macro_lora_pt import (
    LANGS,
    PromptCompletionDataset,
    macro_lang_backward,
    per_language_validate,
    read_jsonl,
    sample_batch,
    save_best_known,
)


# ---------------------------------------------------------------------------
# SFT phase (mechanics imported from train_macro_lora_pt.py; this just wires
# them up to args and reports progress with an "[sft ...]" prefix).
# ---------------------------------------------------------------------------

def run_sft(model, tokenizer, device, args):
    train_datasets = {
        lang: PromptCompletionDataset(
            read_jsonl(args.train_dir / f"train_{lang}.jsonl"), tokenizer, args.max_seq_length
        )
        for lang in LANGS
    }
    val_datasets = {
        lang: PromptCompletionDataset(
            read_jsonl(args.val_dir / f"val_{lang}.jsonl"), tokenizer, args.max_seq_length
        )
        for lang in LANGS
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.sft_learning_rate)
    rng = random.Random(args.seed)

    adapter_path = Path(args.sft_adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    best_adapter_path = adapter_path / "best"
    best_val_loss = float("inf")
    stale_evals = 0

    losses, steps, t0 = 0.0, 0, time.time()
    lang_acc_sums = {lang: 0.0 for lang in LANGS}
    print("[sft] Starting (simultaneous zh/id/si, macro-averaged loss)...")
    pbar = tqdm(range(1, args.sft_iters + 1), desc="sft", unit="it")
    for it in pbar:
        step_t0 = time.time()
        batches_by_lang = {
            lang: sample_batch(train_datasets[lang], args.sft_per_lang_batch_size, rng)
            for lang in LANGS
        }
        optimizer.zero_grad()
        loss_value, lang_acc = macro_lang_backward(model, batches_by_lang, tokenizer.pad_token_id, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        step_dt = time.time() - step_t0

        losses += loss_value
        steps += 1
        for lang in LANGS:
            lang_acc_sums[lang] += lang_acc[lang]
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        acc_str = " ".join(f"acc[{lang}]={lang_acc[lang]:.3f}" for lang in LANGS)
        tqdm.write(f"[sft {it}/{args.sft_iters}] loss={loss_value:.4f} {acc_str} dt={step_dt:.1f}s")

        if it % args.sft_steps_per_report == 0 or it == args.sft_iters:
            elapsed = time.time() - t0
            acc_str = " ".join(f"acc[{lang}]={lang_acc_sums[lang] / steps:.3f}" for lang in LANGS)
            tqdm.write(f"[sft {it}] train_loss(macro)={losses / steps:.4f} {acc_str} elapsed={elapsed:.0f}s")
            losses, steps = 0.0, 0
            lang_acc_sums = {lang: 0.0 for lang in LANGS}

        if it % args.sft_steps_per_eval == 0 or it == args.sft_iters:
            val_loss = per_language_validate(
                model, val_datasets, args.sft_per_lang_batch_size, args.sft_val_batches,
                tokenizer.pad_token_id, device, rng,
            )
            tqdm.write(f"[sft {it}] val_loss(macro)={val_loss:.4f}")
            stop_early = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stale_evals = 0
                model.save_pretrained(str(best_adapter_path))
                tqdm.write(f"[sft {it}] New best macro val loss; saved best adapter to {best_adapter_path}")
            else:
                stale_evals += 1
                tqdm.write(f"[sft {it}] No improvement ({stale_evals}/{args.sft_patience})")
                if args.sft_patience > 0 and stale_evals >= args.sft_patience:
                    tqdm.write(f"[sft {it}] Early stopping: no macro val loss improvement for {args.sft_patience} evals.")
                    stop_early = True
            save_best_known(model, adapter_path, best_adapter_path)
            tqdm.write(f"[sft {it}] Saved best-known LoRA adapter to {adapter_path}")
            if stop_early:
                break

    save_best_known(model, adapter_path, best_adapter_path)
    print(f"[sft] Saved final (best-known) LoRA adapter to {adapter_path}")


# ---------------------------------------------------------------------------
# GRPO phase (ported from grpo_finetune.ipynb)
# ---------------------------------------------------------------------------

def load_split_rows(lang, seed, val_fraction):
    """Row-level train/val split. Uses a fresh Random(seed) per language, so
    (as noted in grpo_finetune.ipynb) it won't be byte-identical to
    prepare_training_data.py's split -- that script couples all three
    languages' shuffles through one shared RNG sequence. Internally
    consistent for GRPO's own train/val separation regardless."""
    rows = load_rows(lang)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_val = max(1, int(len(rows) * val_fraction))
    val_idx = set(order[:n_val])
    train_rows = [rows[i] for i in order if i not in val_idx]
    val_rows = [rows[i] for i in order if i in val_idx]
    return train_rows, val_rows


def build_chat_prompt(tokenizer, lang, row, options):
    prompt = build_prompt(lang, row, options)
    messages = [{"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
    except TypeError:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    return prompt, chat_prompt


def reward_fn(pred_letter, gold_candidates, unparseable_penalty):
    if pred_letter is None:
        return unparseable_penalty
    return 1.0 if pred_letter in gold_candidates else 0.0


@torch.no_grad()
def sample_group(model, tokenizer, device, lang, row, group_size, max_new_tokens, unparseable_penalty,
                  do_sample=True, temperature=None, top_p=None):
    """Returns a list of dicts: {input_ids (prompt+completion), prompt_len, reward}."""
    options = resolve_options(lang, row)
    valid_letters = {l for l in "ABCD" if options[l]}
    gold_candidates = set(resolve_gold_candidates(lang, row["Gold_Answer"]))
    _, chat_prompt = build_chat_prompt(tokenizer, lang, row, options)

    inputs = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=group_size,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    output_ids = model.generate(**inputs, **gen_kwargs)

    samples = []
    for seq in output_ids:
        completion_ids = seq[prompt_len:].tolist()
        while completion_ids and completion_ids[-1] == tokenizer.pad_token_id:
            completion_ids.pop()
        response = tokenizer.decode(completion_ids, skip_special_tokens=True)
        pred = extract_letter(response, valid_letters, options)
        reward = reward_fn(pred, gold_candidates, unparseable_penalty)
        samples.append({
            "input_ids": inputs["input_ids"][0].tolist() + completion_ids,
            "prompt_len": prompt_len,
            "reward": reward,
        })
    return samples


def collate_weighted(samples, pad_token_id, device):
    max_len = max(len(s["input_ids"]) for s in samples)
    input_ids = torch.full((len(samples), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(samples), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
    weights = torch.zeros(len(samples), dtype=torch.float)
    for i, s in enumerate(samples):
        ids = s["input_ids"]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, s["prompt_len"]:L] = torch.tensor(ids[s["prompt_len"]:], dtype=torch.long)
        weights[i] = s["advantage"]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "weights": weights.to(device),
    }


def grpo_step(model, tokenizer, device, batches_by_lang, group_size, temperature, top_p,
              max_new_tokens, unparseable_penalty):
    """batches_by_lang: {lang: [row, ...]}. Group-normalizes rewards to
    advantages per row, skips degenerate (zero-variance) groups, then does
    one combined forward+backward over all surviving samples across every
    language in the step."""
    all_samples = []
    reward_log = {}
    # Rollout must happen with the model OUT of training mode: generate() on a
    # model that's .train() + gradient-checkpointing-enabled silently disables
    # the KV cache (transformers forces use_cache=False in that combination),
    # turning every sampled token into an O(n^2)-attention full reforward over
    # these long MCQ prompts. grpo_validate already does this; grpo_step
    # (used for the actual policy-gradient rollouts, not just eval) didn't.
    model.eval()
    for lang, rows in batches_by_lang.items():
        lang_rewards = []
        for row in rows:
            group = sample_group(
                model, tokenizer, device, lang, row, group_size, max_new_tokens, unparseable_penalty,
                do_sample=True, temperature=temperature, top_p=top_p,
            )
            rewards = torch.tensor([s["reward"] for s in group])
            lang_rewards.extend(rewards.tolist())
            std, mean = rewards.std(unbiased=False), rewards.mean()
            if std < 1e-6:
                continue  # degenerate group: every sample scored the same, no signal
            advantages = (rewards - mean) / (std + 1e-6)
            for s, adv in zip(group, advantages.tolist()):
                s["advantage"] = adv
                all_samples.append(s)
        reward_log[lang] = sum(lang_rewards) / len(lang_rewards) if lang_rewards else float("nan")
    model.train()

    if not all_samples:
        return None, reward_log  # every group this step was degenerate

    batch = collate_weighted(all_samples, tokenizer.pad_token_id, device)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]

    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(labels.shape)
    valid = (labels != -100)
    tok_count = valid.sum(dim=1).clamp(min=1)
    per_example_loss = (token_ce * valid).sum(dim=1) / tok_count  # mean CE per example = -mean logprob
    weighted_loss = (per_example_loss * batch["weights"]).mean()

    weighted_loss.backward()
    return weighted_loss.item(), reward_log


@torch.no_grad()
def grpo_validate(model, tokenizer, device, val_rows_by_lang, rng, n_rows_per_lang, max_new_tokens,
                   unparseable_penalty):
    model.eval()
    lang_acc = {}
    for lang in LANGS:
        rows = rng.sample(val_rows_by_lang[lang], min(n_rows_per_lang, len(val_rows_by_lang[lang])))
        correct = 0
        for row in rows:
            sample = sample_group(
                model, tokenizer, device, lang, row, group_size=1, max_new_tokens=max_new_tokens,
                unparseable_penalty=unparseable_penalty, do_sample=False,
            )[0]
            correct += 1 if sample["reward"] == 1.0 else 0
        lang_acc[lang] = correct / len(rows)
    model.train()
    return lang_acc, sum(lang_acc.values()) / len(lang_acc)


def run_grpo(model, tokenizer, device, args):
    train_rows_by_lang = {}
    val_rows_by_lang = {}
    for lang in LANGS:
        train_rows_by_lang[lang], val_rows_by_lang[lang] = load_split_rows(
            lang, args.seed, args.grpo_val_fraction
        )
        print(f"[grpo] {lang}: {len(train_rows_by_lang[lang])} train rows, {len(val_rows_by_lang[lang])} val rows")

    out_adapter_path = Path(args.grpo_adapter_path)
    out_adapter_path.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.grpo_learning_rate)
    rng = random.Random(args.seed)

    best_val_macro = -1.0
    stale_evals = 0
    t0 = time.time()
    print("[grpo] Starting...")
    pbar = tqdm(range(1, args.grpo_iters + 1), desc="grpo", unit="it")
    for it in pbar:
        stop_early = False
        batches_by_lang = {
            lang: [rng.choice(train_rows_by_lang[lang]) for _ in range(args.grpo_rows_per_lang_per_step)]
            for lang in LANGS
        }

        optimizer.zero_grad()
        loss_value, reward_log = grpo_step(
            model, tokenizer, device, batches_by_lang, args.grpo_group_size, args.grpo_temperature,
            args.grpo_top_p, args.grpo_max_new_tokens, args.grpo_unparseable_penalty,
        )
        if loss_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        reward_str = " ".join(f"r[{lang}]={reward_log[lang]:.2f}" for lang in LANGS)
        pbar.set_postfix(loss=f"{loss_value:.4f}" if loss_value is not None else "skip")
        loss_display = loss_value if loss_value is not None else float("nan")
        tqdm.write(f"[grpo {it}/{args.grpo_iters}] loss={loss_display:.4f} {reward_str}")

        if it % args.grpo_steps_per_eval == 0 or it == args.grpo_iters:
            lang_acc, val_macro = grpo_validate(
                model, tokenizer, device, val_rows_by_lang, rng, args.grpo_val_rows_per_lang,
                args.grpo_max_new_tokens, args.grpo_unparseable_penalty,
            )
            acc_str = " ".join(f"val_acc[{lang}]={lang_acc[lang]:.3f}" for lang in LANGS)
            tqdm.write(f"[grpo {it}] val_macro_acc={val_macro:.4f} {acc_str}")
            if val_macro > best_val_macro:
                best_val_macro = val_macro
                stale_evals = 0
                model.save_pretrained(str(out_adapter_path / "best"))
                tqdm.write(f"[grpo {it}] New best val macro acc; saved adapter to {out_adapter_path / 'best'}")
            else:
                stale_evals += 1
                tqdm.write(f"[grpo {it}] No improvement ({stale_evals}/{args.grpo_patience})")
                if args.grpo_patience > 0 and stale_evals >= args.grpo_patience:
                    tqdm.write(f"[grpo {it}] Early stopping: no val macro-acc improvement for {args.grpo_patience} evals.")
                    stop_early = True

        if it % args.grpo_steps_per_save == 0 or it == args.grpo_iters:
            model.save_pretrained(str(out_adapter_path))
            tqdm.write(f"[grpo {it}] Saved latest adapter to {out_adapter_path}")

        if stop_early:
            break

    elapsed = time.time() - t0
    print(f"[grpo] Done in {elapsed:.0f}s. Best val macro acc: {best_val_macro:.4f}. "
          f"Best adapter: {out_adapter_path / 'best'}, latest: {out_adapter_path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--resume-sft-adapter", default=None,
                     help="PEFT LoRA adapter dir to start from (e.g. adapters/macro_lora_pt/best). "
                          "If set, SFT continues training this adapter instead of initializing a "
                          "fresh LoRA. Required if --skip-sft is set.")
    ap.add_argument("--skip-sft", action="store_true",
                     help="skip the SFT phase entirely and run GRPO directly on --resume-sft-adapter")
    ap.add_argument("--skip-grpo", action="store_true",
                     help="stop after the SFT phase; don't run GRPO")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                     help="gradient-norm clip applied before every optimizer step (both SFT and "
                          "GRPO phases). Training here runs in fp16 on MPS with no loss-scaling, "
                          "so an unclipped gradient spike (e.g. from a batch the model does badly "
                          "on) can overflow and permanently NaN the weights in a single step.")

    # SFT args
    ap.add_argument("--train-dir", default="data/train", type=Path)
    ap.add_argument("--val-dir", default="data/val", type=Path)
    ap.add_argument("--sft-adapter-path", default=None,
                     help="where SFT saves its adapter. Defaults to adapters/macro_lora_pt for a "
                          "fresh run, or adapters/macro_lora_pt_continued when --resume-sft-adapter "
                          "is set, so a continued run never overwrites the checkpoint it resumed from.")
    ap.add_argument("--sft-per-lang-batch-size", type=int, default=2)
    ap.add_argument("--sft-iters", type=int, default=500)
    ap.add_argument("--sft-steps-per-report", type=int, default=10)
    ap.add_argument("--sft-steps-per-eval", type=int, default=5)
    ap.add_argument("--sft-val-batches", type=int, default=10)
    ap.add_argument("--sft-patience", type=int, default=5)
    ap.add_argument("--sft-learning-rate", type=float, default=1e-4)

    # GRPO args
    ap.add_argument("--grpo-adapter-path", default="adapters/macro_lora_grpo")
    ap.add_argument("--grpo-val-fraction", type=float, default=0.15)
    ap.add_argument("--grpo-rows-per-lang-per-step", type=int, default=1)
    ap.add_argument("--grpo-group-size", type=int, default=8)
    ap.add_argument("--grpo-temperature", type=float, default=0.8)
    ap.add_argument("--grpo-top-p", type=float, default=0.95)
    ap.add_argument("--grpo-max-new-tokens", type=int, default=40)
    ap.add_argument("--grpo-unparseable-penalty", type=float, default=-0.2)
    ap.add_argument("--grpo-iters", type=int, default=200)
    ap.add_argument("--grpo-learning-rate", type=float, default=5e-6)
    ap.add_argument("--grpo-steps-per-eval", type=int, default=10)
    ap.add_argument("--grpo-val-rows-per-lang", type=int, default=10)
    ap.add_argument("--grpo-steps-per-save", type=int, default=20)
    ap.add_argument("--grpo-patience", type=int, default=5,
                     help="stop after this many consecutive evals with no val macro-acc "
                          "improvement (0 disables early stopping)")

    args = ap.parse_args()

    if args.skip_sft and not args.resume_sft_adapter:
        ap.error("--skip-sft requires --resume-sft-adapter (GRPO needs a starting checkpoint)")
    if args.skip_sft and args.skip_grpo:
        ap.error("--skip-sft and --skip-grpo together leave nothing to run")

    if args.sft_adapter_path is None:
        args.sft_adapter_path = (
            "adapters/macro_lora_pt_continued" if args.resume_sft_adapter else "adapters/macro_lora_pt"
        )
    if args.resume_sft_adapter and not args.skip_sft:
        resume_resolved = Path(args.resume_sft_adapter).resolve()
        sft_best_resolved = (Path(args.sft_adapter_path) / "best").resolve()
        sft_root_resolved = Path(args.sft_adapter_path).resolve()
        if resume_resolved in (sft_best_resolved, sft_root_resolved):
            ap.error(
                f"--sft-adapter-path ({args.sft_adapter_path}) would overwrite the checkpoint "
                f"being resumed from (--resume-sft-adapter {args.resume_sft_adapter}). "
                "Pick a different --sft-adapter-path."
            )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        # fp16's narrow dynamic range is numerically unstable through Qwen3.5's
        # Gated DeltaNet torch-fallback attention path (see module docstring):
        # continuing training from an already-fine-tuned adapter reliably NaNs
        # the loss after a single optimizer step in fp16 (confirmed with and
        # without gradient clipping -- clipping made no difference, since the
        # blowup happens in the fp16 forward activations, not the raw gradient
        # norm). bf16 has the same exponent range as fp32, just less mantissa
        # precision, and does not exhibit this failure.
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
    print(f"Loading {args.model_id} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)

    if args.resume_sft_adapter:
        print(f"Loading trainable LoRA adapter from {args.resume_sft_adapter} ...")
        model = PeftModel.from_pretrained(base_model, args.resume_sft_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for checkpointing to work through frozen (non-LoRA) layers
        print("Gradient checkpointing enabled (trades compute for memory).")
    model.to(device)
    model.train()

    if args.skip_sft:
        print("--skip-sft set: skipping SFT phase, running GRPO directly on the loaded adapter.")
    else:
        run_sft(model, tokenizer, device, args)

    if args.skip_grpo:
        print("--skip-grpo set: stopping after the SFT phase.")
    else:
        run_grpo(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
