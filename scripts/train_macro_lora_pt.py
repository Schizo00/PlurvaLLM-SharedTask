"""LoRA fine-tune Qwen3.5-4B (PyTorch/transformers/peft) with simultaneous
zh/id/si training and a macro-averaged loss.

Runs on CUDA, MPS, or CPU (auto-detected, CUDA preferred). mlx_vlm's Qwen3.5
implementation was confirmed unable to train
(the Gated DeltaNet hybrid-attention layers use a custom Metal kernel with no
registered backward/vjp rule); transformers falls back to a plain PyTorch
implementation of the same layers when flash-linear-attention/causal-conv1d
aren't installed, and that fallback IS autograd-compatible (verified with a
real forward+backward pass before writing this script).

Every training step pulls a fixed-size slice from each language (smaller
languages loop/repeat to fill their slice), and the loss averages the three
per-language batch-means with EQUAL weight -- mirroring the shared task's
macro-averaged accuracy metric, rather than a flat token-pooled mean that
would implicitly let whichever language has more/longer completions dominate
the gradient (this is the same design as train_macro_lora.py / _qwen.py, just
ported to torch's autograd instead of mlx's).

Expected data layout (one prompt/completion jsonl per language):
    <train_dir>/train_zh.jsonl   {"prompt": ..., "completion": "A"}
    <train_dir>/train_id.jsonl
    <train_dir>/train_si.jsonl
    <val_dir>/val_zh.jsonl       (same schema, held out)
    <val_dir>/val_id.jsonl
    <val_dir>/val_si.jsonl
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGS = ["zh", "id", "si"]


def read_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PromptCompletionDataset:
    """Tokenizes prompt+completion once upfront; masked-prompt loss is
    implemented via a labels tensor with -100 over the prompt span (the
    standard HF convention for "ignore this position's loss").

    row["prompt"] is the RAW (un-templated) prompt text -- the chat template
    is applied here, with enable_thinking=False, so training-time formatting
    matches exactly what the zero-shot eval used (mlx_vlm's
    apply_chat_template earlier in this project)."""

    def __init__(self, rows, tokenizer, max_seq_length):
        self.examples = []
        for row in rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            try:
                chat_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                chat_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
            prompt_ids = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + completion_ids
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[-max_seq_length:]
                prompt_len = max(0, len(input_ids) - len(completion_ids))
            else:
                prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def sample_batch(dataset, batch_size, rng):
    """Sample WITH REPLACEMENT so smaller-language datasets can still fill a
    full batch_size slice every step (the "loop"/upsample behavior)."""
    idxs = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    return [dataset[i] for i in idxs]


def collate(examples, pad_token_id, device):
    max_len = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        L = len(e["input_ids"])
        input_ids[i, :L] = torch.tensor(e["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(e["labels"], dtype=torch.long)
        attention_mask[i, :L] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def _language_batch_loss(model, examples, pad_token_id, device):
    batch = collate(examples, pad_token_id, device)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]

    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(labels.shape)
    valid = (labels != -100)
    tok_count = valid.sum(dim=1).clamp(min=1)
    per_example_loss = (token_ce * valid).sum(dim=1) / tok_count
    return per_example_loss.mean()


def macro_lang_backward(model, batches_by_lang, pad_token_id, device):
    """EQUAL-weight macro average across languages -- the actual point of
    this whole script -- but implemented as THREE separate backward() calls
    (one per language, each contributing 1/3 of the gradient), not one
    combined loss tensor. A single combined backward would keep all three
    languages' full forward computation graphs alive simultaneously; calling
    backward per-language frees each graph immediately, which matters a lot
    here since Qwen3.5's linear-attention fallback (transformers' plain-torch
    path, used when flash-linear-attention/causal-conv1d aren't installed) is
    already unusually memory-hungry per forward pass. Returns the summed
    macro loss value (float) for reporting; gradients are left in .grad for
    the caller to optimizer.step() once, after all languages are done."""
    total = 0.0
    for lang in LANGS:
        lang_loss = _language_batch_loss(model, batches_by_lang[lang], pad_token_id, device)
        (lang_loss / len(LANGS)).backward()
        total += lang_loss.item()
    return total / len(LANGS)


@torch.no_grad()
def per_language_validate(model, val_datasets, batch_size, val_batches, pad_token_id, device, rng):
    model.eval()
    for lang, ds in val_datasets.items():
        losses = []
        for _ in range(val_batches):
            batch = collate(sample_batch(ds, batch_size, rng), pad_token_id, device)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1, :]
            labels = batch["labels"][:, 1:]
            token_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=-100, reduction="none",
            ).view(labels.shape)
            valid = (labels != -100)
            tok_count = valid.sum(dim=1).clamp(min=1)
            per_example_loss = (token_ce * valid).sum(dim=1) / tok_count
            losses.append(per_example_loss.mean().item())
        tqdm.write(f"  val_loss[{lang}]={sum(losses)/len(losses):.4f}")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train-dir", default="data/train", type=Path)
    ap.add_argument("--val-dir", default="data/val", type=Path)
    ap.add_argument("--adapter-path", default="adapters/macro_lora_pt")
    ap.add_argument("--per-lang-batch-size", type=int, default=2,
                     help="examples per language per step; total batch = this x 3")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # bf16 needs Ampere+ (sm_80) on CUDA; older GPUs (e.g. V100) silently
    # produce NaNs/garbage under bf16 matmuls, so fall back to fp16 there.
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    print(f"Loading {args.model_id} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for checkpointing to work through frozen (non-LoRA) layers
        print("Gradient checkpointing enabled (trades compute for memory).")
    model.train()

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    losses, steps, t0 = 0.0, 0, time.time()
    print("Starting training (simultaneous zh/id/si, macro-averaged loss)...")
    pbar = tqdm(range(1, args.iters + 1), desc="train", unit="it")
    for it in pbar:
        batches_by_lang = {
            lang: sample_batch(train_datasets[lang], args.per_lang_batch_size, rng)
            for lang in LANGS
        }
        optimizer.zero_grad()
        loss_value = macro_lang_backward(model, batches_by_lang, tokenizer.pad_token_id, device)
        optimizer.step()

        losses += loss_value
        steps += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        if it % args.steps_per_report == 0 or it == args.iters:
            elapsed = time.time() - t0
            tqdm.write(f"[iter {it}] train_loss(macro)={losses/steps:.4f} elapsed={elapsed:.0f}s")
            losses, steps = 0.0, 0

        if it % args.steps_per_eval == 0 or it == args.iters:
            per_language_validate(
                model, val_datasets, args.per_lang_batch_size, args.val_batches,
                tokenizer.pad_token_id, device, rng,
            )

        if it % 100 == 0 or it == args.iters:
            model.save_pretrained(str(adapter_path))
            tqdm.write(f"[iter {it}] Saved LoRA adapter to {adapter_path}")

    model.save_pretrained(str(adapter_path))
    print(f"Saved final LoRA adapter to {adapter_path}")


if __name__ == "__main__":
    main()
