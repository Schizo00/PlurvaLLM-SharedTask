"""LoRA fine-tune Qwen3.5 (mlx_vlm) with simultaneous zh/id/si training and a
macro-averaged loss.

Qwen3.5 loads via mlx_vlm (it's natively multimodal), which has its own
trainer (mlx_vlm.trainer.sft_trainer) -- a separate API from mlx_lm's, with a
dict-based batch format ({"input_ids", "attention_mask", "completion_mask",
"pixel_values"}) rather than mlx_lm's plain (tokens, lengths) tuple. Critically,
mlx_vlm's train() does NOT expose an iterate_batches injection point the way
mlx_lm's does, so this script implements its own training loop (mirroring
sft_trainer.train()'s structure) rather than calling that function directly.

Every training step pulls a fixed-size slice from each language (smaller
languages loop/repeat to fill their slice), and the loss averages the three
per-language batch-means with EQUAL weight -- mirroring the shared task's
macro-averaged accuracy metric.

Expected data layout (one chat-format jsonl per language):
    <train_dir>/train_zh.jsonl
        {"messages": [{"role": "user", "content": "<built prompt>"},
                       {"role": "assistant", "content": "A"}]}
    <train_dir>/train_id.jsonl
    <train_dir>/train_si.jsonl
    <val_dir>/val_zh.jsonl   (same schema, held out)
    <val_dir>/val_id.jsonl
    <val_dir>/val_si.jsonl

No "images"/"image" key needed for text-only rows -- VisionDataset defaults
to an empty image list and pixel_values=None flows through the whole
pipeline untouched.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_vlm import load
from mlx_vlm.trainer import VisionDataset, find_all_linear_names, get_peft_model, save_adapter
from mlx_vlm.trainer.sft_trainer import TrainingArgs, evaluate, iterate_batches

LANGS = ["zh", "id", "si"]


def read_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_datasets(data_dir: Path, prefix: str, config, processor):
    datasets = {}
    for lang in LANGS:
        path = data_dir / f"{prefix}_{lang}.jsonl"
        rows = read_jsonl(path)
        datasets[lang] = VisionDataset(
            rows, config, processor, train_on_completions=True,
        )
    return datasets


def _pad_to_len(arr, target_len, pad_value=0):
    cur_len = arr.shape[1]
    if cur_len == target_len:
        return arr
    pad_width = [(0, 0), (0, target_len - cur_len)]
    return mx.pad(arr, pad_width, constant_values=pad_value)


def stratified_iterate_batches(datasets_by_lang, per_lang_batch_size, max_seq_length):
    """Every yielded batch = per_lang_batch_size examples from EACH language,
    concatenated in LANGS order. mlx_vlm's own iterate_batches independently
    pads each language's chunk to its own max length, so chunks must be
    re-padded to a common length before they can be concatenated."""
    iters = {
        lang: iterate_batches(ds, per_lang_batch_size, max_seq_length, train=True)
        for lang, ds in datasets_by_lang.items()
    }
    while True:
        sub = [next(iters[lang]) for lang in LANGS]
        max_len = max(b["input_ids"].shape[1] for b in sub)

        merged = {
            "input_ids": mx.concatenate(
                [_pad_to_len(b["input_ids"], max_len) for b in sub], axis=0
            ),
            "attention_mask": mx.concatenate(
                [_pad_to_len(b["attention_mask"], max_len) for b in sub], axis=0
            ),
            "pixel_values": None,  # text-only training
        }
        if any("completion_mask" in b for b in sub):
            merged["completion_mask"] = mx.concatenate(
                [
                    _pad_to_len(
                        b.get("completion_mask", mx.zeros_like(b["input_ids"])), max_len
                    )
                    for b in sub
                ],
                axis=0,
            )
        yield merged


def macro_lang_loss_factory(per_lang_batch_size):
    """Per-example loss (not the library default's whole-batch pooled mean),
    sliced by the known fixed per-language segments, averaged across
    languages with EQUAL weight -- mirrors macro-averaged accuracy scoring."""

    def loss_fn(model, batch):
        input_ids = batch["input_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        labels = batch["input_ids"][:, 1:]

        outputs = model(input_ids, batch["pixel_values"], attention_mask)
        logits = outputs.logits.astype(mx.float32)

        seq_len = input_ids.shape[1]
        if logits.shape[1] != seq_len:
            if logits.shape[1] < seq_len:
                pad = ((0, 0), (0, seq_len - logits.shape[1]), (0, 0))
                logits = mx.pad(logits, pad, constant_values=-100)
            else:
                logits = logits[:, -seq_len:, :]

        lengths = mx.minimum(attention_mask.sum(axis=1), seq_len)
        length_mask = mx.arange(seq_len)[None, :] < lengths[:, None]
        mask = length_mask
        if "completion_mask" in batch:
            mask = mask * batch["completion_mask"][:, 1:]

        token_ce = nn.losses.cross_entropy(logits, labels) * mask
        tok_count = mx.maximum(mask.sum(axis=1), 1)
        per_example_loss = token_ce.sum(axis=1) / tok_count

        macro_total = mx.array(0.0)
        for i in range(len(LANGS)):
            seg = per_example_loss[i * per_lang_batch_size : (i + 1) * per_lang_batch_size]
            macro_total = macro_total + seg.mean()
        macro_loss = macro_total / len(LANGS)

        return macro_loss, mask.sum()

    return loss_fn


def per_language_validate(model, val_datasets, batch_size, val_batches, max_seq_length):
    from mlx_vlm.trainer.sft_trainer import vision_language_loss_fn
    from functools import partial

    loss_fn = partial(vision_language_loss_fn, train_on_completions=True)
    for lang, ds in val_datasets.items():
        val_loss = evaluate(
            model, ds, batch_size, val_batches, max_seq_length=max_seq_length,
            loss_fn=loss_fn, train_on_completions=True,
        )
        print(f"  val_loss[{lang}]={val_loss:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="mlx_vlm-loadable model, e.g. mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--train-dir", default="data/train", type=Path)
    ap.add_argument("--val-dir", default="data/val", type=Path)
    ap.add_argument("--adapter-path", default="adapters/macro_lora_qwen.safetensors")
    ap.add_argument("--per-lang-batch-size", type=int, default=4,
                     help="examples per language per step; total batch = this x 3")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    args = ap.parse_args()

    print(f"Loading base model from {args.model_path} (mlx_vlm) ...")
    model, processor = load(args.model_path)
    config = model.config if isinstance(model.config, dict) else model.config.__dict__

    linear_layers = find_all_linear_names(model)
    model = get_peft_model(model, linear_layers, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0)

    train_datasets = build_datasets(args.train_dir, "train", config, processor)
    val_datasets = build_datasets(args.val_dir, "val", config, processor)

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_fn = macro_lang_loss_factory(args.per_lang_batch_size)
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    adapter_path = Path(args.adapter_path)
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    losses, n_tokens, steps, train_time = 0.0, 0, 0, 0.0
    batch_iter = stratified_iterate_batches(
        train_datasets, args.per_lang_batch_size, args.max_seq_length
    )

    print("Starting training (simultaneous zh/id/si, macro-averaged loss)...")
    for it, batch in zip(range(1, args.iters + 1), batch_iter):
        tic = time.perf_counter()
        (lvalue, toks), grad = loss_value_and_grad(model, batch)
        optimizer.update(model, grad)
        mx.eval(model.state, optimizer.state, lvalue)
        losses += lvalue.item()
        n_tokens += toks.item()
        steps += 1
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = losses / steps
            print(
                f"[iter {it}] train_loss(macro)={train_loss:.4f} "
                f"tokens/sec={n_tokens/train_time:.1f}"
            )
            losses, n_tokens, steps, train_time = 0.0, 0, 0, 0.0

        if it % args.steps_per_eval == 0 or it == args.iters:
            per_language_validate(
                model, val_datasets, args.per_lang_batch_size * len(LANGS),
                args.val_batches, args.max_seq_length,
            )

        if it % 100 == 0 or it == args.iters:
            save_adapter(model, adapter_path)
            print(f"[iter {it}] Saved adapter weights to {adapter_path}.")

    save_adapter(model, adapter_path)
    print(f"Saved final adapter weights to {adapter_path}.")


if __name__ == "__main__":
    main()
