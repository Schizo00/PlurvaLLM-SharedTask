"""LoRA fine-tune with simultaneous zh/id/si training and a macro-averaged loss.

Every training step pulls a fixed-size slice from each language (smaller
languages loop/repeat to fill their slice), and the loss averages the three
per-language batch-means with EQUAL weight -- mirroring the shared task's
macro-averaged accuracy metric, instead of the default mlx_lm behavior of
pooling all tokens in a batch into one flat (implicitly size-weighted) mean.

Only works for plain mlx_lm-loadable text models (e.g. MERaLiON, SeaLLMs).
Qwen3.5 loads via mlx_vlm, which has its own separate trainer
(mlx_vlm.trainer.train, loss_fn=vision_language_loss_fn) with a different
batch/loss shape -- this script does NOT cover that path.

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
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CompletionsDataset
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, iterate_batches, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

LANGS = ["zh", "id", "si"]


def read_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_datasets(data_dir: Path, prefix: str, tokenizer):
    datasets = {}
    for lang in LANGS:
        path = data_dir / f"{prefix}_{lang}.jsonl"
        rows = read_jsonl(path)
        datasets[lang] = CompletionsDataset(
            rows, tokenizer, prompt_key="prompt", completion_key="completion",
            mask_prompt=True,
        )
    return datasets


def stratified_iterate_batches(datasets_by_lang, per_lang_batch_size, max_seq_length, seed=None):
    """Every yielded batch = per_lang_batch_size examples from EACH language,
    concatenated in LANGS order. Smaller languages loop (upsample) via
    loop=True so every language contributes equally to every step."""
    iters = {
        lang: iterate_batches(ds, per_lang_batch_size, max_seq_length, loop=True, seed=seed)
        for lang, ds in datasets_by_lang.items()
    }
    while True:
        sub = [next(iters[lang]) for lang in LANGS]
        batch = mx.concatenate([b for b, _l in sub], axis=0)
        lengths = mx.concatenate([_l for _b, _l in sub], axis=0)
        yield batch, lengths


def macro_lang_loss_factory(per_lang_batch_size):
    """Per-example loss (not per-token pooled), sliced by the known fixed
    per-language segments, averaged across languages with EQUAL weight --
    directly mirrors macro-averaged accuracy scoring, so gradient signal
    isn't implicitly dominated by whichever language has more tokens."""

    def loss_fn(model, batch, lengths):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)

        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

        token_ce = nn.losses.cross_entropy(logits, targets) * mask
        tok_count = mx.maximum(mask.sum(axis=1), 1)
        per_example_loss = token_ce.sum(axis=1) / tok_count

        macro_total = mx.array(0.0)
        for i in range(len(LANGS)):
            seg = per_example_loss[i * per_lang_batch_size : (i + 1) * per_lang_batch_size]
            macro_total = macro_total + seg.mean()
        macro_loss = macro_total / len(LANGS)

        return macro_loss, mask.sum()

    return loss_fn


class PerLanguageValCallback(TrainingCallback):
    """Runs its OWN per-language evaluate() calls on a fixed schedule,
    instead of relying on train()'s single blended val_dataset -- so a
    stagnating language can't hide behind two improving ones."""

    def __init__(self, model, val_datasets, batch_size, val_batches, max_seq_length,
                 steps_per_eval):
        self.model = model
        self.val_datasets = val_datasets
        self.batch_size = batch_size
        self.val_batches = val_batches
        self.max_seq_length = max_seq_length
        self.steps_per_eval = steps_per_eval

    def on_train_loss_report(self, train_info: dict):
        it = train_info.get("iteration", 0)
        print(f"[iter {it}] train_loss(macro)={train_info.get('train_loss'):.4f}")
        if it % self.steps_per_eval != 0:
            return
        for lang, ds in self.val_datasets.items():
            val_loss = evaluate(
                self.model, ds, self.batch_size, self.val_batches,
                max_seq_length=self.max_seq_length,
            )
            print(f"  [iter {it}] val_loss[{lang}]={val_loss:.4f}")

    def on_val_loss_report(self, val_info: dict):
        # train()'s built-in blended validation is disabled (val_dataset=None
        # passed to train()); this is intentionally a no-op.
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="mlx_lm-loadable model (NOT Qwen3.5/mlx_vlm)")
    ap.add_argument("--train-dir", default="data/train", type=Path)
    ap.add_argument("--val-dir", default="data/val", type=Path)
    ap.add_argument("--adapter-path", default="adapters/macro_lora.safetensors")
    ap.add_argument("--per-lang-batch-size", type=int, default=4,
                     help="examples per language per step; total batch = this x 3")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 = all layers")
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    print(f"Loading base model from {args.model_path} ...")
    model, tokenizer = load(args.model_path)

    lora_config = {
        "rank": args.lora_rank,
        "scale": 20.0,
        "dropout": 0.0,
    }
    model.freeze()
    linear_to_lora_layers(model, args.lora_layers, lora_config)
    print_trainable_parameters(model)

    train_datasets = build_datasets(args.train_dir, "train", tokenizer)
    val_datasets = build_datasets(args.val_dir, "val", tokenizer)

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    training_args = TrainingArgs(
        batch_size=args.per_lang_batch_size * len(LANGS),
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        max_seq_length=args.max_seq_length,
        adapter_file=args.adapter_path,
    )

    callback = PerLanguageValCallback(
        model, val_datasets,
        batch_size=args.per_lang_batch_size * len(LANGS),
        val_batches=args.val_batches,
        max_seq_length=args.max_seq_length,
        steps_per_eval=args.steps_per_eval,
    )

    def custom_iterate_batches(dataset, batch_size, max_seq_length, **kwargs):
        # `dataset`/`batch_size` from train() are ignored -- batches are
        # built from train_datasets (per-language) + per_lang_batch_size,
        # which are captured from the enclosing scope instead.
        return stratified_iterate_batches(
            train_datasets, args.per_lang_batch_size, max_seq_length,
        )

    print("Starting training (simultaneous zh/id/si, macro-averaged loss)...")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=next(iter(train_datasets.values())),  # placeholder, unused by custom iterator
        val_dataset=None,  # per-language validation handled by the callback instead
        args=training_args,
        loss=macro_lang_loss_factory(args.per_lang_batch_size),
        iterate_batches=custom_iterate_batches,
        training_callback=callback,
    )


if __name__ == "__main__":
    main()
