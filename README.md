# Fine-tuning guide

This repo LoRA-fine-tunes candidate base models on the PlurVA multiple-choice
dev sets (`zh`/`id`/`si`), using a **macro-averaged loss** across the three
languages so training tracks the shared task's macro-averaged accuracy metric
instead of letting the largest language (`zh`) dominate the gradient.

Fine-tuning is done with [scripts/train_macro_lora_pt.py](scripts/train_macro_lora_pt.py)
(PyTorch / `transformers` / `peft`). Device and dtype are auto-detected —
CUDA if available (bf16 on Ampere+, fp16 fallback on older GPUs like V100),
else MPS, else CPU.

## 1. Install dependencies

```bash
pip install torch transformers peft tqdm
```
For CUDA, install the `torch` build matching your CUDA version from
https://pytorch.org/get-started/locally/ first.

## 2. Prepare training data

```bash
python scripts/prepare_training_data.py \
    --train-dir data/train \
    --val-dir data/val \
    --val-fraction 0.15 \
    --seed 42
```

This reuses the exact prompt-building and gold-answer-resolution logic from
[scripts/eval_baseline.py](scripts/eval_baseline.py) (scenario-line inclusion,
Indonesian majority-vote resolution, Sinhala `Both`/`0` → `C`/`D` mapping) so
training prompts are identical in format to what zero-shot eval already
validated. It reads `data/{chinese,indonesian,sri_lankan}_dev.jsonl` and
writes, per language:

```
data/train/train_{zh,id,si}.jsonl   {"prompt": <raw prompt>, "completion": " <letter>"}
data/val/val_{zh,id,si}.jsonl
```

Note: this burns the dev sets as a held-out discriminator for the training
pipeline — it is **not** the final production data source.

## 3. Fine-tune

```bash
python scripts/train_macro_lora_pt.py \
    --model-id Qwen/Qwen3.5-4B \
    --train-dir data/train --val-dir data/val \
    --adapter-path adapters/macro_lora_pt \
    --per-lang-batch-size 2 --iters 500 \
    --learning-rate 1e-4 --lora-rank 16 --lora-alpha 32
```

Gradient checkpointing is on by default (`--no-grad-checkpoint` to disable it
if you have memory to spare and want more speed).

The trained LoRA adapter (PEFT format) is checkpointed every 100 iterations
and at the final iteration into `--adapter-path` (a directory, e.g.
`adapters/macro_lora_pt/`).

### Training args

| Flag | Meaning |
|---|---|
| `--model-id` | HF model id or local path (default `Qwen/Qwen3.5-4B`) |
| `--per-lang-batch-size` | examples per language per step; real batch = this × 3 languages |
| `--iters` | training steps |
| `--steps-per-report` | how often to print the running macro train loss |
| `--steps-per-eval` | how often to run per-language validation loss |
| `--val-batches` | number of sampled batches per language per validation pass |
| `--max-seq-length` | truncation length (right-truncates the prompt, keeps the completion) |
| `--grad-checkpoint` / `--no-grad-checkpoint` | trade compute for memory (default: on) |
| `--learning-rate` | AdamW learning rate |
| `--lora-rank`, `--lora-alpha` | LoRA config |
| `--seed` | RNG seed for batch sampling |

Every step pulls a fixed-size slice from **each** language (smaller languages
loop/repeat to fill their slice). The loss is computed as three separate
backward passes — one per language, each contributing 1/3 of the gradient —
rather than one combined loss tensor, so each language's forward computation
graph is freed immediately instead of all three staying alive at once. The
per-language losses are averaged with equal weight, not pooled by token count,
so `zh` (the largest language) can't dominate the gradient simply by having
more tokens.

## 4. Evaluate

`scripts/eval_baseline.py` runs zero-shot MCQ eval, but only loads plain
`mlx_lm`/`mlx_vlm` model directories — it does not currently support loading
a PEFT/PyTorch adapter. To evaluate a `train_macro_lora_pt.py` adapter, merge
it onto the base model first:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "adapters/macro_lora_pt")
model = model.merge_and_unload()
model.save_pretrained("models/qwen3.5-4b-macro-lora-merged")
```

then evaluate the merged model with your own `transformers`-based scoring
loop, or convert it to MLX to reuse `eval_baseline.py`.
