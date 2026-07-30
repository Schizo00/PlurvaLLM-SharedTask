"""From-scratch LoRA fine-tuning for PlurVA (zh/id/si) combining two
mechanisms, both grounded in this dataset's actual structure rather than
generic architecture tweaks:

  A) Soft-label KL distillation: Indonesian's Gold_Answer is always a
     5-annotator vote list (e.g. "A, A, A, A, C"), for all 366/366 rows, not
     just the 72 with no strict majority. Training targets the full
     empirical vote distribution instead of a collapsed one-hot majority
     label -- see novel_data.build_target_dist. zh/si have no natural vote
     distribution in this dataset, so they degenerate to one-hot (a strict
     generalization of hard-label CE, verified in verify_novel.py's
     loss-equivalence check).

  B) Positional-invariance consistency regularization: extends the team's
     findings doc's proposed-but-unimplemented "permute the correct answer's
     position" augmentation idea into an actual training-time Jensen-Shannon
     consistency penalty between a model's predictions on an example and a
     position-permuted variant of the same content -- targets the documented
     dev/test distribution mismatch (position/format memorization vs. real
     understanding).

Also implements k-fold CV (for hyperparameter/early-stopping selection and a
statistically honest performance estimate on a very small dataset) plus a
final 100%-data refit using the CV-selected config, and a continuous
per-example metric (expected probability mass on the correct answer) that
replaces the old 0/0.5/1 two-example argmax-accuracy metric.

Runs on CUDA, MPS, or CPU (auto-detected). Always uses bf16 except on
non-bf16 CUDA GPUs -- fp16 on MPS was confirmed this session to NaN Qwen3.5's
Gated DeltaNet attention fallback path after a single optimizer step.

See NLP Shared Task Findings.md and the approved plan at
~/.claude/plans/can-we-draft-something-frolicking-starlight.md for the full
rationale.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

# Must be set before torch touches the MPS backend: any op transformers/peft
# calls that MPS doesn't implement falls back to CPU instead of raising (see
# train_sft_then_grpo.py, which needed this for its GRPO generate() path --
# not observed to be load-bearing for this script's plain forward/backward
# loop, but harmless and cheap insurance against a wider range of ops, e.g.
# the JS-divergence gather in consistency_loss).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import build_prompt, load_rows, resolve_options
from novel_data import LETTERS, assign_folds, build_target_dist, permute_example, sample_permutation
from train_macro_lora_pt import LANGS, save_best_known

MAX_GRAD_NORM_DEFAULT = 1.0


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def letter_token_ids(tokenizer) -> dict:
    ids = {}
    for letter in LETTERS:
        toks = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            raise RuntimeError(
                f"completion ' {letter}' tokenizes to {len(toks)} tokens ({toks}); "
                "the answer-position-finding logic in this script assumes exactly 1 "
                "(true for Qwen/Qwen3.5-4B, verified in verify_novel.py -- re-verify "
                "before swapping base models)."
            )
        ids[letter] = toks[0]
    return ids


def tokenize_example(prompt: str, completion: str, tokenizer, max_seq_length: int):
    messages = [{"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
    except TypeError:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    prompt_ids = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        prompt_len = max(0, len(input_ids) - len(completion_ids))
    else:
        prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "labels": labels}


class SoftPromptCompletionDataset:
    """Eagerly tokenizes prompt+completion pairs (like
    train_macro_lora_pt.PromptCompletionDataset), plus carries each example's
    target probability distribution over {A,B,C,D} (one-hot for zh/si, the
    empirical vote distribution for id under Mechanism A). `completion` is
    still built as " {argmax letter}" purely to give the -100 mask a
    boundary -- soft_ce_at_answer_position never looks at which token was
    actually appended, only at the mask position it produced (completions
    are always exactly one token, so there's always exactly one such
    position -- verified in verify_novel.py)."""

    def __init__(self, examples, tokenizer, max_seq_length):
        self.examples = []
        for ex in examples:
            tok_ex = tokenize_example(ex["prompt"], ex["completion"], tokenizer, max_seq_length)
            tok_ex["target_dist"] = [ex["target_dist"][l] for l in LETTERS]
            tok_ex["lang"] = ex["lang"]
            self.examples.append(tok_ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_lang_examples(lang: str, rows: list, hard_labels: bool = False) -> list:
    """Per-row dicts carrying everything needed for both the primary
    (unpermuted) tokenized dataset and on-the-fly permuted variants:
    {lang, row, options, target_dist, prompt, completion}.

    hard_labels=True collapses id's vote distribution to a one-hot majority
    label (ties broken by sorted-first letter) instead of using the full
    empirical distribution -- the ablation grid's "hard" control condition
    for Mechanism A, NOT the default."""
    examples = []
    for row in rows:
        options = resolve_options(lang, row)
        target_dist = build_target_dist(lang, row["Gold_Answer"])
        if hard_labels:
            top_letter = max(target_dist, key=lambda l: (target_dist[l], -LETTERS.index(l)))
            target_dist = {l: (1.0 if l == top_letter else 0.0) for l in LETTERS}
        prompt = build_prompt(lang, row, options)
        letter = max(target_dist, key=target_dist.get)
        examples.append({
            "lang": lang, "row": row, "options": options, "target_dist": target_dist,
            "prompt": prompt, "completion": f" {letter}",
        })
    return examples


def build_permuted_tokenized_example(raw_ex: dict, tokenizer, max_seq_length: int, rng: random.Random):
    perm = sample_permutation(raw_ex["lang"], rng)
    new_options, new_target = permute_example(raw_ex["options"], raw_ex["target_dist"], perm)
    prompt = build_prompt(raw_ex["lang"], raw_ex["row"], new_options)
    letter = max(new_target, key=new_target.get)
    tok = tokenize_example(prompt, f" {letter}", tokenizer, max_seq_length)
    tok["target_dist"] = [new_target[l] for l in LETTERS]
    tok["lang"] = raw_ex["lang"]
    return tok, perm


def collate_soft(examples, pad_token_id, device):
    max_len = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    target_dist = torch.zeros((len(examples), 4), dtype=torch.float32)
    for i, e in enumerate(examples):
        L = len(e["input_ids"])
        input_ids[i, :L] = torch.tensor(e["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(e["labels"], dtype=torch.long)
        attention_mask[i, :L] = 1
        target_dist[i] = torch.tensor(e["target_dist"], dtype=torch.float32)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "target_dist": target_dist.to(device),
    }


# ---------------------------------------------------------------------------
# Losses / metric
# ---------------------------------------------------------------------------

def soft_ce_at_answer_position(logits_shifted, labels_shifted, target_dist_batch, letter_ids):
    """Soft cross-entropy against a target distribution over {A,B,C,D} at the
    single answer-token position each row's -100 mask exposes. Equivalent to
    KL(target || model) up to a model-independent constant (target's own
    entropy), and numerically IDENTICAL to F.cross_entropy when target_dist
    is one-hot -- this is what makes it a strict generalization of hard-label
    CE rather than a special-cased alternative (verified in
    verify_novel.py's loss-equivalence check).

    Returns (mean loss, mean continuous metric, logits at the answer
    position) -- the logits are returned so the caller can reuse them for
    consistency_loss without a second forward pass."""
    valid = (labels_shifted != -100)
    pos = valid.float().argmax(dim=1)
    logits_at_pos = logits_shifted[torch.arange(logits_shifted.size(0), device=logits_shifted.device), pos]
    log_probs = F.log_softmax(logits_at_pos, dim=-1)
    letter_log_probs = log_probs[:, letter_ids]  # (B, 4), columns ordered A,B,C,D
    per_example_loss = -(target_dist_batch * letter_log_probs).sum(dim=1)
    with torch.no_grad():
        metric = (target_dist_batch * letter_log_probs.exp()).sum(dim=1)
    return per_example_loss.mean(), metric.mean().item(), logits_at_pos


def perm_to_idx_tensor(perms: list, device) -> torch.Tensor:
    """perms: list of {old_letter: new_letter} dicts, length B. Returns a
    (B,4) long tensor where row b, column i (canonical letter LETTERS[i])
    gives the index within the PERMUTED example's letter ordering where that
    canonical letter's content now sits -- i.e. LETTERS.index(perm[l])."""
    idx = torch.zeros(len(perms), 4, dtype=torch.long)
    for b, perm in enumerate(perms):
        for i, l in enumerate(LETTERS):
            idx[b, i] = LETTERS.index(perm[l])
    return idx.to(device)


def consistency_loss(logits_at_pos_orig, logits_at_pos_perm, perm_idx, letter_ids):
    """Jensen-Shannon divergence (symmetric, bounded -- unlike raw KL, which
    blows up as either side's distribution sharpens toward one-hot late in
    training) between the original example's predicted A/B/C/D distribution
    and the permuted variant's, after remapping the permuted variant's
    distribution back into canonical (original) letter order."""
    q_o = F.softmax(logits_at_pos_orig[:, letter_ids], dim=-1)
    q_p_raw = F.softmax(logits_at_pos_perm[:, letter_ids], dim=-1)
    q_p = torch.gather(q_p_raw, 1, perm_idx)
    m = 0.5 * (q_o + q_p)
    js = (
        0.5 * (q_o * (q_o.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(1)
        + 0.5 * (q_p * (q_p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(1)
    )
    return js.mean()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def lang_step(model, tokenizer, dataset, raw_examples, batch_size, letter_ids, device, rng,
              max_seq_length, pad_token_id, consistency_lambda):
    """One language's contribution to a training step: samples a batch (with
    replacement, matching train_macro_lora_pt.sample_batch's "loop/upsample"
    behavior for smaller languages), builds a permuted variant of each row
    fresh (a new random permutation every time the row is drawn -- stronger
    regularization signal than a permutation fixed once at dataset
    construction), forwards original+permuted together in ONE batched call,
    and combines soft-CE + JS consistency into a single per-language loss."""
    idxs = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    orig_examples = [dataset[i] for i in idxs]

    if consistency_lambda > 0:
        perm_pairs = [build_permuted_tokenized_example(raw_examples[i], tokenizer, max_seq_length, rng)
                      for i in idxs]
        perm_examples = [p[0] for p in perm_pairs]
        perms = [p[1] for p in perm_pairs]
        combined = orig_examples + perm_examples
    else:
        combined = orig_examples

    batch = collate_soft(combined, pad_token_id, device)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits_shifted = out.logits[:, :-1, :]
    labels_shifted = batch["labels"][:, 1:]
    B = len(orig_examples)

    loss_orig, metric_orig, logits_at_pos_orig = soft_ce_at_answer_position(
        logits_shifted[:B], labels_shifted[:B], batch["target_dist"][:B], letter_ids,
    )

    if consistency_lambda > 0:
        loss_perm, metric_perm, logits_at_pos_perm = soft_ce_at_answer_position(
            logits_shifted[B:], labels_shifted[B:], batch["target_dist"][B:], letter_ids,
        )
        perm_idx = perm_to_idx_tensor(perms, device)
        c_loss = consistency_loss(logits_at_pos_orig, logits_at_pos_perm, perm_idx, letter_ids)
        lang_loss = 0.5 * (loss_orig + loss_perm) + consistency_lambda * c_loss
        metric = 0.5 * (metric_orig + metric_perm)
        c_loss_value = c_loss.item()
    else:
        lang_loss = loss_orig
        metric = metric_orig
        c_loss_value = 0.0

    return lang_loss, metric, c_loss_value


def soft_lang_backward(model, tokenizer, datasets, raw_examples_by_lang, batch_size, letter_ids,
                        device, rng, max_seq_length, pad_token_id, consistency_lambda):
    """Soft/consistency-aware analog of train_macro_lora_pt.macro_lang_backward
    -- three separate per-language .backward() calls (equal-weight macro
    average, mirroring the shared task's macro-accuracy metric), each freeing
    its forward graph immediately rather than keeping all three alive."""
    total = 0.0
    lang_metric = {}
    lang_closs = {}
    for lang in LANGS:
        lang_loss, metric, c_loss_value = lang_step(
            model, tokenizer, datasets[lang], raw_examples_by_lang[lang], batch_size, letter_ids,
            device, rng, max_seq_length, pad_token_id, consistency_lambda,
        )
        (lang_loss / len(LANGS)).backward()
        total += lang_loss.item()
        lang_metric[lang] = metric
        lang_closs[lang] = c_loss_value
    return total / len(LANGS), lang_metric, lang_closs


@torch.no_grad()
def per_language_soft_validate(model, val_datasets, batch_size, val_batches, letter_ids,
                                pad_token_id, device, rng):
    """Macro (equal-weight) val loss + continuous metric, ORIGINAL variant
    only (no permutation/consistency term at validation time -- consistency
    regularization is a training-time inductive bias, not part of the
    quantity used for model selection/early stopping)."""
    model.eval()
    lang_losses, lang_metrics = {}, {}
    for lang, ds in val_datasets.items():
        losses, metrics = [], []
        for _ in range(val_batches):
            idxs = [rng.randrange(len(ds)) for _ in range(batch_size)]
            batch = collate_soft([ds[i] for i in idxs], pad_token_id, device)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits_shifted = out.logits[:, :-1, :]
            labels_shifted = batch["labels"][:, 1:]
            loss, metric, _ = soft_ce_at_answer_position(
                logits_shifted, labels_shifted, batch["target_dist"], letter_ids,
            )
            losses.append(loss.item())
            metrics.append(metric)
        lang_losses[lang] = sum(losses) / len(losses)
        lang_metrics[lang] = sum(metrics) / len(metrics)
        tqdm.write(f"  val_loss[{lang}]={lang_losses[lang]:.4f} val_metric[{lang}]={lang_metrics[lang]:.4f}")
    model.train()
    macro_loss = sum(lang_losses.values()) / len(lang_losses)
    macro_metric = sum(lang_metrics.values()) / len(lang_metrics)
    return macro_loss, macro_metric, lang_losses, lang_metrics


# ---------------------------------------------------------------------------
# Data assembly (CV round vs. final refit)
# ---------------------------------------------------------------------------

def load_split(args):
    """Returns (train_rows_by_lang, val_rows_by_lang). val is {} in
    --final-refit mode (100% of dev data used for training, no held-out
    set -- early stopping isn't possible there; the iteration budget is a
    fixed CV-selected constant instead, per --refit-iters)."""
    train_rows, val_rows = {}, {}
    for lang in LANGS:
        rows = load_rows(lang)
        if args.final_refit:
            train_rows[lang] = rows
            val_rows[lang] = []
        else:
            folds = assign_folds(rows, lang, args.cv_folds, args.fold_seed)
            train_rows[lang] = [r for r, f in zip(rows, folds) if f != args.cv_round]
            val_rows[lang] = [r for r, f in zip(rows, folds) if f == args.cv_round]
    return train_rows, val_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    ap.add_argument("--load-in-4bit", action="store_true",
                     help="QLoRA-style NF4 4-bit base model (CUDA only) -- shrinks the base "
                          "model footprint from ~8GB to ~2.5-3GB, for tight-VRAM GPUs (e.g. T4).")

    # Data protocol: either one CV round (train on folds != R, val on fold R)
    # or a final refit (100% of dev data, fixed iteration budget, no val).
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-round", type=int, default=0)
    ap.add_argument("--final-refit", action="store_true")
    ap.add_argument("--refit-iters", type=int, default=None,
                     help="required with --final-refit: fixed iteration budget (no early "
                          "stopping possible without a held-out val set), typically the "
                          "median best-iteration across CV rounds")

    # Mechanism toggles (ablation grid).
    ap.add_argument("--hard-labels", action="store_true",
                     help="disable Mechanism A: collapse id's vote distribution to a one-hot "
                          "majority label instead of using the full empirical distribution")
    ap.add_argument("--consistency-lambda", type=float, default=0.5,
                     help="Mechanism B weight; 0 disables positional-consistency regularization "
                          "entirely (skips the permuted forward pass for speed)")

    # Training loop.
    ap.add_argument("--per-lang-batch-size", type=int, default=2)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=5)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5,
                     help="stop after this many consecutive evals with no macro val-loss "
                          "improvement (0 disables; ignored in --final-refit mode)")
    ap.add_argument("--learning-rate", type=float, default=1e-4)

    ap.add_argument("--out-dir", default=None,
                     help="defaults to results/novel_cv/round_{R} or results/novel_refit")
    args = ap.parse_args()

    if args.final_refit and args.refit_iters is None:
        ap.error("--final-refit requires --refit-iters")

    if args.out_dir is None:
        args.out_dir = "results/novel_refit" if args.final_refit else f"results/novel_cv/round_{args.cv_round}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device == "cuda" and not torch.cuda.is_bf16_supported() else torch.bfloat16

    if args.load_in_4bit and device != "cuda":
        ap.error("--load-in-4bit requires CUDA")

    print(f"Loading {args.model_id} on {device} ({dtype}){' [4-bit NF4]' if args.load_in_4bit else ''} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = torch.tensor([letter_token_ids(tokenizer)[l] for l in LETTERS], device=device)

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config, device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if args.grad_checkpoint and not args.load_in_4bit:
        # prepare_model_for_kbit_training already wired gradient checkpointing
        # up above when --load-in-4bit is set; enabling it again here would
        # just be a redundant no-op, but skip it for clarity.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled.")
    if not args.load_in_4bit:
        model.to(device)
    model.train()

    train_rows, val_rows = load_split(args)
    raw_examples_by_lang = {
        lang: build_lang_examples(lang, train_rows[lang], hard_labels=args.hard_labels)
        for lang in LANGS
    }
    train_datasets = {
        lang: SoftPromptCompletionDataset(raw_examples_by_lang[lang], tokenizer, args.max_seq_length)
        for lang in LANGS
    }
    val_datasets = None
    if not args.final_refit:
        val_raw = {
            lang: build_lang_examples(lang, val_rows[lang], hard_labels=args.hard_labels)
            for lang in LANGS
        }
        val_datasets = {
            lang: SoftPromptCompletionDataset(val_raw[lang], tokenizer, args.max_seq_length)
            for lang in LANGS
        }
        for lang in LANGS:
            print(f"{lang}: {len(train_datasets[lang])} train, {len(val_datasets[lang])} val "
                  f"(fold {args.cv_round}/{args.cv_folds})")
    else:
        for lang in LANGS:
            print(f"{lang}: {len(train_datasets[lang])} train (100% dev data, final refit)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)

    adapter_path = out_dir / "adapter"
    best_adapter_path = adapter_path / "best"
    best_val_loss = float("inf")
    best_iter = 0
    stale_evals = 0

    iters = args.refit_iters if args.final_refit else args.iters
    losses, steps, t0 = 0.0, 0, time.time()
    metric_sums = {lang: 0.0 for lang in LANGS}
    last_summary = None
    pbar = tqdm(range(1, iters + 1), desc="train", unit="it")
    for it in pbar:
        optimizer.zero_grad()
        loss_value, lang_metric, lang_closs = soft_lang_backward(
            model, tokenizer, train_datasets, raw_examples_by_lang, args.per_lang_batch_size,
            letter_ids, device, rng, args.max_seq_length, tokenizer.pad_token_id,
            args.consistency_lambda,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        losses += loss_value
        steps += 1
        for lang in LANGS:
            metric_sums[lang] += lang_metric[lang]
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        metric_str = " ".join(f"metric[{lang}]={lang_metric[lang]:.3f}" for lang in LANGS)
        closs_str = " ".join(f"closs[{lang}]={lang_closs[lang]:.3f}" for lang in LANGS)
        tqdm.write(f"[{it}/{iters}] loss={loss_value:.4f} {metric_str} {closs_str}")

        if it % args.steps_per_report == 0 or it == iters:
            elapsed = time.time() - t0
            metric_str = " ".join(f"metric[{lang}]={metric_sums[lang]/steps:.3f}" for lang in LANGS)
            tqdm.write(f"[{it}] train_loss(macro)={losses/steps:.4f} {metric_str} elapsed={elapsed:.0f}s")
            losses, steps = 0.0, 0
            metric_sums = {lang: 0.0 for lang in LANGS}

        if not args.final_refit and (it % args.steps_per_eval == 0 or it == iters):
            val_loss, val_metric, lang_losses, lang_metrics = per_language_soft_validate(
                model, val_datasets, args.per_lang_batch_size, args.val_batches, letter_ids,
                tokenizer.pad_token_id, device, rng,
            )
            tqdm.write(f"[{it}] val_loss(macro)={val_loss:.4f} val_metric(macro)={val_metric:.4f}")
            stop_early = False
            if val_loss < best_val_loss:
                best_val_loss, best_iter, stale_evals = val_loss, it, 0
                model.save_pretrained(str(best_adapter_path))
                tqdm.write(f"[{it}] New best macro val loss; saved to {best_adapter_path}")
            else:
                stale_evals += 1
                tqdm.write(f"[{it}] No improvement ({stale_evals}/{args.patience})")
                if args.patience > 0 and stale_evals >= args.patience:
                    tqdm.write(f"[{it}] Early stopping.")
                    stop_early = True
            save_best_known(model, adapter_path, best_adapter_path)
            last_summary = {
                "cv_round": args.cv_round, "cv_folds": args.cv_folds, "iter": it,
                "best_iter": best_iter, "best_val_loss": best_val_loss,
                "val_loss_macro": val_loss, "val_metric_macro": val_metric,
                "val_loss_by_lang": lang_losses, "val_metric_by_lang": lang_metrics,
                "consistency_lambda": args.consistency_lambda, "hard_labels": args.hard_labels,
                "val_row_ids": {lang: [r.get("ID") for r in val_rows[lang]] for lang in LANGS},
                "finished": False,
            }
            (out_dir / "summary.json").write_text(json.dumps(last_summary, indent=2))
            if stop_early:
                break

    if args.final_refit:
        model.save_pretrained(str(adapter_path))
        (out_dir / "summary.json").write_text(json.dumps({
            "final_refit": True, "refit_iters": iters,
            "consistency_lambda": args.consistency_lambda, "hard_labels": args.hard_labels,
            "finished": True,
        }, indent=2))
        print(f"Final refit done. Adapter saved to {adapter_path}")
    else:
        save_best_known(model, adapter_path, best_adapter_path)
        # Only mark "finished" once training has genuinely stopped on purpose
        # (iteration ceiling reached or early stopping fired) -- NOT when the
        # process is killed mid-loop (OOM, crash, disconnect), so a resumed
        # sweep can tell a truncated round apart from a real one.
        if last_summary is not None:
            last_summary["finished"] = True
            (out_dir / "summary.json").write_text(json.dumps(last_summary, indent=2))
        print(f"CV round {args.cv_round}/{args.cv_folds} done. "
              f"best_iter={best_iter} best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
