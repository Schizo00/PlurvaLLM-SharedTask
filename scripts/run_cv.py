"""Orchestrates a k-fold CV sweep for scripts/train_novel.py.

Each round runs as its own SUBPROCESS, not in-process -- a crash or NaN in
one fold doesn't lose the other folds' already-written results (matches this
repo's existing crash-isolated-checkpoint philosophy, e.g.
train_macro_lora_pt.save_best_known). Safe to re-run: rounds whose
results/novel_cv/round_{r}/summary.json already reports a finished round are
skipped, so a partially-completed sweep can be resumed.

Usage:
    python scripts/run_cv.py --cv-folds 5 --iters 300 --consistency-lambda 0.5
    python scripts/run_cv.py --cv-folds 5 --iters 300 --hard-labels --consistency-lambda 0
        # ablation grid config 1 (baseline control): hard labels, no consistency reg

After all rounds finish, prints mean +/- std of val_loss_macro / val_metric_macro
across folds and the median best_iter (rounded up to the nearest
--steps-per-eval multiple) -- the config to pass to train_novel.py
--final-refit --refit-iters <N> for the actual submission checkpoint.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_NOVEL = SCRIPT_DIR / "train_novel.py"


def round_out_dir(base_dir: Path, round_idx: int) -> Path:
    return base_dir / f"round_{round_idx}"


def round_is_done(out_dir: Path, iters: int) -> bool:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    # Rely on the explicit "finished" flag train_novel.py only sets once
    # training stops on purpose (iteration ceiling or early stopping) --
    # NOT "iter >= best_iter", which is true after literally the first eval
    # by construction (best_iter can never exceed the current iter) and
    # would silently mark an interrupted/crashed round as done.
    return summary.get("finished", False)


def run_round(round_idx: int, args, out_dir: Path):
    cmd = [
        sys.executable, str(TRAIN_NOVEL),
        "--model-id", args.model_id,
        "--cv-folds", str(args.cv_folds), "--cv-round", str(round_idx),
        "--fold-seed", str(args.fold_seed), "--seed", str(args.seed),
        "--iters", str(args.iters),
        "--per-lang-batch-size", str(args.per_lang_batch_size),
        "--steps-per-eval", str(args.steps_per_eval),
        "--steps-per-report", str(args.steps_per_report),
        "--val-batches", str(args.val_batches),
        "--patience", str(args.patience),
        "--learning-rate", str(args.learning_rate),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--consistency-lambda", str(args.consistency_lambda),
        "--max-grad-norm", str(args.max_grad_norm),
        "--out-dir", str(out_dir),
    ]
    if args.hard_labels:
        cmd.append("--hard-labels")
    print(f"[run_cv] round {round_idx}/{args.cv_folds}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[run_cv] round {round_idx} FAILED (exit {result.returncode}) -- "
              f"other rounds' results are untouched, re-run this script to retry.", file=sys.stderr)
    return result.returncode == 0


def aggregate(base_dir: Path, cv_folds: int, steps_per_eval: int):
    val_losses, val_metrics, best_iters = [], [], []
    per_round = []
    for r in range(cv_folds):
        summary_path = round_out_dir(base_dir, r) / "summary.json"
        if not summary_path.exists():
            print(f"[run_cv] round {r} has no summary.json yet -- skipping aggregation.")
            return None
        summary = json.loads(summary_path.read_text())
        if not summary.get("finished", False):
            print(f"[run_cv] round {r}'s summary.json is from an interrupted run "
                  f"(finished=False) -- re-run run_cv.py to finish it before aggregating.")
            return None
        val_losses.append(summary["val_loss_macro"])
        val_metrics.append(summary["val_metric_macro"])
        best_iters.append(summary["best_iter"])
        per_round.append(summary)

    median_iter = int(statistics.median(best_iters))
    refit_iters = ((median_iter + steps_per_eval - 1) // steps_per_eval) * steps_per_eval

    agg = {
        "cv_folds": cv_folds,
        "val_loss_macro_mean": statistics.mean(val_losses),
        "val_loss_macro_std": statistics.pstdev(val_losses) if len(val_losses) > 1 else 0.0,
        "val_metric_macro_mean": statistics.mean(val_metrics),
        "val_metric_macro_std": statistics.pstdev(val_metrics) if len(val_metrics) > 1 else 0.0,
        "best_iters": best_iters,
        "median_best_iter": median_iter,
        "suggested_refit_iters": refit_iters,
        "per_round": per_round,
    }
    (base_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"\n[run_cv] {cv_folds}-fold CV complete.")
    print(f"  val_loss_macro:   {agg['val_loss_macro_mean']:.4f} +/- {agg['val_loss_macro_std']:.4f}")
    print(f"  val_metric_macro: {agg['val_metric_macro_mean']:.4f} +/- {agg['val_metric_macro_std']:.4f}")
    print(f"  best_iters per fold: {best_iters} -> median {median_iter}, suggested refit_iters={refit_iters}")
    print(f"\n  Next: python scripts/train_novel.py --final-refit --refit-iters {refit_iters} "
          f"--consistency-lambda {agg['per_round'][0].get('consistency_lambda', 0.5)} ...")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--per-lang-batch-size", type=int, default=2)
    ap.add_argument("--steps-per-eval", type=int, default=5)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--consistency-lambda", type=float, default=0.5)
    ap.add_argument("--hard-labels", action="store_true")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--base-dir", default="results/novel_cv")
    ap.add_argument("--aggregate-only", action="store_true",
                     help="skip running rounds; just aggregate existing summaries")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for r in range(args.cv_folds):
            out_dir = round_out_dir(base_dir, r)
            if round_is_done(out_dir, args.iters):
                print(f"[run_cv] round {r}/{args.cv_folds} already done -- skipping (resume).")
                continue
            ok = run_round(r, args, out_dir)
            if not ok:
                print(f"[run_cv] stopping after round {r} failure. Re-run this script to resume.")
                sys.exit(1)

    aggregate(base_dir, args.cv_folds, args.steps_per_eval)


if __name__ == "__main__":
    main()
