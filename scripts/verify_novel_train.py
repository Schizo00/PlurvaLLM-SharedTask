"""Checks 6-9 of the verification suite (need train_novel.py / torch / the
base model). Split out from verify_novel.py so checks 1-5 (pure data-layer,
fast) can run standalone without importing torch at all.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise AssertionError(label)


# ---------------------------------------------------------------------------
# 6. Loss-equivalence unit check
# ---------------------------------------------------------------------------

def verify_loss_equivalence():
    import torch
    import torch.nn.functional as F
    from train_novel import soft_ce_at_answer_position

    torch.manual_seed(0)
    B, T, V = 4, 6, 20
    logits_shifted = torch.randn(B, T, V)
    labels_shifted = torch.full((B, T), -100, dtype=torch.long)
    pos = torch.tensor([2, 3, 1, 4])
    letter_ids = torch.tensor([5, 6, 7, 8])
    gold_letter_idx = torch.tensor([0, 2, 1, 3])  # which of the 4 letters is "correct"

    for b in range(B):
        labels_shifted[b, pos[b]] = letter_ids[gold_letter_idx[b]]
    target_dist = torch.zeros(B, 4)
    for b in range(B):
        target_dist[b, gold_letter_idx[b]] = 1.0

    soft_loss, metric, _ = soft_ce_at_answer_position(logits_shifted, labels_shifted, target_dist, letter_ids)

    ce_losses = [
        F.cross_entropy(logits_shifted[b, pos[b]].unsqueeze(0), labels_shifted[b, pos[b]].unsqueeze(0))
        for b in range(B)
    ]
    hard_loss = torch.stack(ce_losses).mean()

    check(f"soft CE (one-hot target) == F.cross_entropy: soft={soft_loss.item():.6f} hard={hard_loss.item():.6f}",
          torch.allclose(soft_loss, hard_loss, atol=1e-5))
    check(f"metric in [0,1] (got {metric})", 0.0 <= metric <= 1.0)


# ---------------------------------------------------------------------------
# 7. Consistency-loss identity check
# ---------------------------------------------------------------------------

def verify_consistency_loss_identity():
    import torch
    from train_novel import consistency_loss, perm_to_idx_tensor

    torch.manual_seed(0)
    B = 5
    logits = torch.randn(B, 20)
    letter_ids = torch.tensor([5, 6, 7, 8])
    identity_perms = [{"A": "A", "B": "B", "C": "C", "D": "D"} for _ in range(B)]
    perm_idx = perm_to_idx_tensor(identity_perms, device="cpu")

    c = consistency_loss(logits, logits, perm_idx, letter_ids)
    check(f"consistency_loss ~ 0 under identity permutation + identical logits (got {c.item():.8f})",
          c.item() < 1e-6)

    # Sanity: a genuinely different permutation with different logits should
    # NOT be ~0 in general (catches a perm_idx indexing bug that would make
    # every case degenerate to 0).
    non_identity_perms = [{"A": "B", "B": "A", "C": "C", "D": "D"} for _ in range(B)]
    perm_idx2 = perm_to_idx_tensor(non_identity_perms, device="cpu")
    logits2 = torch.randn(B, 20)
    c2 = consistency_loss(logits, logits2, perm_idx2, letter_ids)
    check(f"consistency_loss is NOT trivially ~0 for differing logits/permutation (got {c2.item():.6f})",
          c2.item() > 1e-4)


# ---------------------------------------------------------------------------
# 8/9. Tiny end-to-end smoke test + CV round independence
# ---------------------------------------------------------------------------

def _run_cv_round(model_id, round_idx, out_dir, iters=3, cv_folds=2):
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "train_novel.py"),
        "--model-id", model_id,
        "--cv-folds", str(cv_folds), "--cv-round", str(round_idx),
        "--iters", str(iters), "--per-lang-batch-size", "1",
        "--steps-per-eval", "1", "--val-batches", "1", "--patience", "0",
        "--out-dir", out_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return result


def verify_smoke_test(model_id="Qwen/Qwen3.5-4B"):
    out_dir = "/tmp/novel_smoke_round0"
    result = _run_cv_round(model_id, 0, out_dir)
    check(f"smoke test (round 0) exit code 0 (stderr tail:\n{result.stderr[-3000:]})", result.returncode == 0)
    check("no 'nan' loss in stdout", "loss=nan" not in result.stdout.lower())
    summary = json.loads((Path(out_dir) / "summary.json").read_text())
    check(f"val_metric_macro in [0,1] (got {summary['val_metric_macro']})",
          0.0 <= summary["val_metric_macro"] <= 1.0)
    print(f"  round 0 summary: best_iter={summary['best_iter']} "
          f"val_loss={summary['val_loss_macro']:.4f} val_metric={summary['val_metric_macro']:.4f}")
    return summary


def verify_cv_round_independence(model_id="Qwen/Qwen3.5-4B"):
    out_dir0 = "/tmp/novel_smoke_round0"
    out_dir1 = "/tmp/novel_smoke_round1"
    if not (Path(out_dir0) / "summary.json").exists():
        _run_cv_round(model_id, 0, out_dir0)
    result1 = _run_cv_round(model_id, 1, out_dir1)
    check(f"smoke test (round 1) exit code 0 (stderr tail:\n{result1.stderr[-3000:]})", result1.returncode == 0)

    summary0 = json.loads((Path(out_dir0) / "summary.json").read_text())
    summary1 = json.loads((Path(out_dir1) / "summary.json").read_text())
    for lang in ["zh", "id", "si"]:
        ids0 = set(summary0["val_row_ids"][lang])
        ids1 = set(summary1["val_row_ids"][lang])
        check(f"{lang}: round 0 and round 1 val-fold row IDs are disjoint "
              f"(round0 n={len(ids0)}, round1 n={len(ids1)}, overlap={len(ids0 & ids1)})",
              len(ids0 & ids1) == 0)


if __name__ == "__main__":
    print("== 6. Loss-equivalence unit check ==")
    verify_loss_equivalence()
    print("\n== 7. Consistency-loss identity check ==")
    verify_consistency_loss_identity()
    print("\n== 8. Tiny end-to-end smoke test ==")
    verify_smoke_test()
    print("\n== 9. CV round independence ==")
    verify_cv_round_independence()
    print("\nAll checks (6-9) passed.")
