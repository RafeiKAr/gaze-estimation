"""Moving-average smoothing of the personalized gaze predictions (post-hoc).

The experiment does NOT retrain any model. It reads the predictions that
personal_exp.py has already saved and smooths them in time:

  * sequential split  : the test block is the last 15 % of a session, in
                        recording order (18 sessions, seed 42; sessions 21/22
                        also seeds 43/44)
  * two_videos        : train on one session, test on the whole other session
                        (21 -> 22, 22 -> 21, seeds 42-44)
  * demo test points  : five fixed test points of the web demo (optional,
                        demo/debug_frames/tracking_log.json)

The random split is NOT used, because its test frames are not in time order.

Filters (window k in frames, 6 labels per second, k = 1 is "no filter"):
  causal   : mean of the last k predictions (usable live, adds lag)
  centered : mean of k predictions around the frame (offline only, no lag)

Metrics:
  error : diagonal error E_diag in % (accuracy, metrics.diagonal_error_pct)
  s2s   : RMS sample-to-sample change of the prediction in % of the diagonal
          (precision / stability, lower = calmer)

Groups: all 18 sessions, and the 10 sessions in which personalization changed
the model (best epoch > 0). In the other 8 sessions the result is the
unchanged base model (see Chapter 4).

Results are written to results/moving_average/ (new files only):
  ma_runs.csv     one row per run, filter and window
  ma_summary.csv  mean over the 18 sequential sessions + Wilcoxon tests
                  (each window vs. k = 1, Holm correction per metric)
  ma_demo.csv     demo test points (if the log file exists); the first 0.8 s
                  of each point are already removed by the demo

Usage:  python moving_average.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from metrics import diagonal_error_pct

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "results" / "personal" / "predictions"
RUNS_FILE = ROOT / "results" / "personal" / "personal_runs.jsonl"
DEMO_LOG = ROOT / "demo" / "debug_frames" / "tracking_log.json"
OUT_DIR = ROOT / "results" / "moving_average"

LABELS_PER_SEC = 6
WINDOWS = (1, 3, 5, 9, 15)
NAME = re.compile(r"pred_(\d+)-(\d+)_norm_subject_original_(sequential|two_videos)_standard_s(\d+)\.csv")


def causal_ma(p, k):
    """Mean of the last k rows (fewer at the start of the sequence)."""
    c = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), axis=0)
    i = np.arange(1, len(p) + 1)
    lo = np.maximum(0, i - k)
    return (c[i] - c[lo]) / (i - lo)[:, None]


def centered_ma(p, k):
    """Mean of k rows centred on each frame (shorter window at the edges)."""
    h = k // 2
    c = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), axis=0)
    i = np.arange(len(p))
    lo, hi = np.maximum(0, i - h), np.minimum(len(p), i + h + 1)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def s2s_pct(p):
    """RMS sample-to-sample change in % of the screen diagonal."""
    if len(p) < 2:
        return float("nan")
    return float(100.0 * np.sqrt((np.diff(p, axis=0) ** 2).sum(1).mean()) / np.sqrt(2.0))


def evaluate(gt, p):
    return {"error": diagonal_error_pct(gt, p), "s2s": s2s_pct(p)}


def improved_sessions():
    """Sessions in which training changed the model (sequential, seed 42, best epoch > 0)."""
    good = set()
    if not RUNS_FILE.exists():
        return good
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (r.get("split") == "sequential" and r.get("seed") == 42 and r.get("labels") == "original"
                and r.get("base") == "norm_subject" and r["session"] == r["test_session"]
                and r.get("loss", "standard") == "standard" and r.get("best_epoch", 0) > 0):
            good.add(r["session"])
    return good


def run_sessions():
    rows = []
    for f in sorted(PRED_DIR.glob("pred_*.csv")):
        m = NAME.fullmatch(f.name)
        if not m:
            continue
        train, test, split, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        d = pd.read_csv(f)
        gt, p = d[["gt_x", "gt_y"]].to_numpy(float), d[["pred_x", "pred_y"]].to_numpy(float)
        for filt, fn in (("causal", causal_ma), ("centered", centered_ma)):
            for k in WINDOWS:
                if filt == "centered" and k == 1:
                    continue
                rows.append({"train": train, "test": test, "split": split, "seed": seed, "filter": filt,
                             "k": k, "seconds": round(k / LABELS_PER_SEC, 2), "n": len(d),
                             **evaluate(gt, fn(p, k))})
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(runs, good):
    seq = runs[(runs.split == "sequential") & (runs.seed == 42)]
    out = []
    for group, sel in (("all", seq), ("adapted", seq[seq.test.isin(good)])):
        base = sel[(sel["filter"] == "causal") & (sel.k == 1)].set_index("test")
        for filt in ("causal", "centered"):
            block = []
            for k in WINDOWS:
                cur = sel[(sel["filter"] == ("causal" if k == 1 else filt)) & (sel.k == k)].set_index("test")
                row = {"group": group, "filter": filt, "k": k, "seconds": round(k / LABELS_PER_SEC, 2),
                       "sessions": len(cur)}
                for met in ("error", "s2s"):
                    a, b = base[met].align(cur[met], join="inner")
                    ok = a.notna() & b.notna()
                    row[met] = float(b[ok].mean())
                    row[f"{met}_diff"] = float((b[ok] - a[ok]).mean())
                    row[f"p_{met}"] = (float(wilcoxon(b[ok], a[ok]).pvalue)
                                       if k > 1 and ok.sum() >= 5 and np.any(b[ok] != a[ok]) else np.nan)
                block.append(row)
            for met in ("error", "s2s"):
                idx = [i for i, r in enumerate(block) if not np.isnan(r[f"p_{met}"])]
                for i, v in zip(idx, holm([block[i][f"p_{met}"] for i in idx])):
                    block[i][f"p_{met}_holm"] = float(v)
            out += block
    return pd.DataFrame(out)


def run_demo():
    if not DEMO_LOG.exists():
        return None
    log = json.loads(DEMO_LOG.read_text(encoding="utf-8"))
    rows = []
    for k in WINDOWS:
        gts, ps = [], []
        for point in log.get("test_points", []):
            p = np.asarray(point["preds"], float)
            if len(p) == 0:
                continue
            ps.append(causal_ma(p, k))
            gts.append(np.tile(point["target"], (len(p), 1)))
        gt, p = np.vstack(gts), np.vstack(ps)
        rows.append({"k": k, "n": len(p), "error": diagonal_error_pct(gt, p),
                     "s2s": float(np.nanmean([s2s_pct(q) for q in ps]))})
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = run_sessions()
    good = improved_sessions()
    runs.to_csv(OUT_DIR / "ma_runs.csv", index=False)
    summary = summarize(runs, good)
    summary.to_csv(OUT_DIR / "ma_summary.csv", index=False)
    demo = run_demo()
    if demo is not None:
        demo.to_csv(OUT_DIR / "ma_demo.csv", index=False)
    pd.set_option("display.width", 200)
    print("Sessions adapted by training (sequential, seed 42):", sorted(good))
    cols = ["group", "filter", "k", "seconds", "sessions", "error", "error_diff", "p_error_holm", "s2s",
            "s2s_diff", "p_s2s_holm"]
    print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
    tv = runs[runs.split == "two_videos"].groupby(["filter", "k"])[["error", "s2s"]].mean()
    print("\nTwo videos (21->22, 22->21, seeds 42-44), mean:\n", tv.round(2))
    if demo is not None:
        print("\nDemo test points (causal):\n", demo.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
