"""Personalization experiments on the recorded sessions (command line).

One run = one session: the GazeCapture base model is loaded, only the
regression head (FC) is trained on the session's frames, early stopping uses
the validation split, and the test error is reported at the best validation
epoch. Every finished run is appended immediately to a JSON-lines file, so an
interrupted series can be resumed: runs that are already in the file are
skipped.

Splits of a session (--split)
    random      70/15/15 random frames (seed). Neighbouring video frames are
                almost identical, so train and test share near-duplicates;
                this split is optimistic.
    sequential  temporal blocks: first 70 % train, next 15 % validation,
                last 15 % test. Measures how well the calibration transfers
                to a LATER part of the recording (realistic use).
                Between the blocks --gap-sec seconds of frames are dropped
                (default 1 s = 6 labels), so that no near-identical
                neighbouring frames end up on both sides of a block border.
    two_videos  train/validation (80/20 random) from one session, test on
                ALL frames of another session (--pairs 21:22,...). Measures
                transfer to a different recording of the same person.

Base models: models/base/base_<norm_subject|norm_random>.pt, created by
train_base.py (best run of the hyperparameter configuration, selected by
validation error). Only the regression head is trained; the backbone stays
completely frozen, including the BatchNorm statistics (--bn frozen, default).
--bn adapt reproduces the old notebook behaviour (BatchNorm running statistics
are updated during training).

Every run also stores its test predictions in results/personal/predictions/
(needed for the centre/edge/corner analysis, spatial_analysis.py).

Examples
    # E0: base models (once)
    python train_base.py
    # E1: all 18 sessions, both base models, original labels
    python personal_exp.py --sessions all --base norm_subject norm_random
    # E1b: Gen 4 with cleaned labels (WAIT frames removed, see session_design.py)
    python personal_exp.py --sessions gen4 --labels clean
    # E2: data loading strategies on sessions 21 and 22, three seeds
    python personal_exp.py --sessions 21 22 --split random --seeds 42 43 44
    python personal_exp.py --sessions 21 22 --split sequential --seeds 42 43 44
    python personal_exp.py --split two_videos --pairs 21:22,22:21 --seeds 42 43 44
    # tables from the stored runs (+ LaTeX rows of the session table)
    python personal_exp.py --summary
    python personal_exp.py --summary --latex
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import constant_baseline_pct, diagonal_error_pct, random_baseline_pct

SESSIONS_GEN = {
    "gen1": ["01", "02", "03", "04"],
    "gen2": ["05", "06", "07", "08", "09"],
    "gen4": ["19", "20", "21", "22", "23", "24", "25", "26", "27"],
}
ALL_SESSIONS = [s for v in SESSIONS_GEN.values() for s in v]
LABEL_FILES = {"original": "norm_labels.csv", "clean": "norm_labels_clean.csv"}
SESSION_ROOT = Path("personalization")
CHECKPOINT = "models/base/base_{base}.pt"          # written by train_base.py
LABELS_PER_SEC = 6                                  # 30 fps video, one label every 5 frames
DEFAULT_OUT = Path("results/personal/personal_runs.jsonl")


# ----------------------------------------------------------------------------
# Splits (pure numpy, testable without torch)
# ----------------------------------------------------------------------------
def split_indices(n, split="random", seed=42, fractions=(0.70, 0.15), gap=0):
    """Return (train, val, test) index arrays for a session with n frames.

    sequential: temporal blocks; the first `gap` frames of the validation and
    of the test block are dropped so that the blocks are separated in time.
    """
    n_tr, n_va = int(n * fractions[0]), int(n * fractions[1])
    if split == "random":
        idx = np.random.default_rng(seed).permutation(n)
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    if split == "sequential":
        idx = np.arange(n)                       # recording order
        return idx[:n_tr], idx[n_tr + gap:n_tr + n_va], idx[n_tr + n_va + gap:]
    raise ValueError(f"unknown split {split!r}")


def two_video_indices(n_train_session, seed=42, val_fraction=0.20):
    """Train/val indices (80/20 random) inside the training session; test = other session."""
    idx = np.random.default_rng(seed).permutation(n_train_session)
    n_va = int(n_train_session * val_fraction)
    return idx[n_va:], idx[:n_va]


def resolve_sessions(tokens):
    out = []
    for t in tokens:
        t = t.lower()
        if t == "all":
            out += ALL_SESSIONS
        elif t in SESSIONS_GEN:
            out += SESSIONS_GEN[t]
        else:
            out.append(t.zfill(2))
    return list(dict.fromkeys(out))


def load_labels(session, labels="original"):
    df = pd.read_csv(SESSION_ROOT / session / LABEL_FILES[labels])
    if not {"frame", "x", "y"}.issubset(df.columns):
        raise ValueError(f"{session}: CSV needs columns frame, x, y")
    return df


# ----------------------------------------------------------------------------
# Training (torch)
# ----------------------------------------------------------------------------
def _torch_parts():
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    class FrameSet(Dataset):
        def __init__(self, session, df, indices, tf):
            root = SESSION_ROOT / session
            self.root = root / "images2" if (root / "images2").is_dir() else root / "images"
            self.df = df.iloc[np.asarray(indices)].reset_index(drop=True)
            self.tf = tf

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            with Image.open(self.root / str(row["frame"])) as img:
                image = self.tf(img.convert("RGB"))
            return image, torch.tensor([row["x"], row["y"]], dtype=torch.float32)

    return torch, nn, DataLoader, FrameSet, train_tf, eval_tf


def run_one(session, base="norm_subject", labels="original", split="random", test_session=None,
            loss="standard", lr=1e-4, batch_size=32, epochs=500, patience=3, seed=42,
            num_workers=4, device=None, save_predictions=None, verbose=True, gap=0, bn="frozen"):
    torch, nn, DataLoader, FrameSet, train_tf, eval_tf = _torch_parts()
    from model import build_model, predict, sensitive_loss

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_labels(session, labels)
    if split == "two_videos":
        if not test_session:
            raise ValueError("two_videos needs a test session (--pairs)")
        tr, va = two_video_indices(len(df), seed)
        test_df, test_name = load_labels(test_session, labels), test_session
        te = np.arange(len(test_df))
    else:
        tr, va, te = split_indices(len(df), split, seed, gap=gap)
        test_df, test_name = df, session

    def loader(sess, frame, idx, tf, shuffle):
        return DataLoader(FrameSet(sess, frame, idx, tf), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

    train_loader = loader(session, df, tr, train_tf, True)
    train_eval = loader(session, df, tr, eval_tf, False)
    val_loader = loader(session, df, va, eval_tf, False)
    test_loader = loader(test_name, test_df, te, eval_tf, False)

    model, _ = build_model(activation="Sigmoid")
    ckpt = Path(CHECKPOINT.format(base=base))
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing - run 'python train_base.py' first")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    crit = {"standard": nn.L1Loss(), "sensitive": sensitive_loss}[loss]

    def err(ld):
        t, p = predict(model, ld, device)
        return diagonal_error_pct(t, p)

    t0 = time.perf_counter()
    hist = {"train": [err(train_eval)], "val": [err(val_loader)], "test": [err(test_loader)]}
    best_val, best_ep, wait = hist["val"][0], 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, epochs + 1):
        model.train()
        if bn == "frozen":                       # FC only: keep backbone (incl. BatchNorm) frozen
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        for x, y in train_loader:
            opt.zero_grad()
            l = crit(model(x.to(device)), y.to(device))
            l.backward()
            opt.step()
        for k, ld in (("train", train_eval), ("val", val_loader), ("test", test_loader)):
            hist[k].append(err(ld))
        if verbose:
            print(f"  [{session}] epoch {ep}: train {hist['train'][-1]:.2f}%  val {hist['val'][-1]:.2f}%  "
                  f"test {hist['test'][-1]:.2f}%")
        if hist["val"][-1] < best_val:
            best_val, best_ep, wait = hist["val"][-1], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    minutes = (time.perf_counter() - t0) / 60
    model.load_state_dict(best_state)
    t, p = predict(model, test_loader, device)

    xy_tr = df[["x", "y"]].to_numpy(float)[tr]
    xy_te = test_df[["x", "y"]].to_numpy(float)[te]
    if save_predictions:
        Path(save_predictions).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gt_x": t[:, 0], "gt_y": t[:, 1], "pred_x": p[:, 0], "pred_y": p[:, 1]}).to_csv(
            Path(save_predictions) / f"pred_{session}-{test_name}_{base}_{labels}_{split}_{loss}_s{seed}.csv",
            index=False)

    n_epochs = len(hist["val"]) - 1
    return {
        "session": session, "test_session": test_name, "base": base, "labels": labels, "split": split,
        "loss": loss, "lr": lr, "batch_size": batch_size, "seed": seed, "gap": gap, "bn": bn,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "val_error": best_val, "test_error": diagonal_error_pct(t, p),
        "epoch0_test_error": hist["test"][0],          # base model, no personalization
        "best_epoch": best_ep, "epochs": n_epochs,
        "minutes": minutes, "minutes_per_epoch": minutes / max(n_epochs, 1),
        "const_baseline": constant_baseline_pct(xy_tr, xy_te),
        "random_baseline": random_baseline_pct(xy_te, seed=seed),
        "pred_std": [float(v) for v in p.std(0)],
        "history": hist,
    }


# ----------------------------------------------------------------------------
# Result file
# ----------------------------------------------------------------------------
KEY = ("session", "test_session", "base", "labels", "split", "loss", "seed", "gap", "bn")


def load_runs(path=DEFAULT_OUT):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_key(r):
    return tuple(str(r.get(k)) for k in KEY)


def summary(path=DEFAULT_OUT):
    runs = load_runs(path)
    if not runs:
        print("no runs in", path)
        return None
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in runs])
    gen = {s: g.replace("gen", "Gen ") for g, ss in SESSIONS_GEN.items() for s in ss}
    df["generation"] = df.session.map(gen)
    df["gain_vs_const"] = df.const_baseline - df.test_error
    df["gain_vs_base"] = df.epoch0_test_error - df.test_error
    cols = ["session", "generation", "base", "labels", "split", "test_session", "val_error", "test_error",
            "epoch0_test_error", "const_baseline", "best_epoch", "epochs", "minutes", "minutes_per_epoch"]
    pd.set_option("display.width", 200)
    print(df[cols].round(2).to_string(index=False))
    print("\nMeans per configuration:")
    print(df.groupby(["base", "labels", "split", "generation"])[
        ["test_error", "epoch0_test_error", "const_baseline", "minutes"]].mean().round(2).to_string())
    return df


def ci98(x):
    """Mean and 98 % t confidence interval."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean()
    h = stats.t.ppf(0.99, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def latex_sessions(df, labels="original", split="random"):
    """LaTeX rows of the session table: one row per session, both base models,
    plus mean and 98 % CI per generation and over all sessions."""
    d = df[(df.labels == labels) & (df.split == split)]
    if d.empty:
        print("no runs for", labels, split)
        return
    piv = d.pivot_table(index="session", columns="base",
                        values=["val_error", "test_error", "minutes", "const_baseline"], aggfunc="mean")
    bases = ["norm_subject", "norm_random"]

    def cell(col, base, s, fmt="{:.2f}"):
        try:
            v = piv.loc[s, (col, base)]
            return "---" if pd.isna(v) else fmt.format(v)
        except KeyError:
            return "---"

    print("% Session & Gen & Val & Test & t (subject) & Val & Test & t (random) & Const")
    for s in piv.index:
        gen = next(g for g, ss in SESSIONS_GEN.items() if s in ss).replace("gen", "")
        row = [s, gen] + [cell(c, b, s, "{:.1f}" if c == "minutes" else "{:.2f}")
                          for b in bases for c in ("val_error", "test_error", "minutes")]
        row.append(cell("const_baseline", "norm_subject", s))
        print(" & ".join(row) + r" \\")
    groups = [(g.replace("gen", "Gen "), ss) for g, ss in SESSIONS_GEN.items()] + [("All", ALL_SESSIONS)]
    for name, ss in groups:
        sub = piv.loc[[s for s in ss if s in piv.index]]
        vals = []
        for b in bases:
            for c in ("val_error", "test_error", "minutes"):
                try:
                    m, lo, hi = ci98(sub[(c, b)].to_numpy())
                    vals.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
                except KeyError:
                    vals.append("---")
        m, lo, hi = ci98(sub[("const_baseline", "norm_subject")].to_numpy())
        vals.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
        print(f"% {name}: " + " | ".join(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=["all"], help="all | gen1 gen2 gen4 | 01 02 ...")
    ap.add_argument("--base", nargs="+", default=["norm_subject"], choices=["norm_subject", "norm_random"])
    ap.add_argument("--labels", nargs="+", default=["original"], choices=list(LABEL_FILES))
    ap.add_argument("--split", default="random", choices=["random", "sequential", "two_videos"])
    ap.add_argument("--pairs", default="", help="two_videos: train:test pairs, e.g. 01:02,02:01")
    ap.add_argument("--loss", default="standard", choices=["standard", "sensitive"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="one run per seed")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--gap-sec", type=float, default=1.0,
                    help="sequential split: seconds of frames dropped between the blocks (default 1 s)")
    ap.add_argument("--bn", default="frozen", choices=["frozen", "adapt"],
                    help="BatchNorm statistics of the backbone: frozen (default) or updated (old behaviour)")
    ap.add_argument("--save-predictions", default="results/personal/predictions",
                    help="folder for the prediction CSVs ('' = do not save)")
    ap.add_argument("--summary", action="store_true", help="only print the stored results")
    ap.add_argument("--latex", action="store_true", help="with --summary: print LaTeX rows of the session table")
    ap.add_argument("--dry-run", action="store_true", help="list the planned runs and exit")
    args = ap.parse_args()

    if args.summary:
        df = summary(args.out)
        if args.latex and df is not None:
            latex_sessions(df)
        return

    if args.split == "two_videos":
        if not args.pairs:
            ap.error("--split two_videos needs --pairs, e.g. --pairs 01:02,02:01")
        jobs = [(a.zfill(2), b.zfill(2)) for a, b in (p.split(":") for p in args.pairs.split(","))]
    else:
        jobs = [(s, None) for s in resolve_sessions(args.sessions)]

    gap = int(round(args.gap_sec * LABELS_PER_SEC)) if args.split == "sequential" else 0
    plan = [dict(session=s, test_session=t or s, base=b, labels=l, split=args.split, loss=args.loss, seed=sd,
                 gap=gap, bn=args.bn)
            for s, t in jobs for b in args.base for l in args.labels for sd in args.seeds]
    done = {run_key(r) for r in load_runs(args.out)}
    todo = [p for p in plan if run_key(p) not in done]
    print(f"{len(plan)} runs planned, {len(plan) - len(todo)} already in {args.out}, {len(todo)} to do.")
    if args.dry_run:
        for p in todo:
            print("  ", p)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(todo, 1):
        print(f"\n=== run {i}/{len(todo)}: {p}")
        r = run_one(p["session"], p["base"], p["labels"], p["split"],
                    None if args.split != "two_videos" else p["test_session"], p["loss"],
                    args.lr, args.batch_size, args.epochs, args.patience, p["seed"],
                    args.num_workers, save_predictions=args.save_predictions or None,
                    gap=p["gap"], bn=p["bn"])
        print(f"  -> test {r['test_error']:.2f}% (base model {r['epoch0_test_error']:.2f}%, "
              f"constant {r['const_baseline']:.2f}%), {r['minutes']:.1f} min")
        with out.open("a", encoding="utf-8") as f:          # saved immediately
            f.write(json.dumps(r) + "\n")
    summary(args.out)


if __name__ == "__main__":
    main()
