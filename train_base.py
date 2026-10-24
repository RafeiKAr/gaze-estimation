"""Train and save the GazeCapture base models used for personalization.

The configuration is the one selected by the hyperparameter study
(exp.py DEFAULT_CONFIG, see exp_results_all.json): FC only, Sigmoid, L1,
lr 1e-4, batch 32, 1000/200 frames drawn at random (seed 0), early stopping on
a subject-disjoint validation set of 200 frames.

For each split (norm_subject, norm_random) the model is trained with several
seeds; the run with the lowest VALIDATION error is kept (the test set is not
used for this choice). The head is trained on cached ResNet features, then
copied into a full ResNet-18 model, which is saved as

    models/base/base_<split>.pt          (state_dict of the full model)
    models/base/base_<split>.json        (config, seed, errors of all seeds,
                                          same-subset baselines)

personal_exp.py loads exactly these files.

Usage (project root):
    python train_base.py                                  # both splits, seeds 0-4
    python train_base.py --splits norm_subject --seeds 0 1 2
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import exp
from dataset import build_embedding_dataloaders
from model import build_frozen_backbone, build_head, build_model, train_model

OUT_DIR = Path("models/base")


def train_head(config, seed, device, backbone):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, val_loader = build_embedding_dataloaders(
        split_type=config["split_type"], train_size=config["train_size"], test_size=config["test_size"],
        batch_size=config["batch_size"], backbone=backbone, device=device,
        cache_dir=exp.EMBEDDING_CACHE_DIR, sample_seed=config["sample_seed"], val_size=config["val_size"])
    head = build_head(activation=config["activation"])
    initial = copy.deepcopy(head.state_dict())
    tmp = OUT_DIR / f"_tmp_{config['split_type']}_s{seed}"
    res = train_model(head, "head", train_loader, test_loader, device,
                      epochs=config["epochs"], learning_rate=config["learning_rate"],
                      weight_decay=config["weight_decay"], patience=config["patience"],
                      trainable_layers=None, loss_name=config["loss"], model_dir=tmp,
                      verbose=False, val_loader=val_loader)
    best_file = tmp / "best_head.path"
    if res["best_epoch"] > 0 and best_file.exists():
        head.load_state_dict(torch.load(best_file, map_location="cpu", weights_only=True))
    else:                                       # no improvement: keep the untrained head
        head.load_state_dict(initial)
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return head, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["norm_subject", "norm_random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backbone = build_frozen_backbone()

    for split in args.splits:
        config = {**exp.DEFAULT_CONFIG, "split_type": split, "sample_seed": 0, "val_size": 200}
        runs, best = [], None
        for seed in args.seeds:
            t0 = time.perf_counter()
            head, res = train_head(config, seed, device, backbone)
            val = res["val_errors"][res["best_epoch"]]
            test = res["test_errors"][res["best_epoch"]]
            runs.append({"seed": seed, "val_error": val, "test_error": test,
                         "epoch0_test_error": res["epoch0_test_error"], "best_epoch": res["best_epoch"],
                         "epochs": res["epochs_run"], "minutes": (time.perf_counter() - t0) / 60})
            print(f"[{split}] seed {seed}: val {val:.2f}%  test {test:.2f}%  best epoch {res['best_epoch']}")
            if best is None or val < best[0]:
                best = (val, seed, head)

        model, _ = build_model(activation=config["activation"])
        model.fc.load_state_dict(best[2].state_dict())
        torch.save(model.state_dict(), OUT_DIR / f"base_{split}.pt")
        meta = {"split": split, "config": config, "selected_seed": best[1],
                "selection": "lowest validation error", "runs": runs,
                "baseline_same_subset": exp.compute_subset_baselines(config),
                "created": time.strftime("%Y-%m-%d %H:%M")}
        (OUT_DIR / f"base_{split}.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
        print(f"[{split}] saved {OUT_DIR / f'base_{split}.pt'} (seed {best[1]}, val {best[0]:.2f}%)")


if __name__ == "__main__":
    main()
