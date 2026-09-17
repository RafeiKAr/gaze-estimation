"""Reproduce the hyperparameter sweeps reported in results/results.tex.

Each sweep varies a single axis (batch size, learning rate, trainable
layers, last-layer activation, loss function, dataset size, or data split)
while holding every other hyperparameter at DEFAULT_CONFIG, runs it for
several repeats, and reports the mean test diagonal error / training time
together with a 98% confidence interval -- the same statistics used in
results.tex. The one exception is the constant/random baselines, which are
not learned models and are instead evaluated by bootstrap-resampling the
full normalized dataset.

Usage:
    python exp.py --experiment all
    python exp.py --experiment learning_rate --repeats 5 --epochs 500
    python exp.py --experiment baseline --quick   # fast smoke test

Note: the historical numbers in results.tex were produced by manually
re-running notebook cells with hand-edited hyperparameters, so some
sub-experiments were not run at a shared baseline configuration. This
script instead follows the standard ablation convention of holding a
single default configuration fixed except for the swept axis, so absolute
numbers will differ slightly from results.tex even though the methodology
(N repeats + 98% CI) is the same.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import build_dataloaders, build_embedding_dataloaders, load_normalized_targets
from model import build_frozen_backbone, build_head, build_model, confidence_interval, train_model

DEFAULT_CONFIG = {
    "split_type": "norm_random",
    "train_size": 1000,
    "test_size": 200,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "activation": "Sigmoid",
    "trainable_layers": "fc_only",
    "loss": "standard",
    "epochs": 500,
    "patience": 3,
}

CONFIDENCE = 0.98
EMBEDDING_CACHE_DIR = "./cache/embeddings"


def compute_baselines(repeats=30, sample_size=2000, seed=0):
    """Bootstrap-resampled constant and random baseline diagonal error (%)."""
    rng = np.random.default_rng(seed)
    targets = load_normalized_targets()

    const_errors, random_errors = [], []
    for _ in range(repeats):
        idx = rng.integers(0, len(targets), size=sample_size)
        sample = targets[idx]

        const_pred = np.tile(sample.mean(axis=0), (sample_size, 1))
        random_pred = rng.uniform(0, 1, size=(sample_size, 2))

        const_rmse = np.sqrt(np.mean(np.sum((sample - const_pred) ** 2, axis=1)))
        random_rmse = np.sqrt(np.mean(np.sum((sample - random_pred) ** 2, axis=1)))

        const_errors.append(100 * const_rmse / np.sqrt(2))
        random_errors.append(100 * random_rmse / np.sqrt(2))

    return {
        "constant": confidence_interval(const_errors, CONFIDENCE),
        "random": confidence_interval(random_errors, CONFIDENCE),
    }


def run_once(config, device, use_embeddings=True):
    """Run one training config. When the backbone stays frozen (fc_only) and
    use_embeddings is set, train only the head on cached backbone embeddings
    instead of re-running the ResNet forward/backward pass every epoch --
    this is what makes repeated runs of batch_size/learning_rate/activation/
    loss_function sweeps (all fc_only by default) fast after the first one.
    """
    if use_embeddings and config["trainable_layers"] == "fc_only":
        backbone = build_frozen_backbone()
        train_loader, test_loader = build_embedding_dataloaders(
            split_type=config["split_type"],
            train_size=config["train_size"],
            test_size=config["test_size"],
            batch_size=config["batch_size"],
            backbone=backbone,
            device=device,
            cache_dir=EMBEDDING_CACHE_DIR,
        )
        model, model_name = build_head(activation=config["activation"]), "head_only"
        trainable_layers = None
    else:
        train_loader, test_loader = build_dataloaders(
            split_type=config["split_type"],
            train_size=config["train_size"],
            test_size=config["test_size"],
            batch_size=config["batch_size"],
        )
        model, model_name = build_model(activation=config["activation"])
        trainable_layers = config["trainable_layers"]

    result = train_model(
        model,
        model_name,
        train_loader,
        test_loader,
        device,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
        trainable_layers=trainable_layers,
        loss_name=config["loss"],
        model_dir=None,
        verbose=False,
    )
    return result["best_test_error"], result["elapsed_minutes"]


def run_sweep(name, overrides_per_value, repeats, base_config, device, use_embeddings=True):
    """overrides_per_value: dict of {label: config-overrides-dict}."""
    rows = []
    for label, overrides in overrides_per_value.items():
        config = {**base_config, **overrides}

        test_errors, run_times = [], []
        for r in range(repeats):
            print(f"[{name}] {label} - run {r + 1}/{repeats}")
            best_error, minutes = run_once(config, device, use_embeddings=use_embeddings)
            test_errors.append(best_error)
            run_times.append(minutes)

        err_mean, err_lo, err_hi = confidence_interval(test_errors, CONFIDENCE)
        time_mean, time_lo, time_hi = confidence_interval(run_times, CONFIDENCE)

        rows.append({
            "config": label,
            "mean_test_error": err_mean,
            "ci_low": err_lo,
            "ci_high": err_hi,
            "mean_training_time_min": time_mean,
            "runs": test_errors,
            "times": run_times,
        })

        print(
            f"[{name}] {label}: mean_test_error={err_mean:.5f}% "
            f"CI={CONFIDENCE * 100:.0f}%=[{err_lo:.5f}, {err_hi:.5f}] "
            f"mean_time={time_mean:.3f}min"
        )

    return rows


def build_experiments(repeats, base_config, device, use_embeddings=True):
    return {
        "batch_size": lambda: run_sweep(
            "batch_size",
            {str(bs): {"batch_size": bs} for bs in (32, 64, 128)},
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
        "learning_rate": lambda: run_sweep(
            "learning_rate",
            {str(lr): {"learning_rate": lr} for lr in (1e-3, 1e-4, 1e-5, 1e-6)},
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
        "trainable_layers": lambda: run_sweep(
            "trainable_layers",
            {mode: {"trainable_layers": mode} for mode in ("fc_only", "fc_layer4")},
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
        "activation": lambda: run_sweep(
            "activation",
            {act: {"activation": act} for act in ("Sigmoid", "Gaussian", "ReLU", "None")},
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
        "loss_function": lambda: run_sweep(
            "loss_function",
            {loss: {"loss": loss} for loss in ("standard", "sensitive")},
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
        "dataset_size": lambda: run_sweep(
            "dataset_size",
            {
                "10000_2000": {"train_size": 10000, "test_size": 2000},
                "1000_200": {"train_size": 1000, "test_size": 200},
            },
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
        "split_type": lambda: run_sweep(
            "split_type",
            {split: {"split_type": split} for split in ("norm_random", "norm_subject")},
            repeats,
            base_config,
            device,
            use_embeddings,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment",
        choices=["all", "baseline", *[
            "batch_size", "learning_rate", "trainable_layers", "activation", "loss_function",
            "dataset_size", "split_type",
        ]],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per configuration (baseline default 30).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--output-dir", type=str, default="./results/exp_reproduction")
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: tiny dataset/epoch counts, ignores --repeats/--epochs.",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help=(
            "Disable embedding caching for fc_only configs (default: on). Only the "
            "trainable_layers sweep's fc_layer4 arm always retrains the full backbone."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = dict(DEFAULT_CONFIG)
    base_config["epochs"] = args.epochs
    base_config["patience"] = args.patience

    repeats = args.repeats
    if args.quick:
        base_config.update({"train_size": 20, "test_size": 10, "epochs": 1, "patience": 1})
        repeats = 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(repeats, base_config, device, use_embeddings=not args.no_embeddings)
    to_run = list(experiments) if args.experiment == "all" else [args.experiment] if args.experiment != "baseline" else []

    results = {}

    if args.experiment in ("all", "baseline"):
        baseline_repeats = 2 if args.quick else 30
        baselines = compute_baselines(repeats=baseline_repeats)
        results["baseline"] = {
            name: {"mean": mean, "ci_low": lo, "ci_high": hi}
            for name, (mean, lo, hi) in baselines.items()
        }
        print(f"[baseline] constant: {results['baseline']['constant']}")
        print(f"[baseline] random:   {results['baseline']['random']}")

    for name in to_run:
        results[name] = experiments[name]()

    with open(output_dir / "exp_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved results to {output_dir / 'exp_results.json'}")


if __name__ == "__main__":
    main()
