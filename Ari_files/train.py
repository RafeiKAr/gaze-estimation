"""CLI entry point for a single training run.

Example:
    python train.py --split-type norm_random --train-size 1000 --test-size 200 \
        --batch-size 32 --learning-rate 1e-4 --activation Sigmoid --loss standard

Run `python train.py --help` for the full list of options.
"""

import argparse
import json

import torch

from dataset import build_dataloaders, build_embedding_dataloaders
from model import build_frozen_backbone, build_head, build_model, evaluate, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the gaze-estimation model.")

    parser.add_argument("--split-type", choices=["norm_subject", "norm_random"], default="norm_subject")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--activation", choices=["Sigmoid", "Gaussian", "ReLU", "None"], default="Sigmoid")
    parser.add_argument("--trainable-layers", choices=["fc_only", "fc_layer4"], default="fc_only")
    parser.add_argument("--loss", choices=["standard", "sensitive"], default="standard")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--model-dir", type=str, default="./models/best_models")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch logging.")

    parser.add_argument(
        "--use-embeddings", action="store_true",
        help=(
            "Only valid with --trainable-layers fc_only: precompute the frozen "
            "ResNet backbone's embeddings once and train only the head on them, "
            "skipping the backbone forward/backward pass every epoch."
        ),
    )
    parser.add_argument(
        "--embedding-cache-dir", type=str, default="./cache/embeddings",
        help="Where --use-embeddings caches precomputed embeddings, reused across runs with the same split/size.",
    )

    args = parser.parse_args()
    if args.use_embeddings and args.trainable_layers != "fc_only":
        parser.error("--use-embeddings requires --trainable-layers fc_only (the backbone must stay frozen).")
    return args


def run(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_embeddings:
        backbone = build_frozen_backbone()
        train_loader, test_loader = build_embedding_dataloaders(
            split_type=args.split_type,
            train_size=args.train_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            backbone=backbone,
            device=device,
            cache_dir=args.embedding_cache_dir,
        )
        model, model_name = build_head(activation=args.activation), "head_only"
        trainable_layers = None
    else:
        train_loader, test_loader = build_dataloaders(
            split_type=args.split_type,
            train_size=args.train_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model, model_name = build_model(activation=args.activation)
        trainable_layers = args.trainable_layers

    result = train_model(
        model,
        model_name,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        trainable_layers=trainable_layers,
        loss_name=args.loss,
        model_dir=args.model_dir,
        verbose=not args.quiet,
    )

    mae, rmse, diag_pct = evaluate(model, test_loader, device)
    result["final_test_mae"] = float(mae)
    result["final_test_rmse"] = float(rmse)
    result["final_test_diag_pct"] = float(diag_pct)

    print(json.dumps({k: v for k, v in result.items() if k not in ("train_errors", "test_errors")}, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())