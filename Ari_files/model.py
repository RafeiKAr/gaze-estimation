"""Model definition, training and evaluation for gaze-estimation.

Reproduces the ResNet18-based regression head, activation-function and
loss-function options, and the early-stopping training loop used in
Compare_notebook.ipynb, as reusable functions.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchvision.models import ResNet18_Weights, resnet18


class GaussianActivation(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x ** 2) / (2 * self.sigma ** 2))


ACTIVATIONS = {
    "Sigmoid": nn.Sigmoid,
    "Gaussian": lambda: GaussianActivation(sigma=1.0),
    "ReLU": nn.ReLU,
    "None": nn.Identity,
}

TRAINABLE_LAYER_MODES = ("fc_only", "fc_layer4")

# Dimensionality of the ResNet18 feature vector (its avgpool+flatten output,
# i.e. what a frozen backbone's forward pass produces once model.fc is
# replaced by nn.Identity()).
BACKBONE_EMBEDDING_DIM = 512


def build_head(activation="Sigmoid"):
    """Build the regression head alone: BACKBONE_EMBEDDING_DIM -> (x, y).

    This is exactly what `build_model(...).fc` is, exposed separately so it
    can be trained directly on precomputed backbone embeddings when the
    backbone is fully frozen (see dataset.build_embedding_dataloaders).
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {activation!r}, expected one of {list(ACTIVATIONS)}")

    return nn.Sequential(
        nn.Linear(BACKBONE_EMBEDDING_DIM, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        ACTIVATIONS[activation](),
    )


def build_model(activation="Sigmoid"):
    """Build a ResNet18 backbone with a 4-layer regression head -> (x, y)."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_name = model.__class__.__name__
    model.fc = build_head(activation)
    return model, model_name


def build_frozen_backbone():
    """A ResNet18 backbone (no head) with every parameter frozen.

    Its forward pass returns the BACKBONE_EMBEDDING_DIM-dim feature vector
    that would otherwise be fed into the head on every epoch. Only valid
    to reuse across training runs when trainable_layers == "fc_only", since
    layer4 must also stay frozen for cached embeddings to remain correct.
    """
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def set_trainable_layers(model, mode):
    """Freeze the backbone, then unfreeze the head (and optionally layer4)."""
    if mode not in TRAINABLE_LAYER_MODES:
        raise ValueError(f"Unknown trainable_layers mode: {mode!r}, expected one of {TRAINABLE_LAYER_MODES}")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if mode == "fc_layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True


def sensitive_loss(preds, targets):
    """L1 loss weighted by each target's distance from the screen center."""
    per_sample_l1 = torch.abs(preds - targets).mean(dim=1)
    distance_from_center = torch.abs(targets[:, 0] - 0.5) + torch.abs(targets[:, 1] - 0.5)
    weight = 1.0 + distance_from_center
    return (per_sample_l1 * weight).mean()


LOSS_FUNCTIONS = {
    "standard": nn.L1Loss(),
    "sensitive": sensitive_loss,
}


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mae, rmse, diagonal_error_pct) of model on loader.

    diagonal_error_pct is the RMSE expressed as a percentage of the (0,1)
    normalized screen diagonal, i.e. 100 * rmse / sqrt(2).
    """
    model.eval()

    targets_list, preds_list = [], []
    for images, targets in loader:
        images = images.to(device)
        preds = model(images)
        targets_list.append(targets.cpu().numpy())
        preds_list.append(preds.cpu().numpy())

    targets = np.concatenate(targets_list, axis=0)
    predictions = np.concatenate(preds_list, axis=0)

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    diagonal_error_pct = np.round(100 * (rmse / np.sqrt(2)), 5)

    return mae, rmse, diagonal_error_pct


def confidence_interval(data, confidence=0.95):
    """Mean and two-sided t-distribution confidence interval of data."""
    data = np.asarray(data)
    n = len(data)

    mean = np.mean(data)
    standard_error = np.std(data, ddof=1) / np.sqrt(n)
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_value * standard_error

    return mean, mean - margin, mean + margin


def train_model(
    model,
    model_name,
    train_loader,
    test_loader,
    device,
    epochs=500,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=3,
    trainable_layers="fc_only",
    loss_name="standard",
    model_dir=None,
    verbose=True,
):
    """Train with early stopping on the test diagonal error.

    Mirrors the notebook's training loop: freezes the backbone according to
    `trainable_layers`, optimizes with AdamW, and stops once `patience`
    consecutive epochs pass without a new best test diagonal error.

    `model` need not be a full ResNet: pass `trainable_layers=None` to train
    a standalone head (from model.build_head) on precomputed embeddings
    (from dataset.build_embedding_dataloaders) instead of freezing/unfreezing
    parts of a ResNet -- every parameter of `model` is then trained as-is.

    Returns a dict with best_test_error, epochs_run, improvements,
    elapsed_minutes, train_errors and test_errors (one entry per epoch,
    epoch 0 being the pre-training evaluation).
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss_name: {loss_name!r}, expected one of {list(LOSS_FUNCTIONS)}")

    if trainable_layers is None:
        for param in model.parameters():
            param.requires_grad = True
    else:
        set_trainable_layers(model, trainable_layers)
    model.to(device)

    criterion = LOSS_FUNCTIONS[loss_name]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()

    train_errors, test_errors = [], []
    _, _, train_diag = evaluate(model, train_loader, device)
    _, _, test_diag = evaluate(model, test_loader, device)
    train_errors.append(train_diag)
    test_errors.append(test_diag)

    best_error = test_diag
    epochs_without_improvement = 0
    improvements = 1
    last_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        _, _, train_diag = evaluate(model, train_loader, device)
        _, _, test_diag = evaluate(model, test_loader, device)
        train_errors.append(train_diag)
        test_errors.append(test_diag)
        last_epoch = epoch + 1

        if verbose:
            print(
                f"epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f} "
                f"train_diag={train_diag:.4f}% test_diag={test_diag:.4f}%"
            )

        if test_diag < best_error:
            best_error = test_diag
            epochs_without_improvement = 0
            improvements += 1
            if model_dir is not None:
                torch.save(model.state_dict(), model_dir / f"best_{model_name}.path")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping after {epoch + 1} epochs.")
                break

        if model_dir is not None:
            torch.save(model.state_dict(), model_dir / f"last_{model_name}.path")

    elapsed_minutes = (time.perf_counter() - train_start) / 60

    return {
        "best_test_error": best_error,
        "epochs_run": last_epoch,
        "improvements": improvements,
        "elapsed_minutes": elapsed_minutes,
        "train_errors": train_errors,
        "test_errors": test_errors,
    }