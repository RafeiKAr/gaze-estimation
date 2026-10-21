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
from torchvision.models import ResNet18_Weights, resnet18

from metrics import diagonal_error_pct, euclidean_errors


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

TRAINABLE_LAYER_MODES = ("fc_only", "fc_layer4", "fc_resnet")

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
    """Freeze the backbone, then unfreeze the head (and optionally layer4).

    fc_only   : head only                      (427,138 trainable parameters)
    fc_layer4 : head + ResNet layer4            (8,820,866)
    fc_resnet : whole network (full fine-tune)  (11,603,650)
    """
    if mode not in TRAINABLE_LAYER_MODES:
        raise ValueError(f"Unknown trainable_layers mode: {mode!r}, expected one of {TRAINABLE_LAYER_MODES}")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if mode == "fc_layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True

    if mode == "fc_resnet":
        for param in model.parameters():
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
def predict(model, loader, device):
    """Return (targets, predictions) as numpy arrays of shape (N, 2)."""
    model.eval()
    targets_list, preds_list = [], []
    for images, targets in loader:
        preds = model(images.to(device))
        targets_list.append(targets.cpu().numpy())
        preds_list.append(preds.cpu().numpy())
    return np.concatenate(targets_list, axis=0), np.concatenate(preds_list, axis=0)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mean_euclidean_distance, euclidean_rmse, diagonal_error_pct).

    diagonal_error_pct = 100 * Euclidean-RMSE / sqrt(2), see metrics.py.
    BUG FIX: the previous version used sqrt(sklearn.mean_squared_error),
    which averages over x and y and therefore was sqrt(2) too small.
    """
    targets, predictions = predict(model, loader, device)
    dist = euclidean_errors(targets, predictions)
    mean_dist = float(dist.mean())
    rmse = float(np.sqrt(np.mean(dist ** 2)))
    return mean_dist, rmse, round(diagonal_error_pct(targets, predictions), 5)


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
    val_loader=None,
):
    """Train with early stopping.

    Epoch 0 is the evaluation of the *untrained* model (before the first
    optimizer step). With the Sigmoid output and a freshly initialised head
    the outputs are approx. (0.5, 0.5), so epoch 0 is close to a constant
    "screen centre" predictor -- this is expected and is why the curves start
    near the constant baseline of the same test subset.

    Model selection:
      * val_loader given (recommended): early stopping and checkpoint
        selection on the validation error; the reported test error is the
        test error at the best validation epoch (no test-set leakage).
      * val_loader None (legacy behaviour, used for all historical logs):
        early stopping on the TEST error and `best_test_error` is the
        minimum test error over epochs 0..T. This is optimistic and, if
        training never improves, simply returns the epoch-0 value.

    Returns a dict with best_test_error, test_error_at_best_val (or None),
    epoch0_test_error, best_epoch, epochs_run, improvements,
    elapsed_minutes, train_errors, test_errors, val_errors.
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

    train_errors, test_errors, val_errors = [], [], []

    def _log_epoch():
        train_errors.append(evaluate(model, train_loader, device)[2])
        test_errors.append(evaluate(model, test_loader, device)[2])
        if val_loader is not None:
            val_errors.append(evaluate(model, val_loader, device)[2])
        return val_errors[-1] if val_loader is not None else test_errors[-1]

    selection_error = _log_epoch()          # epoch 0 = untrained model
    best_error = selection_error
    best_epoch = 0
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

        selection_error = _log_epoch()
        last_epoch = epoch + 1

        if verbose:
            val_txt = f" val_diag={val_errors[-1]:.4f}%" if val_loader is not None else ""
            print(
                f"epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f} "
                f"train_diag={train_errors[-1]:.4f}%{val_txt} test_diag={test_errors[-1]:.4f}%"
            )

        if selection_error < best_error:
            best_error = selection_error
            best_epoch = last_epoch
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
        # legacy key: min test error (val_loader None) or test@best-val
        "best_test_error": test_errors[best_epoch] if val_loader is not None else best_error,
        "test_error_at_best_val": test_errors[best_epoch] if val_loader is not None else None,
        "epoch0_test_error": test_errors[0],
        "best_epoch": best_epoch,
        "epochs_run": last_epoch,
        "improvements": improvements,
        "elapsed_minutes": elapsed_minutes,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "val_errors": val_errors,
    }
