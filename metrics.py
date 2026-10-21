"""Single source of truth for the evaluation metric and the reference baselines.

Diagonal error (used everywhere in the thesis)
----------------------------------------------
For targets t_i and predictions p_i in the normalized screen space [0, 1]^2

    E_diag = 100 * sqrt( 1/N * sum_i ||p_i - t_i||_2^2 ) / sqrt(2)      [%]

i.e. the *Euclidean* RMSE divided by the screen diagonal sqrt(2).

History / bug fix (Oct 2026)
----------------------------
Earlier versions of model.py, model_train.py, notebooks/baseline_personal.py
and the notebooks computed

    rmse = sqrt(sklearn.metrics.mean_squared_error(targets, preds))

For 2-D targets sklearn averages the squared error over BOTH coordinates,
so this is the per-coordinate RMSE = Euclidean RMSE / sqrt(2). Dividing it
by sqrt(2) once more made every model error exactly sqrt(2) too small,
while the constant/random baselines (exp.py) were computed with the correct
Euclidean formula. Old model values can be converted exactly:

    E_diag(correct) = sqrt(2) * E_diag(old)

`legacy_diagonal_error_pct` reproduces the old value for comparison only.
"""

import numpy as np

SQRT2 = np.sqrt(2.0)


def _as_2d(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected shape (N, 2), got {a.shape}")
    return a


def euclidean_errors(targets, preds):
    """Per-sample Euclidean distance ||p_i - t_i|| in normalized space."""
    return np.linalg.norm(_as_2d(preds) - _as_2d(targets), axis=1)


def diagonal_error_pct(targets, preds):
    """Euclidean RMSE as a percentage of the screen diagonal (see module doc)."""
    d = euclidean_errors(targets, preds)
    return float(100.0 * np.sqrt(np.mean(d ** 2)) / SQRT2)


def mean_distance_pct(targets, preds):
    """Mean Euclidean distance as a percentage of the screen diagonal."""
    return float(100.0 * euclidean_errors(targets, preds).mean() / SQRT2)


def legacy_diagonal_error_pct(targets, preds):
    """OLD (incorrect) formula, kept only to reproduce historical logs."""
    t, p = _as_2d(targets), _as_2d(preds)
    return float(100.0 * np.sqrt(np.mean((t - p) ** 2)) / SQRT2)


def constant_baseline_pct(train_targets, test_targets):
    """Predict the mean TRAINING target for every TEST sample.

    Must be evaluated on exactly the same test subset as the model so that the
    baseline line in a learning-curve plot is comparable with the curves.
    """
    train_targets = _as_2d(train_targets)
    test_targets = _as_2d(test_targets)
    pred = np.tile(train_targets.mean(axis=0), (len(test_targets), 1))
    return diagonal_error_pct(test_targets, pred)


def random_baseline_pct(test_targets, repeats=100, seed=0):
    """Uniform random predictions in [0, 1]^2; mean over `repeats` draws."""
    test_targets = _as_2d(test_targets)
    rng = np.random.default_rng(seed)
    vals = [
        diagonal_error_pct(test_targets, rng.uniform(0.0, 1.0, size=test_targets.shape))
        for _ in range(repeats)
    ]
    return float(np.mean(vals))


def subset_baselines(train_targets, test_targets, seed=0):
    """Both reference baselines on the SAME train/test subset as the model."""
    return {
        "constant": constant_baseline_pct(train_targets, test_targets),
        "random": random_baseline_pct(test_targets, seed=seed),
    }
