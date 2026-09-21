"""
Ari_files: Modular gaze-estimation pipeline.

Modules:
- dataset: Data loading, splits, and embedding caching
- model: ResNet18 + regression head, losses, training loop
- train: Single-run CLI entry point
- main: Data preparation (download, reformat, normalize, split)
- exp: Hyperparameter sweep experiments
"""

__version__ = "1.0.0"
__all__ = ["dataset", "model", "train", "main", "exp"]