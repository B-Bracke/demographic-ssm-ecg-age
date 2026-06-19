"""Evaluation metrics for age estimation."""

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def calculate_metrics(preds, targets):
    """Compute regression metrics between predicted and target ages.

    Args:
        preds: Predicted ages as np.array of shape [N] (in years).
        targets: Ground-truth ages as np.array of shape [N] (in years).

    Returns:
        Dictionary containing MAE, MAE std, RMSE, Pearson r, R², and
        distribution statistics (mean/std) for predictions and targets.
    """
    errors = np.abs(preds - targets)
    mae = np.mean(errors)
    mae_std = np.std(errors)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)

    r_corr, _ = pearsonr(preds, targets)
    r2 = r2_score(targets, preds)

    return {
        "mae": mae,
        "mae_std": mae_std,
        "rmse": rmse,
        "r": r_corr,
        "r2": r2,
        "pred_mean": np.mean(preds),
        "pred_std": np.std(preds),
        "real_mean": np.mean(targets),
        "real_std": np.std(targets),
    }
