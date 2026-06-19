"""Optional post-hoc calibration of predicted ages via linear regression.

Fits a linear mapping from predicted to chronological age on a held-out
calibration subset, then applies the correction to all predictions.
Calibration was not applied during the official MICCAI 2026 submission.

Usage:
    python calibrate_predictions.py --input predictions.csv --output calibrated.csv --samples 512
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calibrate(input_path, output_path, n_samples=512, seed=42, bias_only=False):
    """Calibrate predicted ages using a held-out subset.

    Args:
        input_path: Path to CSV with 'predicted_age' and 'target_age' columns.
        output_path: Path to save calibrated CSV.
        n_samples: Number of samples for calibration fitting.
        seed: Random seed for reproducibility.
        bias_only: If True, apply only a constant shift (no scaling).
    """
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    if len(df) < (n_samples * 2):
        n_actual = int(len(df) * 0.2)
        print(f"Dataset small ({len(df)}). Using {n_actual} samples for calibration.")
    else:
        n_actual = n_samples
        print(f"Using {n_actual} random samples for calibration.")

    rng = np.random.default_rng(seed)
    calib_idx = rng.choice(df.index, size=n_actual, replace=False)

    train_subset = df.loc[calib_idx]
    X_train = train_subset['predicted_age'].values.reshape(-1, 1)
    y_train = train_subset['target_age'].values

    params_path = os.path.join(os.path.dirname(output_path), "calibration_params.txt")

    if not bias_only:
        reg = LinearRegression().fit(X_train, y_train)
        slope, intercept = reg.coef_[0], reg.intercept_
        print(f"Calibration: corrected_age = {slope:.4f} * predicted_age + {intercept:.4f}")

        with open(params_path, 'w') as f:
            f.write(f"corrected_age = {slope:.4f} * predicted_age + {intercept:.4f}\n")
            f.write(f"Calibration samples: {n_actual}\n")

        df['corrected_predicted_age'] = reg.predict(df['predicted_age'].values.reshape(-1, 1))
    else:
        shift = np.mean(y_train) - np.mean(X_train)
        print(f"Bias correction: corrected_age = predicted_age + {shift:.4f}")

        with open(params_path, 'w') as f:
            f.write(f"corrected_age = predicted_age + {shift:.4f}\n")
            f.write(f"Calibration samples: {n_actual}\n")

        df['corrected_predicted_age'] = df['predicted_age'] + shift

    df['is_calibration_sample'] = False
    df.loc[calib_idx, 'is_calibration_sample'] = True

    # Report improvement on non-calibration samples
    test_subset = df.drop(calib_idx)
    if not test_subset.empty:
        orig_mae = np.mean(np.abs(test_subset['predicted_age'] - test_subset['target_age']))
        new_mae = np.mean(np.abs(test_subset['corrected_predicted_age'] - test_subset['target_age']))
        print(f"Test MAE: {orig_mae:.2f} -> {new_mae:.2f} (delta: {orig_mae - new_mae:+.2f})")

    df.to_csv(output_path, index=False)
    print(f"Saved calibrated predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional post-hoc calibration of age predictions')
    parser.add_argument('--input', required=True, help='Input CSV with predicted_age and target_age')
    parser.add_argument('--output', required=True, help='Output path for calibrated CSV')
    parser.add_argument('--samples', type=int, default=512, help='Number of calibration samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_only', action='store_true', help='Apply only a constant shift (no scaling)')
    args = parser.parse_args()

    calibrate(args.input, args.output, args.samples, args.seed, args.bias_only)
