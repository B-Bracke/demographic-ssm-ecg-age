"""Prepare inference predictions for downstream evaluation and survival analysis.

Computes per-patient median predictions (when multiple exams exist) and
categorizes patients by age gap (over/under/correct estimation).

Usage:
    python prepare_data.py --input predictions.csv --output_dir results/
"""

import os
import argparse

import pandas as pd


def categorize_delta_age(delta_age, threshold=8):
    """Categorize age gap into over-/under-estimation or correct prediction."""
    if delta_age > threshold:
        return 'Overestimation'
    elif delta_age < -threshold:
        return 'Underestimation'
    return 'Correct Prediction'


def process_patient_data(input_file, output_dir):
    """Load predictions and resolve ECG age column.

    Uses 'corrected_predicted_age' (from calibration) if available,
    otherwise falls back to 'predicted_age'.

    Args:
        input_file: Path to CSV with prediction results.
        output_dir: Directory to save processed output.

    Returns:
        DataFrame with 'ECG_Age' column set.
    """
    print(f"Loading predictions from {input_file}...")
    df = pd.read_csv(input_file)

    if 'corrected_predicted_age' in df.columns:
        df['ECG_Age'] = df['corrected_predicted_age']
    else:
        df['ECG_Age'] = df['predicted_age']

    df = df.dropna(subset=['ECG_Age'])
    print(f"Samples after dropping NaN: {len(df)}")

    raw_output = os.path.join(output_dir, "patient_predicted_chronological_age.csv")
    df.to_csv(raw_output, index=False)
    return df


def create_median_dataset(df, output_dir, age_threshold=8):
    """Aggregate multiple exams per patient using median and categorize age gap.

    Args:
        df: DataFrame with per-exam predictions.
        output_dir: Directory to save the aggregated dataset.
        age_threshold: Threshold (years) for categorizing age predictions.
    """
    print("Creating median dataset per patient...")
    agg_rules = {
        'target_age': 'median',
        'ECG_Age': 'median',
        'predicted_std': 'median',
        'is_male': 'first',
        'death': 'max',
        'timey': 'max',
    }
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    if 'patient_id' in df.columns:
        median_df = df.groupby('patient_id').agg(agg_rules).reset_index()
    else:
        print("Warning: 'patient_id' column not found. Using per-exam data.")
        median_df = df.copy()

    median_df['Delta_age'] = median_df['ECG_Age'] - median_df['target_age']
    median_df['age_category'] = median_df['Delta_age'].apply(
        lambda x: categorize_delta_age(x, age_threshold)
    )

    median_output = os.path.join(output_dir, "patient_median_predicted_chronological_age.csv")
    median_df.to_csv(median_output, index=False)
    print(f"Saved median dataset to {median_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare prediction data for evaluation')
    parser.add_argument('--input', required=True, help='Input CSV with predictions')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--age_threshold', type=int, default=8,
                        help='Threshold (years) for categorizing age gap')
    args = parser.parse_args()

    df = process_patient_data(args.input, args.output_dir)
    create_median_dataset(df, args.output_dir, args.age_threshold)
