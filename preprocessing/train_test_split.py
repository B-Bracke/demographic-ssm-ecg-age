"""Patient-level age-stratified train/test split for ECG datasets.

Splits data at the patient level (not per exam) to prevent data leakage,
with stratification by age to preserve the age distribution across splits.

Usage:
    python train_test_split.py --data /path/to/exams.csv \
                               --test_size 0.05 \
                               --seed 42
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_split(df, test_size=0.15, min_age=16, max_age=85, seed=42):
    """Split data into train/test sets by patient with age stratification.

    Args:
        df: DataFrame with columns 'id_patient' and 'age'.
        test_size: Fraction of patients for the test set.
        min_age: Minimum age for stratified splitting.
        max_age: Maximum age for stratified splitting.
        seed: Random seed for reproducibility.

    Returns:
        train_df: Training DataFrame (all exams for training patients).
        test_df: Test DataFrame (all exams for test patients).
    """
    unique_patients = df.groupby('id_patient', as_index=False).sample(1, random_state=seed)

    mask_valid = unique_patients['age'].between(min_age, max_age)
    valid_patients = unique_patients[mask_valid]
    outlier_patients = unique_patients[~mask_valid]

    train_pats, test_pats = train_test_split(
        valid_patients,
        test_size=test_size,
        stratify=valid_patients['age'],
        random_state=seed,
    )

    # Patients outside [min_age, max_age] are assigned to training only
    train_pats = pd.concat([train_pats, outlier_patients])

    train_df = df[df['id_patient'].isin(train_pats['id_patient'])].copy()
    test_df = df[df['id_patient'].isin(test_pats['id_patient'])].copy()

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patient-level age-stratified train/test split')
    parser.add_argument('--data', required=True, help='Path to CSV file with exam metadata')
    parser.add_argument('--test_size', type=float, default=0.05, help='Fraction of patients for test set')
    parser.add_argument('--min_age', type=int, default=16, help='Minimum age for stratified splitting')
    parser.add_argument('--max_age', type=int, default=85, help='Maximum age for stratified splitting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train, test = get_train_test_split(df, test_size=args.test_size,
                                       min_age=args.min_age, max_age=args.max_age,
                                       seed=args.seed)

    print(f"Total exams: {len(df)}")
    print(f"Train exams: {len(train)} | Unique patients: {train['id_patient'].nunique()}")
    print(f"Test exams:  {len(test)}  | Unique patients: {test['id_patient'].nunique()}")

    # Save individual splits
    train_path = args.data.replace(".csv", "_train.csv")
    test_path = args.data.replace(".csv", "_valid.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Saved: {train_path}, {test_path}")

    # Save combined CSV (validation first, then train) for train.py compatibility
    combined_df = pd.concat([test, train]).reset_index(drop=True)
    combined_path = args.data.replace(".csv", "_train_valid_combined.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined CSV saved: {combined_path} (split index: {len(test)})")
