"""Survival analysis: Cox proportional hazards models.

Evaluates the association between ECG-predicted age gap categories
(overestimation, correct, underestimation) and all-cause mortality.

Usage:
    python survival_analysis.py --input_dir results/ --output_dir results/survival/
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from matplotlib.patches import Patch

COLOR_MAPPING = {
    'Overestimation': '#E69F00',
    'Correct Prediction': '#56B4E9',
    'Underestimation': '#009E73',
}
CATEGORIES = ['Underestimation', 'Correct Prediction', 'Overestimation']


def get_clean_data(df, duration_col, event_col):
    """Filter and prepare data for survival analysis.

    Args:
        df: Input DataFrame.
        duration_col: Column with follow-up duration (years).
        event_col: Column with event indicator (0/1).

    Returns:
        Filtered DataFrame with 'time_years' and 'event' columns.
    """
    df = df.copy()
    df['time_years'] = df[duration_col]
    surv_df = df[(df[duration_col].notnull()) & (df[duration_col] > 0)]
    surv_df['event'] = surv_df[event_col].fillna(0).astype(int)
    return surv_df



def run_cox_model(data, duration_col, event_col, title, filename, output_dir):
    """Fit Cox proportional hazards model adjusted for age, sex, and age gap category.

    Args:
        data: Input DataFrame with survival data.
        duration_col: Column with follow-up duration (years).
        event_col: Column with event indicator (0/1).
        title: Title for the output.
        filename: Base filename for saved outputs.
        output_dir: Directory to save results.
    """
    print(f"Running Cox model: {title}")
    df_surv = get_clean_data(data, duration_col, event_col)

    df_surv['age_category'] = df_surv['age_category'].astype('category')
    df_surv['is_male'] = df_surv['is_male'].astype('category')

    model_cols = ['time_years', 'event', 'age_category', 'target_age', 'is_male']
    formula_parts = ['age_category', 'target_age', 'is_male']

    df_model = df_surv[model_cols].dropna()

    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='time_years', event_col='event',
            formula=" + ".join(formula_parts))

    # Save summary
    summary_path = os.path.join(output_dir, f"{filename}_summary.txt")
    with open(summary_path, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"=== Cox Model: {title} ===")
        print(f"N = {len(df_model)}, Events = {df_model['event'].sum()}")
        print("-" * 30)
        cph.print_summary()
        sys.stdout = original_stdout
    print(f"Saved summary to {summary_path}")

    # Plot adjusted survival curves
    fig, ax = plt.subplots(figsize=(10, 8))
    custom_colors = [COLOR_MAPPING[cat] for cat in CATEGORIES]

    try:
        cph.plot_partial_effects_on_outcome(
            covariates='age_category',
            values=CATEGORIES,
            plot_baseline=False,
            color=custom_colors,
            ax=ax,
        )
        plt.title(title)
        plt.xlabel("Time (years)")
        plt.ylabel("Survival Probability")
        ymin = min(ax.get_ylim()[0] - 0.05, 0.9)
        plt.ylim(ymin, 1.0)
        plt.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_adjusted.pdf"))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot partial effects for {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run survival analysis on ECG age predictions')
    parser.add_argument('--input_dir', required=True, help='Directory containing prepared data')
    parser.add_argument('--output_dir', required=True, help='Directory to save survival results')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.input_dir, "patient_median_predicted_chronological_age.csv"))
    df = df.dropna(subset=['target_age', 'ECG_Age'])
    if 'is_calibration_sample' in df.columns:
        df = df[df['is_calibration_sample'] == False]

    # Cox proportional hazards
    run_cox_model(df, 'timey', 'death',
                  "Cox PH Model (Age + Sex + Age Gap)", "cox_pep_basic", args.output_dir)
