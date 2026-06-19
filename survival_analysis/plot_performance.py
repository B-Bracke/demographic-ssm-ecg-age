"""Generate performance visualizations for ECG age predictions.

Produces scatter plots, Bland-Altman plots, delta-age histograms,
and optional uncertainty plots, stratified by sex and age category.

Usage:
    python plot_performance.py --input_dir results/ --output_dir results/plots/
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

COLOR_MAPPING = {
    'Overestimation': '#E69F00',
    'Correct Prediction': '#56B4E9',
    'Underestimation': '#009E73',
}


def plot_delta_age_histogram(data, filename, output_dir):
    """Plot histogram of age prediction errors with normality test overlay."""
    print("Generating delta-age histogram...")
    delta_age = data['Delta_age'].dropna()

    shapiro_stat, shapiro_p = stats.shapiro(delta_age)
    print(f"  Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4e}")

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(delta_age, bins=30, edgecolor='black', alpha=0.7, label='Delta Age')

    mean, std = np.mean(delta_age), np.std(delta_age)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    bin_width = bins[1] - bins[0]
    scaled_pdf = stats.norm.pdf(x, mean, std) * len(delta_age) * bin_width
    plt.plot(x, scaled_pdf, 'k', linewidth=2, label='Normal Distribution')

    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    text_x = mean + (std * 0.5)
    text_y = plt.ylim()[1] * 0.9
    plt.text(text_x, text_y, f'Mean: {mean:.2f}\nStd: {std:.2f}\nShapiro p: {shapiro_p:.1e}',
             color='red', bbox=dict(facecolor='white', alpha=0.8))

    plt.title('Histogram of Delta Age (ECG Age - Chronological Age)')
    plt.xlabel('Delta Age (Years)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    save_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scatter_jointplot(data, filename, title, output_dir):
    """Plot predicted vs chronological age scatter with marginal histograms."""
    print(f"Generating scatter plot: {title}")
    jplot = sns.JointGrid(x='target_age', y='ECG_Age', data=data, space=0)
    sns.scatterplot(x='target_age', y='ECG_Age', hue='age_category',
                    data=data, ax=jplot.ax_joint, palette=COLOR_MAPPING,
                    s=15, alpha=0.2, legend=False)

    jplot.ax_joint.plot([18, 100], [18, 100], color='black', linestyle='-', linewidth=1)
    jplot.ax_joint.set_xlim(18, 100)
    jplot.ax_joint.set_ylim(18, 100)

    bins = np.arange(18, 102, 2)
    sns.histplot(data['target_age'], ax=jplot.ax_marg_x, bins=bins, color='gray', kde=False)
    jplot.ax_marg_y.hist(data['ECG_Age'], bins=bins, color='gray', orientation='horizontal',
                         edgecolor='black', alpha=0.8)

    # Marginal statistics for chronological age
    mean_x, std_x = data['target_age'].mean(), data['target_age'].std()
    jplot.ax_marg_x.axvline(mean_x, color='blue', linestyle='--', linewidth=1)
    jplot.ax_marg_x.axvline(mean_x + std_x, color='blue', linestyle=':', linewidth=1)
    jplot.ax_marg_x.axvline(mean_x - std_x, color='blue', linestyle=':', linewidth=1)
    jplot.ax_marg_x.text(mean_x, jplot.ax_marg_x.get_ylim()[1], f'Mean: {mean_x:.2f}\nStd: {std_x:.2f}',
                         fontsize=10, ha='center', va='bottom', color='blue')

    # Marginal statistics for ECG age
    mean_y, std_y = data['ECG_Age'].mean(), data['ECG_Age'].std()
    jplot.ax_marg_y.axhline(mean_y, color='blue', linestyle='--', linewidth=1)
    jplot.ax_marg_y.axhline(mean_y + std_y, color='blue', linestyle=':', linewidth=1)
    jplot.ax_marg_y.axhline(mean_y - std_y, color='blue', linestyle=':', linewidth=1)
    jplot.ax_marg_y.text(jplot.ax_marg_y.get_xlim()[1], mean_y, f'Mean: {mean_y:.2f}\nStd: {std_y:.2f}',
                         fontsize=10, ha='left', va='center', color='blue', rotation=270)

    r, _ = stats.pearsonr(data['target_age'], data['ECG_Age'])
    mae = np.mean(np.abs(data['ECG_Age'] - data['target_age']))
    mae_std = np.std(np.abs(data['ECG_Age'] - data['target_age']))
    r2 = 1 - (np.sum((data['ECG_Age'] - data['target_age'])**2) /
              np.sum((data['target_age'] - np.mean(data['target_age']))**2))

    stats_text = f'r = {r:.2f}\nr² = {r2:.2f}\nMAE = {mae:.2f}±{mae_std:.2f}'
    jplot.ax_joint.text(0.95, 0.05, stats_text, transform=jplot.ax_joint.transAxes,
                        ha='right', va='bottom', fontsize=12)

    if title:
        jplot.fig.suptitle(title, fontsize=14)
        jplot.fig.subplots_adjust(top=0.90)

    jplot.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close()


def plot_bland_altman(data, filename, title, output_dir):
    """Generate Bland-Altman agreement plot."""
    print(f"Generating Bland-Altman plot: {title}")
    data = data.copy()
    data['Mean_Age'] = (data['target_age'] + data['ECG_Age']) / 2
    data['Diff_Age'] = data['ECG_Age'] - data['target_age']

    mean_diff = data['Diff_Age'].mean()
    std_diff = data['Diff_Age'].std()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Mean_Age', y='Diff_Age', data=data, s=15, alpha=0.5)
    plt.axhline(mean_diff, color='blue', linestyle='--', label='Mean Diff')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='±1.96 SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--')
    plt.xlabel('Mean Age')
    plt.ylabel('Difference (ECG Age - Chronological Age)')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{filename}_bland_altman.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate performance plots for ECG age predictions')
    parser.add_argument('--input_dir', required=True, help='Directory containing prepared data')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.input_dir, "patient_median_predicted_chronological_age.csv"))
    df = df.dropna(subset=['target_age', 'ECG_Age'])

    if 'is_calibration_sample' in df.columns:
        df = df[df['is_calibration_sample'] == False]

    plot_delta_age_histogram(df, "delta_age_histogram", args.output_dir)
    plot_scatter_jointplot(df, "scatter_all", "Predicted vs Chronological Age (All)", args.output_dir)
    plot_scatter_jointplot(df[df['is_male'] == True], "scatter_male", "Male Patients", args.output_dir)
    plot_scatter_jointplot(df[df['is_male'] == False], "scatter_female", "Female Patients", args.output_dir)
    plot_bland_altman(df, "ba_all", "Bland-Altman (All)", args.output_dir)