import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_plot_with_confidence_and_threshold(time, virusload, median_simulation, lower_bound, upper_bound, extended_time, chains, threshold, isolation_day, save_path, title):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.plot(extended_time, np.log10(np.clip(median_simulation, a_min=1e-10, a_max=None)), '-', label='Model Median')
    plt.fill_between(extended_time, np.log10(np.clip(lower_bound, a_min=1e-10, a_max=None)), np.log10(np.clip(upper_bound, a_min=1e-10, a_max=None)), color='gray', alpha=0.5, label='95% CI')
    plt.axhline(y=np.log10(threshold), color='r', linestyle='--', label='Threshold')
    plt.axvline(x=isolation_day, color='blue', linestyle='--', label='21-day Isolation')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.legend()
    plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f'Saved plot: {save_path}')  # Debugging

def save_plot_raw_data_log(time, virusload, save_path, title):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.legend()
    plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f'Saved plot: {save_path}')  # Debugging

def save_plot_f1_f2(extended_time, f1_median, f1_lower, f1_upper, f2_median, f2_lower, f2_upper, f1_plot_path, f2_plot_path, title_prefix):
    plt.figure()
    plt.plot(extended_time, f1_median, '-', label='Median f1')
    plt.fill_between(extended_time, f1_lower, f1_upper, color='gray', alpha=0.5, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('f1')
    plt.legend()
    plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
    plt.title(f'{title_prefix} - f1')
    plt.savefig(f1_plot_path)
    plt.close()
    print(f'Saved plot: {f1_plot_path}')  # Debugging

    plt.figure()
    plt.plot(extended_time, f2_median, '-', label='Median f2')
    plt.fill_between(extended_time, f2_lower, f2_upper, color='gray', alpha=0.5, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('f2')
    plt.legend()
    plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
    plt.title(f'{title_prefix} - f2')
    plt.savefig(f2_plot_path)
    plt.close()
    print(f'Saved plot: {f2_plot_path}')  # Debugging

def plot_isolation_time_difference(case_type, base_isolation_day):
    results_dir_base = Path(f'/Users/james/ebola_modelling/results/{case_type}/therapy')
    results_path = results_dir_base / 'therapy_analysis.csv'

    # Load the CSV data
    df = pd.read_csv(results_path)
    print(f'Loaded CSV data from {results_path}')  # Debugging

    # Calculate the isolation time difference
    df['isolation_time_difference'] = base_isolation_day - df['isolation_day']

    # Pivot the data to create a matrix for the heatmap
    isolation_matrix = df.pivot_table(values='isolation_time_difference', index='t_star', columns='epsilon', aggfunc=np.mean)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(isolation_matrix, annot=True, cmap='viridis', cbar_kws={'label': 'Isolation Time Difference (days)'})
    plt.title(f'{case_type.capitalize()} - Isolation Time Difference Heatmap')
    plt.xlabel('Epsilon (Therapy Efficacy)')
    plt.ylabel('t_star (Therapy Start Day)')
    heatmap_path = results_dir_base / 'isolation_time_difference_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    print(f'Saved heatmap: {heatmap_path}')  # Debugging
