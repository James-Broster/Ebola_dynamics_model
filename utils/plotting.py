import matplotlib.pyplot as plt
import numpy as np
import pints.plot
from pathlib import Path

def save_plot_raw_data_log(time, virusload, save_path, title):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_plot_with_confidence_and_threshold(time, virusload, median_simulation, lower_bound, upper_bound, extended_time, chains, threshold, isolation_day, save_path, title, therapy_data=None):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.plot(extended_time, np.log10(median_simulation), '-', label='Model Median')
    plt.fill_between(extended_time, np.log10(np.clip(lower_bound, a_min=1e-10, a_max=None)), np.log10(np.clip(upper_bound, a_min=1e-10, a_max=None)), color='gray', alpha=0.5, label='95% CI')
    plt.axhline(y=np.log10(threshold), color='r', linestyle='--', label=f'Threshold')
    plt.axvline(x=isolation_day, color='blue', linestyle='--', label='21-day Isolation')
    if therapy_data:
        median_simulation_therapy, lower_bound_therapy, upper_bound_therapy = therapy_data
        if median_simulation_therapy is not None:
            plt.plot(extended_time, np.log10(median_simulation_therapy), '-', label='Model Median with Therapy', color='green')
            plt.fill_between(extended_time, np.log10(np.clip(lower_bound_therapy, a_min=1e-10, a_max=None)), np.log10(np.clip(upper_bound_therapy, a_min=1e-10, a_max=None)), color='lightgreen', alpha=0.5, label='95% CI with Therapy')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.title(title)
    plt.legend()
    plt.ylim(bottom=0)  # Start y-axis at 0
    plt.savefig(save_path)
    plt.close()

    pints.plot.trace(chains)
    plt.savefig(Path(save_path).parent / 'mcmc_trace.png')
    plt.close()

def save_plot_f1_f2(extended_time, f1_median, f1_lower, f1_upper, f2_median, f2_lower, f2_upper, save_path_f1, save_path_f2, title_prefix):
    plt.figure()
    plt.plot(extended_time, f1_median, '-', label='f1 Median')
    plt.fill_between(extended_time, f1_lower, f1_upper, color='gray', alpha=0.5, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('f1')
    plt.title(f'{title_prefix} - f1')
    plt.legend()
    plt.savefig(save_path_f1)
    plt.close()
    
    plt.figure()
    plt.plot(extended_time, f2_median, '-', label='f2 Median')
    plt.fill_between(extended_time, f2_lower, f2_upper, color='gray', alpha=0.5, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('f2')
    plt.title(f'{title_prefix} - f2')
    plt.legend()
    plt.savefig(save_path_f2)
    plt.close()
