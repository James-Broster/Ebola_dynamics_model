import matplotlib.pyplot as plt
import numpy as np
import pints.plot
from pathlib import Path

def save_plot_raw_data_log(time, virusload, save_path):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_plot_with_confidence_and_threshold(time, virusload, median_simulation, lower_bound, upper_bound, extended_time, chains, threshold, isolation_day, save_path):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.plot(extended_time, np.log10(median_simulation), '-', label='Model Median')
    plt.fill_between(extended_time, np.log10(lower_bound), np.log10(upper_bound), color='gray', alpha=0.5, label='95% CI')
    plt.axhline(y=np.log10(threshold), color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.axvline(x=isolation_day, color='blue', linestyle='--', label='21-day Isolation')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.legend()
    plt.ylim(bottom=0)  # Start y-axis at 0
    plt.savefig(save_path)
    plt.close()

    pints.plot.trace(chains)
    plt.savefig(Path(save_path).parent / 'mcmc_trace.png')
    plt.close()
