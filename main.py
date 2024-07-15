from data.load_parameters import load_parameters
from data.data_preparation import prepare_data
from scripts.parameter_estimation import estimate_parameters
from scripts.run_models import run_model
from utils.plotting import plot_with_confidence_and_threshold, plot_raw_data_log
import numpy as np

def main():
    parameters = load_parameters()
    data = prepare_data()
    print(data.head())  # Debugging: print the first few rows

    time = data['time'].values
    virusload = data['virusload'].values

    valid_idx = ~np.isnan(virusload)
    time = time[valid_idx]
    virusload = virusload[valid_idx]

    print(f"Initial time values: {time}")
    print(f"Initial virusload values: {virusload}")

    # Plot log10 of raw data
    plot_raw_data_log(time, virusload)

    initial_params = parameters['initial_parameters']
    bounds = parameters['bounds']

    chains = estimate_parameters(time, virusload, initial_params, bounds)

    samples = chains[:, 5000:, :].reshape(-1, 4)
    params_median = np.median(samples, axis=0)

    print(f"Estimated parameters median: {params_median}")

    # Extend the time range to 26 days
    extended_time = np.arange(0, 27, 1)
    
    # Simulate for each sample to get the confidence intervals
    all_simulations = []
    for sample in samples:
        simulated = run_model(sample, extended_time)
        all_simulations.append(simulated)
    all_simulations = np.array(all_simulations)
    
    median_simulation = np.median(all_simulations, axis=0)
    lower_bound = np.percentile(all_simulations, 2.5, axis=0)
    upper_bound = np.percentile(all_simulations, 97.5, axis=0)

    print(f"Simulated virusload: {median_simulation[:5]}")  # Print first few values

    # Plot with confidence intervals and threshold
    plot_with_confidence_and_threshold(time, virusload, median_simulation, lower_bound, upper_bound, extended_time, chains, threshold=1e4, isolation_day=21)

if __name__ == "__main__":
    main()
