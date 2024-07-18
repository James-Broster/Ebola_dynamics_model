import argparse
import os
import pandas as pd
from data.load_parameters import load_parameters
from data.data_preparation import prepare_data
from scripts.parameter_estimation import estimate_parameters
from scripts.run_models import run_model
from utils.plotting import save_plot_with_confidence_and_threshold, save_plot_raw_data_log, save_plot_f1_f2
import numpy as np
from pathlib import Path

def save_results(case_type, params_median, extended_time, median_simulation, lower_bound, upper_bound, f1_simulations, f2_simulations):
    results_dir = Path(f'/Users/james/ebola_modelling/results/{case_type}')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the parameters to a CSV file
    params_df = pd.DataFrame([params_median], columns=['alpha_f', 'beta', 'delta_f', 'gamma', 'f1_0', 'f2_0', 'V_0'])
    params_df.to_csv(results_dir / 'estimated_parameters.csv', index=False)

    # Save the simulation results to a CSV file
    simulations_df = pd.DataFrame({
        'time': extended_time,
        'median_virus_load': median_simulation,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'f1': np.mean(f1_simulations, axis=0),  # Average f1 across simulations
        'f2': np.mean(f2_simulations, axis=0)   # Average f2 across simulations
    })
    simulations_df.to_csv(results_dir / 'simulation_results.csv', index=False)

def main(case_type):
    dataset_file = 'viral_load.csv' if case_type == 'fatal' else 'non_fatal.csv'
    
    initial_params, bounds = load_parameters(case_type)
    data = prepare_data(dataset_file)
    print(data.head())  # Debugging: print the first few rows

    time = data['time'].values
    virusload = data['virusload'].values

    # Filter data to include only up until day 14
    filter_idx = time <= 14
    time = time[filter_idx]
    virusload = virusload[filter_idx]

    # Remove NaN values
    valid_idx = ~np.isnan(virusload)
    time = time[valid_idx]
    virusload = virusload[valid_idx]

    print(f"Filtered time values: {time}")
    print(f"Filtered virusload values: {virusload}")

    # Create results directory
    results_dir = Path(f'/Users/james/ebola_modelling/results/{case_type}')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot log10 of raw data and save
    raw_data_log_path = results_dir / f'raw_data_log_{case_type}.png'
    save_plot_raw_data_log(time, virusload, raw_data_log_path)

    chains = estimate_parameters(time, virusload, initial_params, bounds)

    samples = chains[:, 5000:, :].reshape(-1, 7)
    params_median = np.median(samples, axis=0)

    print(f"Estimated parameters median: {params_median}")

    # Extend the time range to 26 days
    extended_time = np.arange(0, 27, 1)
    
    # Simulate for each sample to get the confidence intervals
    f1_simulations = []
    f2_simulations = []
    V_simulations = []
    for sample in samples:
        simulated = run_model(sample[:4], sample[4:], extended_time)  # separate parameters and initial conditions
        f1_simulations.append(simulated[:, 0])
        f2_simulations.append(simulated[:, 1])
        V_simulations.append(simulated[:, 2])
    f1_simulations = np.array(f1_simulations)
    f2_simulations = np.array(f2_simulations)
    V_simulations = np.array(V_simulations)
    
    median_simulation = np.median(V_simulations, axis=0)
    lower_bound = np.percentile(V_simulations, 2.5, axis=0)
    upper_bound = np.percentile(V_simulations, 97.5, axis=0)

    f1_median = np.median(f1_simulations, axis=0)
    f1_lower = np.percentile(f1_simulations, 2.5, axis=0)
    f1_upper = np.percentile(f1_simulations, 97.5, axis=0)

    f2_median = np.median(f2_simulations, axis=0)
    f2_lower = np.percentile(f2_simulations, 2.5, axis=0)
    f2_upper = np.percentile(f2_simulations, 97.5, axis=0)

    print(f"Simulated virusload: {median_simulation[:5]}")  # Print first few values

    # Plot with confidence intervals and threshold and save
    model_plot_path = results_dir / f'viral_load_{case_type}.png'
    save_plot_with_confidence_and_threshold(
        time, virusload, median_simulation, lower_bound, upper_bound, extended_time, 
        chains, threshold=1e4, isolation_day=21, save_path=model_plot_path
    )

    # Plot f1 and f2 predictions and save
    f1_plot_path = results_dir / f'f1_{case_type}.png'
    f2_plot_path = results_dir / f'f2_{case_type}.png'
    save_plot_f1_f2(extended_time, f1_median, f1_lower, f1_upper, f2_median, f2_lower, f2_upper, f1_plot_path, f2_plot_path)

    # Save results to CSV
    save_results(case_type, params_median, extended_time, median_simulation, lower_bound, upper_bound, f1_simulations, f2_simulations)

    # Find the day when the model reaches the threshold
    threshold = 1e4
    day_reaches_threshold = None
    for day, value in zip(extended_time, median_simulation):
        if value < threshold:
            day_reaches_threshold = day
            break

    if day_reaches_threshold is not None:
        print(f"The model reaches the threshold on day: {day_reaches_threshold}")
        difference_from_isolation = day_reaches_threshold - 21
        print(f"Difference from the 21-day isolation period: {difference_from_isolation} days")
    else:
        print("The model does not reach the threshold within the extended time period.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ebola model simulation.')
    parser.add_argument('case_type', type=str, choices=['fatal', 'non_fatal'], help='Case type: fatal or non_fatal')
    args = parser.parse_args()

    case_type = args.case_type
    main(case_type)
