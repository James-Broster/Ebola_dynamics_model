import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from data.load_parameters import load_parameters
from data.data_preparation import prepare_data
from scripts.parameter_estimation import estimate_parameters
from scripts.run_models import run_model
from utils.plotting import save_plot_with_confidence_and_threshold, save_plot_raw_data_log, save_plot_f1_f2, plot_isolation_time_difference
import matplotlib.pyplot as plt

def save_results(results_dir, params_median, extended_time, median_simulation, lower_bound, upper_bound, f1_simulations, f2_simulations):
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created directory: {results_dir}')  # Debugging

    # Save the parameters to a CSV file
    params_df = pd.DataFrame([params_median], columns=['alpha_f', 'beta', 'delta_f', 'gamma', 'f1_0', 'f2_0', 'V_0'])
    params_df.to_csv(results_dir / 'estimated_parameters.csv', index=False)
    print(f'Saved parameters to CSV: {results_dir / "estimated_parameters.csv"}')  # Debugging

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
    print(f'Saved simulation results to CSV: {results_dir / "simulation_results.csv"}')  # Debugging

def run_baseline_model(case_type):
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
    results_dir = Path(f'/Users/james/ebola_modelling/results/{case_type}/no_therapy')
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created directory: {results_dir}')  # Debugging

    # Plot log10 of raw data and save
    raw_data_log_path = results_dir / 'raw_data_log.png'
    save_plot_raw_data_log(time, virusload, raw_data_log_path, title=f'Log10 Virus Load - {case_type.capitalize()} - No Therapy')

    # Estimate parameters using MCMC
    chains = estimate_parameters(time, virusload, initial_params, bounds)

    samples = chains[:, 5000:, :].reshape(-1, 7)
    params_median = np.median(samples, axis=0)

    print(f"Estimated parameters median: {params_median}")

    # Extend the time range to 26 days
    extended_time = np.arange(0, 27, 1)

    # Simulate without therapy to get the baseline
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

    # Plot with confidence intervals and threshold and save
    model_plot_path = results_dir / 'viral_load.png'
    save_plot_with_confidence_and_threshold(
        time, virusload, median_simulation, lower_bound, upper_bound, extended_time, 
        chains, threshold=1e4, isolation_day=21, save_path=model_plot_path, title=f'Viral Load - {case_type.capitalize()} - No Therapy'
    )

    # Plot f1 and f2 predictions and save
    f1_plot_path = results_dir / 'f1.png'
    f2_plot_path = results_dir / 'f2.png'
    save_plot_f1_f2(extended_time, f1_median, f1_lower, f1_upper, f2_median, f2_lower, f2_upper, f1_plot_path, f2_plot_path, title_prefix=f'{case_type.capitalize()} - No Therapy')

    # Save results to CSV
    save_results(results_dir, params_median, extended_time, median_simulation, lower_bound, upper_bound, f1_simulations, f2_simulations)

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

    return params_median, samples, (median_simulation, lower_bound, upper_bound), day_reaches_threshold

def run_therapy_simulations(case_type, params, samples, baseline_results, epsilon_values, t_star_values, extended_time, threshold, time, virusload, base_isolation_day):
    initial_conditions = params[4:]  # Initial conditions are at the end of params array
    results = []

    results_dir_base = Path(f'/Users/james/ebola_modelling/results/{case_type}/therapy')
    
    for epsilon in epsilon_values:
        for t_star in t_star_values:
            V_simulations_therapy = []
            for sample in samples:
                simulated = run_model(sample[:4], sample[4:], extended_time, (epsilon, t_star))  # apply therapy parameters
                V_simulations_therapy.append(simulated[:, 2])
            V_simulations_therapy = np.array(V_simulations_therapy)

            median_simulation_therapy = np.median(V_simulations_therapy, axis=0)
            lower_bound_therapy = np.percentile(V_simulations_therapy, 2.5, axis=0)
            upper_bound_therapy = np.percentile(V_simulations_therapy, 97.5, axis=0)

            isolation_day = None
            for day, value in zip(extended_time, median_simulation_therapy):
                if value < threshold:
                    isolation_day = day
                    break

            results.append((epsilon, t_star, isolation_day))

            # Plotting
            baseline_median_simulation, baseline_lower_bound, baseline_upper_bound = baseline_results
            plt.figure()
            plt.plot(time, np.log10(virusload), 'o', label='Data')
            plt.plot(extended_time, np.log10(np.clip(baseline_median_simulation, a_min=1e-10, a_max=None)), '-', label='Model Median', color='orange')
            plt.fill_between(extended_time, np.log10(np.clip(baseline_lower_bound, a_min=1e-10, a_max=None)), np.log10(np.clip(baseline_upper_bound, a_min=1e-10, a_max=None)), color='gray', alpha=0.5, label='95% CI')
            plt.plot(extended_time, np.log10(np.clip(median_simulation_therapy, a_min=1e-10, a_max=None)), '-', label='Model Median with Therapy', color='green')
            plt.fill_between(extended_time, np.log10(np.clip(lower_bound_therapy, a_min=1e-10, a_max=None)), np.log10(np.clip(upper_bound_therapy, a_min=1e-10, a_max=None)), color='lightgreen', alpha=0.5, label='95% CI with Therapy')
            plt.axhline(y=np.log10(threshold), color='r', linestyle='--', label='Threshold')
            plt.axvline(x=t_star, color='blue', linestyle='--', label=f'Therapy start day {t_star}')
            plt.xlabel('Time')
            plt.ylabel('Log10 Virus Load')
            plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
            plt.title(f'{case_type.capitalize()} - Therapy (ε={epsilon}, t*={t_star})')
            plt.legend()
            plot_path = results_dir_base / f'therapy_e{epsilon}_t{t_star}.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved plot: {plot_path}')  # Debugging

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['epsilon', 't_star', 'isolation_day'])
    results_path = results_dir_base / 'therapy_analysis.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f'Saved therapy analysis to CSV: {results_path}')  # Debugging

    # Plot the isolation time difference heatmap
    plot_isolation_time_difference(case_type, base_isolation_day)
    print('Isolation time difference heatmap created')  # Debugging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ebola model simulation.')
    parser.add_argument('case_type', type=str, choices=['fatal', 'non_fatal'], help='Case type: fatal or non_fatal')
    args = parser.parse_args()

    case_type = args.case_type

    # Run baseline model
    params_median, samples, baseline_results, base_isolation_day = run_baseline_model(case_type)

    # Run therapy simulations
    epsilon_values = [0.5, 0.7, 0.9, 1.0]  # Example efficacy values including 1.0
    t_star_values = [1, 3, 5, 7, 10, 13]  # Example therapy initiation times including days 7, 10, and 13
    extended_time = np.arange(0, 27, 1)
    threshold = 1e4  # Viral load threshold for ending isolation
    # Prepare data again to pass to the function for plotting
    dataset_file = 'viral_load.csv' if case_type == 'fatal' else 'non_fatal.csv'
    data = prepare_data(dataset_file)
    time = data['time'].values
    virusload = data['virusload'].values
    filter_idx = time <= 14
    time = time[filter_idx]
    virusload = virusload[filter_idx]
    valid_idx = ~np.isnan(virusload)
    time = time[valid_idx]
    virusload = virusload[valid_idx]
    run_therapy_simulations(case_type, params_median, samples, baseline_results, epsilon_values, t_star_values, extended_time, threshold, time, virusload, base_isolation_day)
