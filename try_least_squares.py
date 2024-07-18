import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares

# Function to read the data
def read_data(file_path):
    data = pd.read_csv(file_path)
    return data['time'].values, data['virusload'].values

# ODE model
def virus_model(y, t, alpha_f, beta, delta_f, gamma):
    f1, f2, V = y
    dXdt = alpha_f * f2 * V - beta * f1 * V
    dYdt = -alpha_f * f2 * V
    dVdt = gamma * f1 * V - delta_f * V
    return [dXdt, dYdt, dVdt]

# Function to solve the ODE
def solve_ode(parameters, initial_conditions, time):
    alpha_f, beta, delta_f, gamma = parameters
    return odeint(virus_model, initial_conditions, time, args=(alpha_f, beta, delta_f, gamma))

# Residuals function for least squares
def residuals(parameters, time, virusload):
    initial_conditions = [0.0047, 0.9953, 3e4]  # f1(0), f2(0), V(0)
    simulated_values = solve_ode(parameters, initial_conditions, time)
    simulated_virusload = simulated_values[:, 2]
    residuals = np.log10(virusload) - np.log10(simulated_virusload)
    
    # Handle non-finite residuals
    if not np.all(np.isfinite(residuals)):
        residuals = np.nan_to_num(residuals, nan=1e6, posinf=1e6, neginf=-1e6)
    
    return residuals

# Function to plot the results
def plot_results(time, virusload, fitted_virusload, save_path):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.plot(time, np.log10(fitted_virusload), '-', label='Fitted Model')
    plt.xlabel('Time')
    plt.ylabel('Log10 Virus Load')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def main():
    # File path to the data
    file_path = '/Users/james/ebola_modelling/data/viral_load.csv'
    
    # Read the data
    time, virusload = read_data(file_path)
    
    # Initial guess for parameters: alpha_f, beta, delta_f, gamma
    initial_guess = [1e-9, 5e-7, 2.5, 2.5e3]  # Adjusted initial guess
    
    # Check initial residuals
    initial_residuals = residuals(initial_guess, time, virusload)
    print(f"Initial residuals: {initial_residuals}")
    
    # Perform least squares fitting
    result = least_squares(residuals, initial_guess, args=(time, virusload), bounds=([1e-12, 1e-12, 0.1, 1], [1e-6, 1e-6, 10, 1e5]))
    
    # Get the fitted parameters
    fitted_parameters = result.x
    print(f"Fitted parameters: {fitted_parameters}")
    
    # Solve the ODE with the fitted parameters
    initial_conditions = [0.0047, 0.9953, 3e4]  # f1(0), f2(0), V(0)
    fitted_values = solve_ode(fitted_parameters, initial_conditions, time)
    fitted_virusload = fitted_values[:, 2]
    
    # Plot the results
    plot_results(time, virusload, fitted_virusload, 'fitted_viral_load.png')

if __name__ == "__main__":
    main()
