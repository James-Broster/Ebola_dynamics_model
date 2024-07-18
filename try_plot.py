import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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

# Function to plot the results
def plot_results(time, virusload, simulated_virusload, save_path):
    plt.figure()
    plt.plot(time, np.log10(virusload), 'o', label='Data')
    plt.plot(time, np.log10(simulated_virusload), '-', label='Model')
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
    
    # Parameters and initial conditions
    parameters = [9.79e-10, 5.1e-7, 2.27, 2e3]  # alpha_f, beta, delta_f, gamma
    initial_conditions = [0.0047, 0.9953, 3e4]  # f1(0), f2(0), V(0)
    
    # Solve the ODE
    simulated_values = solve_ode(parameters, initial_conditions, time)
    simulated_virusload = simulated_values[:, 2]
    
    # Plot the results
    plot_results(time, virusload, simulated_virusload, 'viral_load_simulation.png')

if __name__ == "__main__":
    main()

