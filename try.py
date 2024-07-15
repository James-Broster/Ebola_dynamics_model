import pints
import pints.plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Provided data
time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
virusload = np.array([32359.37, 15135612, 67608298, 229086765, 245470892, 398107171, 213796209, 186208714, 23988329, 630957.3, 4265795, 323593.7, 53703.18, np.nan, 141253.8])

# Remove NaN values for fitting
valid_idx = ~np.isnan(virusload)
time = time[valid_idx]
virusload = virusload[valid_idx]

# Differential equations model
def model(y, t, alpha_f, beta, delta_f, gamma):
    X, Y, V = y
    dXdt = alpha_f * Y * V - beta * X * V
    dYdt = -alpha_f * Y * V
    dVdt = gamma * X * V - delta_f * V
    return [dXdt, dYdt, dVdt]

# Initial conditions
X_0 = 0.00670143873431456
Y_0 = 1.0  # Assuming initial Y is 1
V_0 = 36504.3997450364
y0 = [X_0, Y_0, V_0]

# PINTS model wrapper
class VirusModel(pints.ForwardModel):
    def n_parameters(self):
        return 4  # alpha_f, beta, delta_f, gamma

    def simulate(self, parameters, times):
        alpha_f, beta, delta_f, gamma = parameters
        y = odeint(model, y0, times, args=(alpha_f, beta, delta_f, gamma))
        return y[:, 2]  # Return the virus load (V)

# Create PINTS model
virus_model = VirusModel()

# Log-likelihood using GaussianKnownSigmaLogLikelihood
problem = pints.SingleOutputProblem(virus_model, time, virusload)
log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=virusload.std())

# Prior distributions for parameters: Uniform between reasonable bounds
log_prior = pints.UniformLogPrior(
    [1e-12, 1e-12, 0.1, 1],  # Lower bounds for alpha_f, beta, delta_f, gamma
    [1e-6, 1e-6, 10, 10000]  # Upper bounds for alpha_f, beta, delta_f, gamma
)

# Posterior (log_likelihood + log_prior)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)

# Initial parameter guess
initial_parameters = [1.57763931289105e-9, 6.12783508806595e-7, 1.50836617187542, 1386.64900596809]

# MCMC sampling
chains = 3
mcmc = pints.MCMCController(log_posterior, chains, [initial_parameters]*chains)
mcmc.set_max_iterations(10000)
mcmc.set_log_to_screen(True)

# Run!
chains = mcmc.run()

# Plot the MCMC traces
pints.plot.trace(chains)
plt.show()

# Extract samples and plot fit
samples = chains[:, 5000:, :].reshape(-1, 4)  # Remove burn-in

# Use the median of the posterior distribution for the fit
params_median = np.median(samples, axis=0)

# Simulate using the median parameters
simulated = odeint(model, y0, time, args=(params_median[0], params_median[1], params_median[2], params_median[3]))
simulated_virusload = simulated[:, 2]

# Plot data vs model
plt.figure()
plt.plot(time, virusload, 'o', label='Data')
plt.plot(time, simulated_virusload, '-', label='Model')
plt.xlabel('Time')
plt.ylabel('Virus Load')
plt.legend()
plt.show()
