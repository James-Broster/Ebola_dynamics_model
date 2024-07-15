import pints
import numpy as np
from models.base_model import solve_ode

def estimate_parameters(time, virusload, initial_parameters, bounds):
    class VirusModel(pints.ForwardModel):
        def n_parameters(self):
            return 4

        def simulate(self, parameters, times):
            y0 = [0.00670143873431456, 1.0, 36504.3997450364]
            result = solve_ode(parameters, y0, times)
            return result[:, 2]

    virus_model = VirusModel()

    problem = pints.SingleOutputProblem(virus_model, time, virusload)
    log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=virusload.std())

    log_prior = pints.UniformLogPrior(
        [bounds['alpha_f'][0], bounds['beta'][0], bounds['delta_f'][0], bounds['gamma'][0]],
        [bounds['alpha_f'][1], bounds['beta'][1], bounds['delta_f'][1], bounds['gamma'][1]]
    )

    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    initial_params = [initial_parameters['alpha_f'], initial_parameters['beta'], initial_parameters['delta_f'], initial_parameters['gamma']]

    chains = 3
    mcmc = pints.MCMCController(log_posterior, chains, [initial_params]*chains)
    
    # Set maximum iterations
    mcmc.set_max_iterations(50000)  # Increase the number of iterations, e.g., to 50000
    mcmc.set_log_to_screen(True)
    
    chains = mcmc.run()

    return chains
