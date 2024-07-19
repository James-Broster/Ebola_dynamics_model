import pints
import numpy as np
from models.base_model import solve_ode

def estimate_parameters(time, virusload, initial_parameters, bounds):
    class VirusModel(pints.ForwardModel):
        def n_parameters(self):
            return 7  # 4 parameters + 3 initial conditions

        def simulate(self, parameters, times):
            alpha_f, beta, delta_f, gamma, f1_0, f2_0, V_0 = parameters
            y0 = [f1_0, f2_0, V_0]
            result = solve_ode((alpha_f, beta, delta_f, gamma), y0, times)
            return result[:, 2]  # return the virus load (V)

    virus_model = VirusModel()

    problem = pints.SingleOutputProblem(virus_model, time, virusload)
    log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=virusload.std())

    log_prior = pints.UniformLogPrior(
        [
            bounds['alpha_f'][0], bounds['beta'][0], bounds['delta_f'][0], bounds['gamma'][0], 
            bounds['f1_0'][0], bounds['f2_0'][0], bounds['V_0'][0]
        ],
        [
            bounds['alpha_f'][1], bounds['beta'][1], bounds['delta_f'][1], bounds['gamma'][1], 
            bounds['f1_0'][1], bounds['f2_0'][1], bounds['V_0'][1]
        ]
    )

    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    initial_params = [
        initial_parameters['alpha_f'], initial_parameters['beta'], initial_parameters['delta_f'], initial_parameters['gamma'],
        initial_parameters['f1_0'], initial_parameters['f2_0'], initial_parameters['V_0']
    ]

    # Print initial logpdf values
    initial_logpdf = log_posterior(initial_params)
    print(f"Initial logpdf: {initial_logpdf}")
    if not np.isfinite(initial_logpdf):
        raise ValueError(f"Initial point for MCMC has non-finite logpdf: {initial_logpdf}")

    chains = 3
    mcmc = pints.MCMCController(log_posterior, chains, [initial_params]*chains)
    
    # Set maximum iterations
    mcmc.set_max_iterations(10000)  # Increase the number of iterations, e.g., to 100000
    mcmc.set_log_to_screen(True)
    
    chains = mcmc.run()

    return chains
