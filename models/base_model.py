from scipy.integrate import odeint

def virus_model(y, t, alpha_f, beta, delta_f, gamma, epsilon=0, t_star=0):
    f1, f2, V = y
    H = 1 if t >= t_star else 0
    dXdt = alpha_f * f2 * V * (1 - epsilon * H) - beta * f1 * V
    dYdt = -alpha_f * f2 * V * (1 - epsilon * H)
    dVdt = gamma * f1 * V - delta_f * V
    return [dXdt, dYdt, dVdt]

def solve_ode(parameters, initial_conditions, time, therapy_params=None):
    if therapy_params:
        epsilon, t_star = therapy_params
    else:
        epsilon, t_star = 0, 0
    alpha_f, beta, delta_f, gamma = parameters
    return odeint(virus_model, initial_conditions, time, args=(alpha_f, beta, delta_f, gamma, epsilon, t_star))
