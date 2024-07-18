from scipy.integrate import odeint

def virus_model(y, t, alpha_f, beta, delta_f, gamma):
    f1, f2, V = y
    dXdt = alpha_f * f2 * V - beta * f1 * V
    dYdt = -alpha_f * f2 * V
    dVdt = gamma * f1 * V - delta_f * V
    return [dXdt, dYdt, dVdt]

def solve_ode(parameters, initial_conditions, time):
    alpha_f, beta, delta_f, gamma = parameters
    return odeint(virus_model, initial_conditions, time, args=(alpha_f, beta, delta_f, gamma))
