from scipy.integrate import odeint

def virus_model(y, t, alpha_f, beta, delta_f, gamma):
    X, Y, V = y
    dXdt = alpha_f * Y * V - beta * X * V
    dYdt = -alpha_f * Y * V
    dVdt = gamma * X * V - delta_f * V
    return [dXdt, dYdt, dVdt]

def solve_ode(parameters, initial_conditions, time):
    alpha_f, beta, delta_f, gamma = parameters
    return odeint(virus_model, initial_conditions, time, args=(alpha_f, beta, delta_f, gamma))
