from models.base_model import solve_ode

def run_model(parameters, initial_conditions, time, therapy_params=None):
    result = solve_ode(parameters, initial_conditions, time, therapy_params)
    return result
