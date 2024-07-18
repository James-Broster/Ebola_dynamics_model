from models.base_model import solve_ode

def run_model(parameters, initial_conditions, time):
    result = solve_ode(parameters, initial_conditions, time)
    return result  # Return the complete state [f1, f2, V]
