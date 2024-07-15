from models.base_model import solve_ode

def run_model(parameters, time):
    y0 = [0.00670143873431456, 1.0, 36504.3997450364]
    result = solve_ode(parameters, y0, time)
    return result[:, 2]
