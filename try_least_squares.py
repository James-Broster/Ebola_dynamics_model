import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Provided data
time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
virusload = np.array([32359.37, 15135612, 67608298, 229086765, 245470892, 398107171, 213796209, 186208714, 23988329, 630957.3, 4265795, 323593.7, 53703.18, np.nan, 141253.8])

# Remove NaN values for plotting
valid_idx = ~np.isnan(virusload)
time = time[valid_idx]
virusload = virusload[valid_idx]

# Parameters from the provided model
alpha_f = 1.57763931289105e-09
delta_f = 1.50836617187542
beta = 6.12783508806595e-07
gamma = 1386.64900596809
X0f = 0.00670143873431456
V0f = 36504.3997450364

# Initial conditions
X_0 = X0f
Y_0 = 1.0  # Assuming initial Y is 1
V_0 = V0f
y0 = [X_0, Y_0, V_0]

# Print initial conditions and parameters
print("Initial conditions:", y0)
print("Parameters: alpha_f =", alpha_f, ", beta =", beta, ", delta_f =", delta_f, ", gamma =", gamma)

# Differential equations model
def model(y, t, alpha_f, beta, delta_f, gamma):
    X, Y, V = y
    dXdt = alpha_f * Y * V - beta * X * V
    dYdt = -alpha_f * Y * V
    dVdt = gamma * X * V - delta_f * V
    return [dXdt, dYdt, dVdt]

# Simulate the model using odeint
simulated = odeint(model, y0, time, args=(alpha_f, beta, delta_f, gamma))
simulated_virusload = simulated[:, 2]

# Print simulated values
print("Simulated virus load:", simulated_virusload)

# Plot data vs model
plt.figure()
plt.plot(time, virusload, 'o', label='Data')
plt.plot(time, simulated_virusload, '-', label='Model')
plt.xlabel('Time')
plt.ylabel('Virus Load')
plt.legend()
plt.show()
