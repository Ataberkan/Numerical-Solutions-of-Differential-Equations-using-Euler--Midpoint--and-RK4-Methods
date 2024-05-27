import numpy as np
import matplotlib.pyplot as plt

# Constants
A = 10
B = 25
C = 8/3
dt = 0.03
num_steps = int(40/dt)  # Total simulation time

x0, y0, z0 = 1, 1, 1

# Define the system of equations
def f(state):
    x, y, z = state
    dxdt = A * (y - x)
    dydt = -x * z + B * x - y
    dzdt = x * y - C * z
    return np.array([dxdt, dydt, dzdt])

def euler(state, dt):
    return state + dt * f(state)

def midpoint(state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    return state + dt * k2

def rk4(state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Simulation
def simulate(method):
    states = [np.array([x0, y0, z0])]
    for _ in range(num_steps):
        states.append(method(states[-1], dt))
    states = np.array(states)
    return states[:, 0], states[:, 1], states[:, 2]

# Plotting
x_euler, y_euler, z_euler = simulate(euler)
x_mid, y_mid, z_mid = simulate(midpoint)
x_rk4, y_rk4, z_rk4 = simulate(rk4)

plt.figure(figsize=(12, 6))
plt.plot(x_euler, z_euler, label='Euler', alpha=0.6)
plt.plot(x_mid, z_mid, label='Midpoint', alpha=0.6)
plt.plot(x_rk4, z_rk4, label='RK4', alpha=0.6)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Phase Plot of x vs z (Euler, Midpoint, RK4)')
plt.legend()
plt.show()
