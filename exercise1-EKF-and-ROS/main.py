"""
    Robot Learning
    Exercise 1

    Extended Kalman Filter

    Polito A-Y 2023-2024
"""
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from ExtendedKalmanFilter import ExtendedKalmanFilter

# Discretization time step (frequency of measurements)
deltaTime=0.01

# Initial true state
x0 = np.array([np.pi/3, 0.5])

# Simulation duration in timesteps
simulationSteps=400
totalSimulationTimeVector=np.arange(0, simulationSteps*deltaTime, deltaTime)

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)
def stateSpaceModel(x,t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """
    g=9.81
    l=1
    dxdt=np.array([x[1], -(g/l)*np.sin(x[0])])
    return dxdt

# True solution x(t)
x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)
#print(x_t_true)


"""
    EKF initialization
"""
# Initial state belief distribution (EKF assumes Gaussian distributions)
x_0_mean = np.zeros(shape=(2,1))  # column-vector
x_0_mean[0] = x0[0] + 3*np.random.randn()
x_0_mean[1] = x0[1] + 3*np.random.randn()
x_0_cov = 10*np.eye(2,2)  # initial value of the covariance matrix

# Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
Q=0.00001*np.eye(2,2)

# Measurement noise covariance matrix for EKF
R = np.array([[0.05]])

# create the extended Kalman filter object
EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime)

"""
    Simulate process
"""
measurement_noise_var = 0.05  # Actual measurement noise variance (uknown to the user)

for t in range(simulationSteps-1):
    # PREDICT step
    EKF.forwardDynamics()
    
    # Measurement model
    z_t = x_t_true[t, 0] + np.sqrt(measurement_noise_var)*np.random.randn()

    # UPDATE step
    EKF.updateEstimate(z_t)



"""
    Plot the true vs. estimated state variables
"""

def plot_results(x_t_true, x_t_est, R):
    # Create a figure and axis (subplot)
    plt.figure(figsize=(10, 5))
    # Plot the first subplot
    true_x1 = [x[0] for x in x_t_true]
    est_x1 = [x[0] for x in x_t_est]
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    plt.plot(true_x1, label='true_x1')
    plt.plot(est_x1, label='est_x1')
    plt.title(f'x1')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    # Plot the second subplot
    true_x2 = [x[1] for x in x_t_true]
    est_x2 = [x[1] for x in x_t_est]
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 1
    plt.plot(true_x2, label='true_x2')
    plt.plot(est_x2, label='est_x2')
    plt.title(f'x2')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the plot
    plt.show()
    return
plot_results(x_t_true, EKF.posteriorMeans, EKF.R)


"""
    variations of R
"""

R_values = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
print(f"R_values = {R_values}")
EKF_vector = []
for r in R_values:
    EKF_vector.append(ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, r, deltaTime))
    
for t in range(simulationSteps-1):
    # Measurement model
    z_t = x_t_true[t, 0] + np.sqrt(measurement_noise_var)*np.random.randn()
    
    for ekf in EKF_vector:
        # PREDICT step
        ekf.forwardDynamics()
        # UPDATE step
        ekf.updateEstimate(z_t)


for ekf in EKF_vector:
    plot_results(x_t_true, ekf.posteriorMeans, ekf.R)

