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

from ExtendedKalmanFilterDB import ExtendedKalmanFilter

# Discretization time step (frequency of measurements)
deltaTime=0.01

# Initial true state
x0 = np.array([np.pi/3, 0, 0.5, 0])

# Simulation duration in timesteps
simulationSteps=400
totalSimulationTimeVector=np.arange(0, simulationSteps*deltaTime, deltaTime)

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)
def stateSpaceModel(x, t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))

        Dynamics are time-invariant in our case, so t is not used.
        
        Parameters:
            x : state variables (column-vector)
            t : time

        Returns:
            f : dx(t)/dt, describes the system of ODEs
    """

    g = 9.81
    l1 = 1
    l2 = 1
    m1 = 1
    m2 = 1
    
    dxdt = np.array(
        [
            x[1],
            (-g*(2*m1+m2)*np.sin(x[0]) - m2*g*np.sin(x[0]-2*x[2]) - 2*np.sin(x[0]-x[2])*m2*((x[3]**2)*l2 + (x[1]**2)*l1*np.cos(x[0]-x[2])))/(l1*(2*m1+m2-m2*np.cos(2*x[0]-2*x[2]))),
            x[3],
            (2*np.sin(x[0]-x[2])*((x[1]**2)*l1*(m1+m2) + g*(m1+m2)*np.cos(x[0]) + (x[3]**2)*l2*m2*np.cos(x[0]-x[2])))/(l2*(2*m1+m2-m2*np.cos(2*x[0]-2*x[2])))
        ]
    )
    return dxdt

# True solution x(t)
x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)
#print(x_t_true)


"""
    EKF initialization
"""
# Initial state belief distribution (EKF assumes Gaussian distributions)
x_0_mean = np.zeros(shape=(4,1))  # column-vector
x_0_mean[0] = x0[0] + 3*np.random.randn()
x_0_mean[1] = x0[1] + 3*np.random.randn()
x_0_mean[2] = x0[2] + 3*np.random.randn()
x_0_mean[3] = x0[3] + 3*np.random.randn()
x_0_cov = 10*np.eye(4,4)  # initial value of the covariance matrix

# Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
Q=0.00001*np.eye(4,4)

# Measurement noise covariance matrix for EKF
R = 0.05*np.eye(2,2)

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
    z_t =   np.array([
        x_t_true[t, 0] + np.sqrt(measurement_noise_var)*np.random.randn(),
        x_t_true[t, 2] + np.sqrt(measurement_noise_var)*np.random.randn()
            ]).reshape((2,1))

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

    # Create a figure and axis (subplot)
    plt.figure(figsize=(10, 5))
    # Plot the first subplot
    true_x1 = [x[2] for x in x_t_true]
    est_x1 = [x[2] for x in x_t_est]
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    plt.plot(true_x1, label='true_x3')
    plt.plot(est_x1, label='est_x3')
    plt.title(f'x3')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    # Plot the second subplot
    true_x2 = [x[3] for x in x_t_true]
    est_x2 = [x[3] for x in x_t_est]
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 1
    plt.plot(true_x2, label='true_x4')
    plt.plot(est_x2, label='est_x4')
    plt.title(f'x4')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the plot
    plt.show()
    
    return

plot_results(x_t_true, EKF.posteriorMeans, EKF.R)
