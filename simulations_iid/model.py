# libraries 

MODEL_NAME = 'model_1'
import numpy as np


# parameters of model
theta_fun = lambda t : 1-.5*t
sg_fun = lambda t :  1+ (t-.5)**2
mu = 3

# distribution of starting point
def R0(m = 0):
    return np.random.normal(m,1)

# number of generated curves
N = 250

# discretisation of unit interval
grid_size = 500
T = np.linspace(0,1,grid_size)

# EulerMaruyama approximation step, for curves generation
dt = 1e-3