
MODEL_NAME = 'similApp'
import numpy as np
import pickle
import scipy

with open('params/example_params_00.pickle', 'rb') as handle:
    params = pickle.load(handle)

# parameters of model
theta = params['theta']; theta[theta<0]=0
sigma = params['sigma']
mu = params['mu']
grid_size = params['grid_size']
xticklabs = params['xticklabs']

# theta = (theta + np.abs(theta.min()))+1e-2
import functools

@functools.lru_cache(maxsize=None)
def theta_fun(t):
    return (scipy.interpolate.interp1d(np.linspace(0,1,theta.size), theta)(t) +.5*1e-1)

@functools.lru_cache(maxsize=None)
def sg_fun(t):
    return scipy.interpolate.interp1d(np.linspace(0,1,sigma.size), sigma)(t)

# distribution of starting point
def R0(m = mu):
    return np.random.normal(params['R0_mean'],params['R0_std'])

# number of generated curves
N = 250

# discretisation of unit interval
T = np.linspace(0,1,grid_size)

# EulerMaruyama approximation step, for curves generation
dt = 1/grid_size






# MODEL_NAME = 'similApp'
# import numpy as np
# import pickle
# import scipy

# with open('params/example_params_00.pickle', 'rb') as handle:
#     params = pickle.load(handle)

# # parameters of model
# theta = params['theta']; theta[theta<0]=0
# sigma = params['sigma']
# mu = params['mu']
# grid_size = params['grid_size']

# # theta = (theta + np.abs(theta.min()))+1e-2
# import functools

# @functools.lru_cache(maxsize=None)
# def theta_fun(t):
#     return (scipy.interpolate.interp1d(np.linspace(0,1,theta.size), theta)(t) +.5*1e-1)

# @functools.lru_cache(maxsize=None)
# def sg_fun(t):
#     return scipy.interpolate.interp1d(np.linspace(0,1,sigma.size), sigma)(t)

# # distribution of starting point
# def R0(m = mu):
#     return np.random.normal(m,5)

# # number of generated curves
# N = 365

# # discretisation of unit interval
# T = np.linspace(0,1,grid_size)

# # EulerMaruyama approximation step, for curves generation
# dt = 1/grid_size

