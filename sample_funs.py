# libraries 
import numpy as np
from numpy.random import normal
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools

import scipy
from scipy.integrate import simps as simps



def LLS(T,Y,K,h):
    
    """
    Perform local linear smoothing on the given observations.

    Parameters:
    T (array-like): Grid of time points.
    Y (array-like): Observations corresponding to the time points.
    K (function): Kernel bivariate function that computes weights based on distances.
    h (float): Bandwidth parameter for the smoothing.

    Returns:
    smoothed_values (array): Smoothed values corresponding to each time point.

    """
    K_h = lambda t: K(t/h)/h   
    cali_T = lambda t0 : np.array( [ ( (T - t0) )**p for p in [0,1] ] ).T
    bold_W = lambda t0 : np.diag(np.array([K_h(_t - t0) for _t in T]))
    def minimizers(t0):
        res =  np.linalg.inv(  cali_T(t0).T @ bold_W(t0) @ cali_T(t0) )  @ cali_T(t0).T @  bold_W(t0) @  Y 
        return res
    return lambda t0: minimizers(t0)



def QV(arr):
    """
    Calculate the quadratic variation process of a numpy array.
    
    Parameters:
        arr (numpy.ndarray): Input numpy array.
        
    Returns:
        numpy.ndarray: Quadratic variation process array.
    """
    squared_diff = np.diff(arr) ** 2
    quadratic_variation = np.array([[0] + list(np.cumsum(squared_diff))]).ravel()
    return quadratic_variation


def EulerMaruyama_approx_sol(dt, T, mu, theta ,sigma, R0,ax=None):
    """ 
    Returns discrete numerical approximation of stochastic process solving an SDE of the form
            dR(t) = theta(t) ( mu - R(t))dt + sigma(t)  dW(t)
    by approximating the continous solution X(t) by random variables 
    X_n = X(t_n) where t_n = n∆t, n = 0,1,2,...,1/dt and ∆t = 1/grid_size

    Parameters
    ----------
    grid_size : int
        Discretisation of [0,1] interval. Determines step size ∆t = 1/N
    mu : function
        Drift of SDE, takes (x, t) as input
    sigma : function
        Diffusion of SDE, takes (x, t) as input 
    R0 : float
        Starting point of solution

    Returns
    ----------
    R : array(float) np.array of size N evaluated on the equispaced grid T
    """ 
    grid_size = int(1/dt)
    t = np.linspace(0,1,grid_size)
    R = np.zeros(grid_size)

    R[0] = R0()
    dt = 1/grid_size
    
    for i in range(grid_size-1):
        dR = theta(t[i])*(mu - R[i])*dt + sigma(t[i])*normal()*np.sqrt(dt)
        R[i+1] = R[i] + dR
    
    if ax is not None:
        ax.plot(t,R,lw=1,alpha=.3)
    return R[::int(grid_size/T.size)]


def RMSE(ar1,ar2):
    return ((ar1-ar2)**2).mean(0)**.5


# ======================


def runs(Rs):
    grid_size = Rs.shape[1]
    T = np.linspace(0,1,grid_size)
    
    
    # compute QV process of each curve
    emp_QVs = np.array([QV(R) for R in Rs])

    # 1st approach: AVERAGE, SMOOTH
    # average to get an estimate of the QV process
    mean_emp_QV = emp_QVs.mean(0)

    # bandwith parameter
    h = .15*grid_size**(-1/5)
    # smoothing kernel
    K = lambda x : 3/4*(1-x**2) if  3/4*(1-x**2) > 0 else 0

    # local linear smooth of empirical QV to estimate derivative
    _ =  np.array([LLS(T,mean_emp_QV,K,h)(t) for t in T])
    LLS_mean_emp_QV = _[:,0]; LLS_Der_mean_emp_QV = _[:,1]


    # 2nd approach: TAKE NUMERICAL DERIVATIVE, AVERAGE, SMOOTH
    # compute the numerical derivative of the empirical QV of each sample curve, and average them
    NumDer_emp_QVs = (np.diff(emp_QVs)*grid_size).mean(0)
    # smooth the resulting average derivative
    LLS_empirical_der_meanQVs = np.array([ LLS(T[:-1],NumDer_emp_QVs,K,h)(t) for t in T])[:,0]

    # Mean and Variance
    # bandwith
    h = .75*grid_size**(-1/5)

    # estimate mean function and its derivative by LLS
    _ = np.array([LLS(T,Rs.mean(0),K,h)(t) for t in T])
    LLS_mean = _[:,0]; LLS_Der_mean = _[:,1]

    # estimate variance function and its derivative by LLS
    _ = np.array([LLS(T,(Rs**2).mean(0) - Rs.mean(0)**2,K,h)(t) for t in T])
    LLS_variance = _[:,0]; LLS_Der_variance = _[:,1] 
    
    
    return {'LLS_mean_emp_QV': LLS_mean_emp_QV, 
            'LLS_Der_mean_emp_QV':LLS_Der_mean_emp_QV,
            'LLS_empirical_der_meanQVs':LLS_empirical_der_meanQVs,
            'LLS_mean': LLS_mean, 'LLS_Der_mean': LLS_Der_mean,
            'LLS_variance':LLS_variance, 'LLS_Der_variance':LLS_Der_variance}




