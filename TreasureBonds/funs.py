# libraries 
import numpy as np
from numpy.random import normal

import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools

import scipy
from scipy.integrate import simps as simps

def giveMeNoise(size = 276):
    return np.random.multivariate_normal(mean = np.zeros(size), cov = np.eye(size)*(1/32/1.2)**2)

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



def RMSE(ar1,ar2):
    return ((ar1-ar2)**2).mean(0)**.5




def estimate_FVM(Rs, h_qv = .15, h_mean = .75 ):
    grid_size = Rs.shape[1]
    T = np.linspace(0,1,grid_size)
    
    
    # compute QV process of each curve
    emp_QVs = np.array([QV(R) for R in Rs])

    # 1st approach: AVERAGE, SMOOTH
    # average to get an estimate of the QV process
    mean_emp_QV = emp_QVs.mean(0)

    # bandwith parameter
    h = h_qv*grid_size**(-1/5)
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
    h = h_mean*grid_size**(-1/5)

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


