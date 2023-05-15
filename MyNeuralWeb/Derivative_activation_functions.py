import pandas as pd
import numpy as np
from scipy.special import expit
import math
class Derivative_Actifation_functions:
    @staticmethod
    def softmax_backword(dA, cache):
        Z = cache
        dZ = dA  
        return dZ.T
    @staticmethod
    def relu_backward(dA, cache):
        
        Z = cache
        # dZ = np.array(dA, copy=True)
        
        # When z <= 0, you should set dz to 0 as well. 
        # dZ[Z[:, :1] <= 0] = 0
        dZ = np.ones_like(dA)
        dZ[Z <= 0] = 0
        return dZ
    
    @staticmethod
    def leaky_relu_derivative(dA, cache):
        Z = cache
        alpha = 0.1
        dZ = np.ones_like(dA)
        dZ[Z < 0] = alpha
        return dZ

    @staticmethod
    def sigmoid_backward_scipy(dA, cache):
        
        Z = cache
        
        s = expit(Z)
        dZ = dA * s * (1-s)
        
        return dZ

    @staticmethod
    def sigmoid_backward(dA, cache):
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        return dZ
    
    @staticmethod
    def lineear_func_back(dA, cache):
        # print("DA", len(dA), len(dA[0]))
        dZ = dA
        return dZ

#(len(z[1]), len(z))
    def tanh_derivative(dA, cache):
        Z = cache
        t = np.tanh(Z)
        dZ = dA * (1 - t**2)
        return dZ