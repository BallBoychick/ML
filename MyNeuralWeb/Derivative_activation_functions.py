import pandas as pd
import numpy as np

class Derivative_Actifation_functions:
    @staticmethod
    def softmax_backword(dA, cache):
        Z = cache
        dZ = dA  
        return dZ.T
    @staticmethod
    def relu_backward(dA, cache):
        
        Z = cache
        dZ = np.array(dA, copy=True)
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        return dZ
    @staticmethod
    def sigmoid_backward(dA, cache):
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        return dZ
    
    @staticmethod
    def lineear_func_back(dA, cache):
        return 1