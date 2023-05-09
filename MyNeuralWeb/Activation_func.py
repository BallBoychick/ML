import pandas as pd
import numpy as np
from scipy.special import expit

class Actifation_funcs:
    @staticmethod
    def sigmoid_scipy(z):
        A = expit(z)
        cache = z
        return A, cache
    @staticmethod
    def softmax(z):
        '''Return the softmax output of a vector.'''
        r1 = np.reshape(z, (len(z[1]), len(z)))
        A = []
        for i in r1:
            exp_z = np.exp(i)
            sum = exp_z.sum()
            softmax_z = (exp_z/sum)
            A.append(softmax_z)
            cache = z
        return np.asarray(A), cache

    @staticmethod
    def sigmoid(Z):  

        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    @staticmethod
    def relu(Z):
        
        A = np.maximum(0,Z)
        
        cache = Z 
        return A, cache
    
    @staticmethod
    def leaky_relu(Z):
        alpha = 0.1
        A = np.maximum(alpha*Z, Z)
        cache = Z
        return A, cache

    @staticmethod
    def linear_func(Z):
        cache = Z
        return Z, cache
