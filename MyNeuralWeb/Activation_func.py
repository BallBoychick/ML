import pandas as pd
import numpy as np

class Actifation_funcs:
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
    def linear_func(Z):
        cache = Z
        return Z, cache
