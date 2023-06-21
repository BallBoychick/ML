import numpy as np
import scipy
from LossFunction import LossFunction
from Activation_func import Actifation_funcs
from Derivative_activation_functions import Derivative_Actifation_functions
from dAL import dALoss
#----------------------------------Start-BackWord------------------------------------------------------#


# def to_full_batch(y, num_classes):
#     y = [int(y) for y in y]
#     y_full = np.zeros((len(y), num_classes))
#     for j, yj in enumerate(y):
#         y_full[j, yj] = 1
#     return y_full

#-------------------Class---------------------------------#
class NeuralWeb:
    def __init__(self, layers_dims, learning_rate, num_iterations):
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        self.initialize = self.initialize_parameters_deep(self.layers_dims)


    def initialize_parameters_deep(self, layer_dims):
    
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #* 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            # parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) 
        return parameters
    
    #------------------Start-Forward----------------------------#
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        # print("Z", Z)
        # print("cache_after_linear_forw", cache)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):  
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        print("Z", Z)
        A, activation_cache = activation(Z)
        cache = (linear_cache, activation_cache)
        print("A\n", A)
        # print("CACHE_AFTER_LINEAR_ACT_FORW",  cache)
        return A, cache

    def L_model_forward(self, X, parameters):

        caches = []
        A = X
        # print("A_before", A)
        L = len(parameters) // 2                  # number of layers in the neural network !!!!!!!!!!!!!!!!!!!!!!
        # print("AAAAAA", A)
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = Actifation_funcs.tanh)
            caches.append(cache)
            # print("AА2", A)
            # print("AFTER_EVERY_ITER", cache)
        # print("A_AFTER_RELU",A)
        # print("A2", A)
        print("\nrelu\n")
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = Actifation_funcs.linear_func)
        print("\nline\n")
        caches.append(cache)
        # print("A_AFTER_LINEAR",AL)

        # print("CACHES_AFTER_EPOHS", caches)
        print("AL\n", AL)
        return AL, caches

    #-----------------------Start-backword---------------------------#

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        # print("dZ",  len(A_prev))
        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True) #axis = 1
        dA_prev = np.dot(W.T,dZ)
        return dA_prev, dW, db


    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        print("dA", dA)
        dZ = activation(dA, activation_cache)
        print("dZ", dZ)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        # dAL = AL - to_full_batch(Y, len(AL[1])) #ETO OTDELNO
        dAL = dALoss.dAL_regr(AL, Y)
        print("dAL", dAL)
        current_cache = caches[L-1]

        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = Derivative_Actifation_functions.lineear_func_back)
        # print("grads1", grads)
        print("\nline2\n")
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = Derivative_Actifation_functions.tanh_derivative)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        # print("grads2", grads)
        return grads


    def update_parameters(self, parameters, grads, learning_rate):

        L = len(parameters) // 2 # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
            
        return parameters
    def iterations(self, X, Y):
        self.X = X.to_numpy()
        self.Y = Y.to_numpy()
        # print("SELF.X", self.X)
        param = self.initialize
        for i in range(0, self.num_iterations):
            print("Param", param)
            AL, caches = self.L_model_forward(self.X.T, param)
            cost = LossFunction.compute_cost_regr(AL, self.Y)
            # print("ALLLLLL", AL)
            print("Cost", cost)
            grads =  self.L_model_backward(AL, self.Y, caches)
            param = self.update_parameters(param, grads, self.learning_rate)
        return param, cost

    def predict(self, X, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        probas, caches = self.L_model_forward(X.T, parameters)
        Final = []
        for i in probas:
            Final.append(np.argmax(i))
                
        return Final

    def predict_regr(self, X, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        probas, caches = self.L_model_forward(self.X.T, parameters)
        
                
        return probas
