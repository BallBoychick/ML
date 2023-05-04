import numpy as np
import scipy

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
def sigmoid(Z):  

    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    cache = Z 
    return A, cache

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):  
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b) #save (A, W, b)
        A, activation_cache = sigmoid(Z) #save argument of Z
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network !!!!!!!!!!!!!!!!!!!!!!

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    Y = [int(Y) for Y in Y]
    E = -np.log(np.array([AL[j, Y[j]] for j in range(len(Y))]))



    return np.sum(E)



#----------------------------------Start-BackWord------------------------------------------------------#
def softmax_backword(dA, cache):
    Z = cache
    dZ = dA  
    return dZ.T

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ





def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "softmax":
        dZ = softmax_backword(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def to_full_batch(y, num_classes):
    y = [int(y) for y in y]
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    dAL = AL - to_full_batch(Y, len(AL[1]))
    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

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
        return parameters

    def iterations(self, X, Y):
        self.X = X.to_numpy()
        self.Y = Y.to_numpy()
        for i in range(0, self.num_iterations):
            AL, caches = L_model_forward(X.T, self.initialize)
            cost = compute_cost(AL, Y)
            print("Cost", cost)
            grads =  L_model_backward(AL, Y, caches)
            param = update_parameters(self.initialize, grads, self.learning_rate)
        return param, cost

    def predict(self, X, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        probas, caches = L_model_forward(X.T, parameters)
        Final = []
        for i in probas:
            Final.append(np.argmax(i))
                
        return Final
