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
#   A = []
#   for i in z:
#     exp_z = np.exp(i)
#     sum = exp_z.sum()
#     softmax_z = (exp_z/sum)
#     A.append(softmax_z)
#     cache = z
#   return np.asarray(A), cache
def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    # print("Z", Z)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):  
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b) #save (A, W, b)
        A, activation_cache = sigmoid(Z) #save argument of Z
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        # print("Z_soft", Z)
        A, activation_cache = softmax(Z)
        # print("AAAAA", A)
    # assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    # print("X", X.shape)
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
        # print("AAAAAAA", A.shape)
    # print("AAAA", A)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)
    # print("AL", AL)

    # assert(AL.shape == (3,X.shape[1]))
    # print("ALLLLLLLL", AL.shape) 

    return AL, caches


def compute_cost(AL, Y):
    #FIX COST FOR MULTICLASS
    # print("Y", Y.shape)
    # m = Y.shape[0]
    # y2 = y.to_numpy()
    Y = [int(Y) for Y in Y]
    # # Compute loss from aL and y.
    # cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    # # print(cost)
    # cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    # # assert(cost.shape == ())
    # print("YYYY", Y)
    E = -np.log(np.array([AL[j, Y[j]] for j in range(len(Y))]))

    # def sparse_cross_entropy_batch(z, y):
    #   return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))
    # E = np.sum(sparse_cross_entropy_batch(z, y))

    return np.sum(E)



#----------------------------------Start-BackWord------------------------------------------------------#
def softmax_backword(dA, cache):
    Z = cache
    dZ = dA
    
    return dZ.T
 


def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ





def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
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
    # print("ALL", AL)
    # print("Y", Y)
    dAL = AL - to_full_batch(Y, len(AL[1]))
    # print("dAL", dAL)
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]

    #every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2) 
    #the cache of linear_activation_forward() with "softmax" (there is one, index L-1)

    # print("current_cache", current_cache) 
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    # print("GRADS", grads)
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


# def predict(X, y, parameters):
#     m = X.shape[1]
#     n = len(parameters) // 2 # number of layers in the neural network
#     p = np.zeros((1,m))
    
#     # Forward propagation
#     probas, caches = L_model_forward(X, parameters)

    
#     # convert probas to 0/1 predictions
#     for i in range(0, probas.shape[1]):
#         if probas[0,i] > 0.5:
#             p[0,i] = 1
#         else:
#             p[0,i] = 0
    
#     #print results
#     #print ("predictions: " + str(p))
#     #print ("true labels: " + str(y))
#     print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    # return p
#-------------------Class---------------------------------#
class NeuralWeb:
    def __init__(self, X, Y, layers_dims, learning_rate, num_iterations):
        self.X = X.to_numpy()
        self.Y = Y.to_numpy()
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        self.initialize = self.initialize_parameters_deep(self.layers_dims)

        self.pred = self.predict(self.X, self.Y, self.initialize)


        self.iters = self.iterations(self.num_iterations)
    def initialize_parameters_deep(self, layer_dims):
    
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #* 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            # assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            # assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))    
        return parameters

    def iterations(self, num_iterations):
        for i in range(0, num_iterations):
            AL, caches = L_model_forward(self.X.T, self.initialize)
            # print("Al", AL)
            # print("YES")
            #FIX coast
            cost = compute_cost(AL, self.Y)
            # print("Cost", cost)

            grads =  L_model_backward(AL, self.Y, caches)
            # print("GRADS", grads)
            update_parameters(self.initialize, grads, self.learning_rate)

        # return AL
        return self.initialize, cost

    def predict(self, X, y, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = L_model_forward(self.X.T, parameters)
        Final = []
        for i in probas:
            Final.append(np.argmax(i))
                
        return Final
