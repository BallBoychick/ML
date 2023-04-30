import numpy as np
import scipy

def softmax(z):
  '''Return the softmax output of a vector.'''
#   exp_z = np.exp(z)
#   sum = exp_z.sum()
#   softmax_z = (exp_z/sum)
#   A = softmax_z
#   cache = z
#   return A, cache
  A = []
  for i in z:
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
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def linear_forward(A, W, b):
    # A = A.T  
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    # print("Z", Z)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):  
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
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
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
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
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)

    assert(AL.shape == (3,X.shape[1]))
    # print("ALLLLLLLL", AL.shape) 

    return AL, caches


def compute_cost(AL, Y):
    #FIX COST FOR MULTICLASS
    # print("Y", Y.shape)
    m = Y.shape[0]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    # print(cost)
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    # assert(cost.shape == ())

    return cost



#----------------------------------Start-BackWord------------------------------------------------------#
def softmax_backword(dA, cache):
#     Z = cache
#     m, n = Z.shape
#     p = softmax(Z)
#   # First we create for each example feature vector, it's outer product with itself
#     # ( p1^2  p1*p2  p1*p3 .... )
#     # ( p2*p1 p2^2   p2*p3 .... )
#     # ( ...                     )
#     tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
#     # Second we need to create an (n,n) identity of the feature vector
#     # ( p1  0  0  ...  )
#     # ( 0   p2 0  ...  )
#     # ( ...            )
#     tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)
#     # Then we need to subtract the first tensor from the second
#     # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
#     # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
#     # ( ...                              )
#     dSoftmax = tensor2 - tensor1
#     # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
#     dZ = np.einsum('ijk,ik->ij', dSoftmax, dA)  # (m, n)

#     return dZ
    n = np.size(dA)
    return np.dot((np.identity(n) - dA.T) * dA, dA)
 


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


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    print("ALL", AL)
    # print("ALL.shape", AL.shape)
    # print("YYYY_before", type(Y))
    # Y = Y.reshape(AL.shape[0], AL.shape[1]) # after this line, Y is the same shape as AL
    # Y = np.reshape(Y * 3, (3, len(Y)))
    Y = np.array([Y]*3)
    print("YYYY_after", Y)
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    print("dAL", dAL)
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    
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
            print("YES")
            cost = compute_cost(AL, self.Y)

            grads =  L_model_backward(AL, self.Y, caches)

            update_parameters(self.initialize, grads, self.learning_rate)

        # return AL
        return self.initialize

    def predict(self, X, y, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = L_model_forward(self.X.T, parameters)

        
        # convert probas to 0/1 predictions
        # for i in range(0, probas.shape[1]):
        #     if probas[0,i] > 0.5:
        #         p[0,i] = 1
        #     else:
        #         p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        # print("Accuracy: "  + str(np.sum((p == y)/m)))

        r2 = np.reshape(probas, (probas.shape[1], probas.shape[0]))

        Final = []
        for i in r2:
            Final.append(np.argmax(i))
                
        return Final

