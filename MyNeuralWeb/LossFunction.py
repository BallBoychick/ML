import pandas as pd
import numpy as np
class LossFunction:
    @staticmethod
    def compute_cost_class(AL, Y):
        Y = [int(Y) for Y in Y]
        E = -np.log(np.array([AL[j, Y[j]] for j in range(len(Y))]))



        return np.sum(E)
    @staticmethod
    def compute_cost_regr(AL, Y):
        #MSE
        Y = [int(Y) for Y in Y]
        return np.sum((Y-AL)**2)/len(Y)