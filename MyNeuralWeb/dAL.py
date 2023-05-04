import pandas as pd
import numpy as np

def to_full_batch(y, num_classes):
        y = [int(y) for y in y]
        y_full = np.zeros((len(y), num_classes))
        for j, yj in enumerate(y):
            y_full[j, yj] = 1
        return y_full

class dALoss:
    @staticmethod
    def dAL_classifier(AL, Y):
        return AL - to_full_batch(Y, len(AL[1]))

    def dAL_regr(AL, Y):
        y = [int(y) for y in y]
        return AL - y