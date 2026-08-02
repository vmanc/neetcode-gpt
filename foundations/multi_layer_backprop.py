import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.dot(np.array(x), np.array(W1).T) + b1
        a1 = np.maximum(0, z1) # ReLU activation
        z2 = np.dot(a1, np.array(W2).T) + b2
        loss = (1 / len(y_true))*np.sum(np.square(z2 - y_true))

        dL_z2 = 2*(z2 - y_true)/len(y_true)
        dL_W2 = np.outer(dL_z2, a1) 
        dL_b2 = dL_z2
        dL_a1 = np.dot(dL_z2, W2)
        dL_z1 = np.where(z1 <= 0, 0, dL_a1)
        dL_W1 = np.outer(dL_z1.T, x)
        dL_b1 = dL_z1

        return {
            'loss': np.round(loss, 4),
            'dW1': np.round(dL_W1, 4),
            'db1': np.round(dL_b1, 4),
            'dW2': np.round(dL_W2, 4),
            'db2': np.round(dL_b2, 4)
        }

