import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # STEP 1: Forward Pass
        z = np.dot(x, w) + b
        # Inputs * weights + bias

        y_hat = 1.0 / (1.0 + np.exp(-z))
        # Sigmoid lagakar prediction nikalo (0 to 1)

        # STEP 2: Error Calculate
        error = y_hat - y_true
        # Prediction aur actual answer ka difference

        # STEP 3: Sigmoid Derivative
        sigmoid_deriv = y_hat * (1.0 - y_hat)
        # Sigmoid ka slope calculate karta hai
        # Batata hai z change hone par prediction kitna change hoga

        # STEP 4: Chain Rule
        delta = error * sigmoid_deriv
        # Loss ka gradient w.r.t. z
        # Error * sigmoid derivative
        # Batata hai z badhne par loss kitna change hoga

        # STEP 5: Weight aur Bias Gradients
        dL_dw = np.round(delta * x, 5)
        # Har weight ka gradient = delta * corresponding input
        # Input bada -> weight ka gradient magnitude bhi bada

        dL_db = round(float(delta), 5)
        # Bias ka gradient = delta
        # Kyunki z = wx + b, aur dz/db = 1

        return (dL_dw, dL_db)
        # Gradients return, weights abhi update nahi hue