import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = x
        # h = current input, initially original x

        for i in range(len(weights)):
            # Har layer par loop chalega
            # Example: 3 layers -> i = 0, 1, 2

            h = h @ weights[i] + biases[i]
            # STEP 1: Linear transformation
            # Current input × current layer weights + bias
            # @ = matrix multiplication

            if i < len(weights) - 1:
                # Check: kya ye last layer nahi hai?
                # 3 layers mein i=0,1 par True; i=2 par False

                h = np.maximum(0, h)
                # STEP 2: ReLU activation
                # Negative values -> 0
                # Positive values -> same
                # Sirf hidden layers par ReLU lagta hai

        return np.round(h, 5)
        # Final output ko 5 decimal places tak round