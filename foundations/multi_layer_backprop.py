import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Inputs ko NumPy arrays mein convert karo
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # ===== FORWARD PASS =====

        z1 = x @ W1.T + b1
        # Input × first layer weights + bias

        a1 = np.maximum(0, z1)
        # ReLU: negative values 0, positive same

        z2 = a1 @ W2.T + b2
        # Hidden layer output se final prediction

        loss = np.mean((z2 - y_true) ** 2)
        # MSE: prediction aur actual ka squared error average


        # ===== BACKWARD PASS =====

        n = len(y_true) if y_true.ndim > 0 else 1
        # Total output values

        dz2 = 2 * (z2 - y_true) / n
        # STEP 1: Output layer par MSE ka gradient
        # Prediction badhne par loss kitna change hoga

        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)
        # STEP 2: Output layer ke weights ka gradient
        # Output error × hidden layer activations

        db2 = dz2
        # Output layer bias ka gradient

        da1 = dz2.reshape(1, -1) @ W2
        da1 = da1.flatten()
        # STEP 3: Error ko W2 ke through hidden layer tak bhejo
        # Chain rule use hota hai

        dz1 = da1 * (z1 > 0).astype(float)
        # STEP 4: ReLU derivative apply karo
        # z1 > 0 -> gradient pass (1)
        # z1 <= 0 -> gradient block (0)

        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        # STEP 5: First layer weights ka gradient
        # Hidden layer error × original input

        db1 = dz1
        # First layer bias ka gradient

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }