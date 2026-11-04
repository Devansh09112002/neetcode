import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def train(self, X, y, epochs, lr):

        n = X.shape[0]
        # Total training samples (rows)
        # Example: X.shape = (100, 3) -> n = 100

        w = np.zeros(X.shape[1])
        # Har feature ka initial weight = 0
        # Example: 3 features -> w = [0, 0, 0]
        # Training mein ye weights gradually learn honge

        b = 0.0
        # Initial bias = 0
        # Bias prediction ko adjust/shift karne mein help karta hai

        for _ in range(epochs):
            # Har epoch mein poore dataset par training
            # Example: epochs=100 -> 100 baar prediction aur updates

            # ===== STEP 1: FORWARD PASS =====

            y_hat = X @ w + b
            # Inputs × Weights + Bias = Predictions
            # @ means matrix multiplication
            # Har sample ke features ko weights se multiply karke sum
            # Output: n predictions, ek har training sample ke liye

            error = y_hat - y
            # Predicted - Actual
            # Positive error -> prediction actual se zyada
            # Negative error -> prediction actual se kam
            # Example: predicted=8, actual=5 -> error=3

            # ===== STEP 2: BACKPROPAGATION =====

            dw = (2.0 / n) * (X.T @ error)
            # MSE ka gradient w.r.t. weights
            # Har weight badhne par loss kitna change hoga
            # X.T = transpose: (n, features) -> (features, n)
            # X.T @ error -> har feature ka weighted error sum
            # 2/n -> MSE derivative ka mathematical factor

            db = (2.0 / n) * np.sum(error)
            # MSE ka gradient w.r.t. bias
            # Sabhi prediction errors ko sum karke 2/n se multiply
            # Bias sabhi predictions mein add hota hai
            # Isliye uska gradient total errors par depend karta hai

            # ===== STEP 3: GRADIENT DESCENT =====

            w = w - lr * dw
            # Weights ko loss kam karne ki direction mein update
            # lr = learning rate, update ka step size
            # dw positive -> weight decrease
            # dw negative -> weight increase
            # Example: w=2, lr=0.01, dw=4 -> new w=1.96

            b = b - lr * db
            # Bias ko bhi gradient descent se update karo
            # Har epoch ke baad updated w aur b next epoch mein use honge

        return (np.round(w, 5), round(float(b), 5))
        # Training complete hone ke baad final weights aur bias
        # Values ko 5 decimal places tak round karke return