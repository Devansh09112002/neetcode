import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(self, x, gamma, beta, running_mean,
                   running_var, momentum, eps, training):

        # Inputs ko NumPy arrays mein convert karo
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        if training:
            # ===== TRAINING MODE =====

            batch_mean = np.mean(x, axis=0)
            # Har feature ka current batch average

            batch_var = np.var(x, axis=0)
            # Har feature ka current batch variance

            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            # Current batch statistics se normalize karo

            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            # Old running mean + current batch mean ka weighted average
            # Example: momentum=0.1 -> 90% old + 10% new

            running_var = (1 - momentum) * running_var + momentum * batch_var
            # Running variance bhi similarly update karo

        else:
            # ===== INFERENCE MODE =====

            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            # Saved training statistics se normalize karo
            # Current batch ka mean/variance use nahi hoga

        out = gamma * x_hat + beta
        # Gamma -> scale, Beta -> shift
        # Dono learnable parameters hain

        return (
            np.round(out, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )