import numpy as np
from typing import List

class Solution:
    def rms_norm(self, x, gamma, eps):

        x = np.array(x)
        # Input ko NumPy array mein convert karo

        gamma = np.array(gamma)
        # Learnable scaling parameters

        rms = np.sqrt(np.mean(x ** 2) + eps)
        # STEP 1: Har value ka square
        # STEP 2: Squared values ka average
        # STEP 3: Square root = RMS
        # eps -> division by zero avoid karta hai

        x_hat = x / rms
        # STEP 4: Har input ko RMS se divide karo
        # Values ka overall magnitude normalize hota hai

        return np.round(gamma * x_hat, 4).tolist()
        # STEP 5: Gamma se scale karo
        # Final output 4 decimal places tak round