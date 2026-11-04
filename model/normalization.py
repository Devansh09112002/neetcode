import numpy as np
from numpy.typing import NDArray

class Solution:
    def forward(self, x, gamma, beta):

        eps = 1e-5
        # Zero se divide hone se bachata hai

        mean = np.mean(x)
        # STEP 1: Input features ka average nikalo

        var = np.var(x)
        # STEP 2: Values mean se kitni spread hain
        # Uska variance calculate karo

        x_norm = (x - mean) / np.sqrt(var + eps)
        # STEP 3: Normalize
        # Mean subtract -> values center around 0
        # Standard deviation se divide -> variance around 1

        out = gamma * x_norm + beta
        # STEP 4: Learnable parameters
        # gamma -> normalized values ko scale karta hai
        # beta -> normalized values ko shift karta hai
        # Network training mein gamma/beta learn karta hai

        return np.round(out, 5)
        # Final output 5 decimal places tak round