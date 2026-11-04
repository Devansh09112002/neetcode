import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        
        selected = embeddings[token_ids]

        # Values ko 5 decimal places tak round karo
        return np.round(selected, 5)