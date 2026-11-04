import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(
        self, seq_len: int, d_model: int
    ) -> NDArray[np.float64]:

        # STEP 1: Empty positional encoding matrix banao
        # seq_len = sentence mein kitne tokens hain
        # d_model = har token ki embedding mein kitne numbers hain
        # Example: seq_len=3, d_model=4
        # Shape: (3, 4) -> 3 positions, har position ke 4 values
        PE = np.zeros((seq_len, d_model))

        # STEP 2: Token positions generate karo
        # np.arange(3) -> [0, 1, 2]
        # reshape(-1, 1) -> [[0], [1], [2]]
        # Column shape banayi taaki broadcasting se calculation ho
        position = np.arange(seq_len).reshape(-1, 1)

        # STEP 3: Different dimensions ke liye frequency scale banao
        # np.arange(0, 4, 2) -> [0, 2]
        # Formula: 10000^(dimension_index / d_model)
        # Example: d_model=4 -> [10000^0, 10000^0.5]
        # Result: [1, 100]
        # Isse different dimensions par sin/cos different speeds se change honge
        div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)

        # STEP 4: EVEN columns mein sine values fill karo
        # 0::2 -> columns 0, 2, 4, ...
        # position / div_term -> har position aur frequency ka combination
        # Example: position=1, div_term=[1,100]
        # sin([1, 0.01]) -> [0.84147, 0.01000]
        PE[:, 0::2] = np.sin(position / div_term)

        # STEP 5: ODD columns mein cosine values fill karo
        # 1::2 -> columns 1, 3, 5, ...
        # Same position/frequency values par cosine apply karo
        # [:PE[:, 1::2].shape[1]] -> odd columns jitne hain utne scales lo
        # Ye odd d_model (jaise 5) ke case mein shape mismatch rokta hai
        PE[:, 1::2] = np.cos(
            position / div_term[:PE[:, 1::2].shape[1]]
        )

        # STEP 6: 5 decimal places tak round karke return karo
        return np.round(PE, 5)