import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        # Step 1: Positive aur negative sentences combine karo
        combined = positive + negative

        # Step 2: Saare unique words nikalo aur alphabetically sort karo
        vocabulary = sorted({
            word
            for sentence in combined
            for word in sentence.split()
        })

        # Step 3: Har word ko ek unique ID do (1 se start)
        # Example: {"bad": 1, "good": 2, "movie": 3, "very": 4}
        word_to_id = {
            word: idx + 1
            for idx, word in enumerate(vocabulary)
        }

        # Step 4: Har sentence ke words ko IDs mein convert karo
        # Example: "good movie" -> [2, 3]
        encoded = [
            torch.tensor([word_to_id[w] for w in s.split()])
            for s in combined
        ]

        # Step 5: Short sentences mein 0 add karo (padding)
        # Example: [1] -> [1, 0]
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)











        