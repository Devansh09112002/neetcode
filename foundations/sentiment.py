import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        # Har token ID ko 16 numbers ke vector mein convert karega
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)

        # 16 features ko combine karke 1 score nikalega
        self.linear_layer = nn.Linear(16, 1)

        # Score ko 0 se 1 ke beech convert karega
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:

        # Step 1: Token IDs -> Embedding vectors
        # Shape: (batch, tokens) -> (batch, tokens, 16)
        embeddings = self.embedding_layer(x)

        # Step 2: Har sentence ke token vectors ka average
        # Shape: (batch, tokens, 16) -> (batch, 16)
        averaged = torch.mean(embeddings, dim=1)

        # Step 3: 16 features -> 1 sentiment score
        # Shape: (batch, 16) -> (batch, 1)
        projected = self.linear_layer(averaged)

        # Step 4: Score -> probability (0 to 1)
        return torch.round(self.sigmoid_layer(projected), decimals=4)










        