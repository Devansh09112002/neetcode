import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):

    def __init__(self):
        super().__init__()
        # Parent nn.Module initialize karo
        # PyTorch layers aur parameters track kar sakega

        torch.manual_seed(0)
        # Random initialization reproducible banane ke liye

        self.first_linear = nn.Linear(784, 512)
        # 784 pixel inputs -> 512 hidden neurons
        # Har neuron weighted sum + bias calculate karega
        # Weights aur biases training mein learn honge

        self.relu = nn.ReLU()
        # Negative activations -> 0
        # Non-linearity introduce karta hai

        self.dropout = nn.Dropout(p=0.2)
        # Training mein randomly 20% activations zero
        # Overfitting reduce karne mein help karta hai
        # Inference (eval mode) mein dropout disabled

        self.projection = nn.Linear(512, 10)
        # 512 hidden features -> 10 output scores
        # Har score ek digit (0-9) se correspond karta hai

        self.sigmoid = nn.Sigmoid()
        # Har output ko 0 aur 1 ke beech convert karta hai
        # Scores independent hain, sum=1 zaroori nahi

    def forward(self, images):

        torch.manual_seed(0)
        # Question mein reproducible results ke liye
        # Normally forward mein seed reset nahi karte

        x = self.first_linear(images)
        # (batch_size, 784) -> (batch_size, 512)

        x = self.relu(x)
        # Hidden layer par ReLU apply

        x = self.dropout(x)
        # Training mein 20% activations randomly zero

        x = self.projection(x)
        # (batch_size, 512) -> (batch_size, 10)

        x = self.sigmoid(x)
        # 10 outputs ko 0-1 range mein convert

        return torch.round(x, decimals=4)
        # Final outputs ko 4 decimals tak round