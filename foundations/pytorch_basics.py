import torch
import torch.nn
from torchtyping import TensorType

class Solution:

    def reshape(self, to_reshape):
        M, N = to_reshape.shape
        # M = rows, N = columns

        reshaped = torch.reshape(to_reshape, (M * N // 2, 2))
        # Tensor ko 2 columns mein reshape karo
        # Example: (2, 4) -> (4, 2)
        # Total elements same rehte hain: 8

        return torch.round(reshaped, decimals=4)


    def average(self, to_avg):

        averaged = torch.mean(to_avg, dim=0)
        # Har column ka average calculate karo
        # dim=0 -> rows ke across average
        # Example: [[2,4],[6,8]] -> [4,6]

        return torch.round(averaged, decimals=4)


    def concatenate(self, cat_one, cat_two):

        concatenated = torch.cat((cat_one, cat_two), dim=1)
        # Dono tensors ko side-by-side join karo
        # dim=1 -> columns ke along join
        # Example: (2,3) + (2,2) -> (2,5)

        return torch.round(concatenated, decimals=4)


    def get_loss(self, prediction, target):

        loss = torch.nn.functional.mse_loss(prediction, target)
        # Prediction aur actual values ka difference
        # Difference ka square -> sabka average (MSE)
        # Low loss = better prediction

        return torch.round(loss, decimals=4)
