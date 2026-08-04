import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        layers = []
        layers.append(nn.Embedding(vocabulary_size, 16))
        layers.append(nn.Linear(16, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embed_dim = torch.mean(self.network[0](x), dim=1)
        linearized = self.network[1](embed_dim)
        output = torch.round(self.network[2](linearized), decimals=4)
        # Return a B, 1 tensor and round to 4 decimal places
        return output
