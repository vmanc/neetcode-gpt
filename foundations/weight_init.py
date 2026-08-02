import torch
import torch.nn as nn
import math
from typing import List


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        init_matrix = torch.sqrt(torch.tensor(2 / (fan_in + fan_out)))*torch.randn(fan_out, fan_in)
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        return  torch.round(init_matrix, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        init_matrix = torch.sqrt(torch.tensor(2 / fan_in))*torch.randn(fan_out, fan_in)
        return torch.round(init_matrix, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2     torch.manual_seed(0)

        weights_list = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            if init_type == 'xavier':
                w = torch.randn(fan_out, fan_in) * torch.sqrt(torch.tensor(2 / (fan_in + fan_out)))
            elif init_type == 'kaiming':
                w = torch.randn(fan_out, fan_in) * torch.sqrt(torch.tensor(2 / fan_in))
            elif init_type == 'random':
                w = torch.randn(fan_out, fan_in)
            else:
                raise ValueError(f"Initialization Type: {init_type} is not supported")
            weights_list.append(w)

        h = torch.randn(input_dim)

        stds = []
        for w in weights_list:
            h = torch.relu(torch.matmul(w, h))
            stds.append(round(torch.std(h, unbiased=True).item(), 2))
        return stds
