import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        activation_stats = []
        with torch.no_grad():
            h = x 
            for layer in model:
                h = layer(h)
                if isinstance(layer, nn.Linear):
                    activation_stats.append(
                        {
                            'mean': round(h.mean().item(), 4), 
                            'std': round(h.std().item(), 4), 
                            'dead_fraction': round((h <= 0).all(dim=0).float().mean().item(), 4)
                        }
                    )
        return activation_stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        outputs = model(x)
        loss_criterion = nn.MSELoss()
        mse_loss = loss_criterion(outputs, y)
        mse_loss.backward()
        gradient_stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                    layer_gradient = layer.weight.grad
                    gradient_stats.append(
                        {
                            'mean': round(layer_gradient.mean().item(), 4), 
                            'std': round(layer_gradient.std().item(), 4), 
                            'norm': round(layer_gradient.norm().item(), 4)
                        }
                    )
        return gradient_stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(layer_activation['dead_fraction'] > 0.5 for layer_activation in activation_stats):
            return 'dead_neurons'
        if any(layer_gradient['norm'] > 1000 for layer_gradient in gradient_stats):
            return 'exploding_gradients'
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        if any(layer_activation['std'] < 0.1 for layer_activation in activation_stats):
            return 'vanishing_gradients'
        if any(layer_activation['std'] > 10.0 for layer_activation in activation_stats):
            return 'exploding_gradients'
        return 'healthy'
