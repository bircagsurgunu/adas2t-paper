# adas2t/models.py

import torch
import torch.nn as nn

class MetaLearnerMLP(nn.Module):
    """A simple MLP for predicting WER from acoustic and algorithmic features."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The size of the input feature vector.
            hidden_dim (int): The number of neurons in hidden layers.
            num_layers (int): The number of hidden layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze to remove the last dimension, making the output shape (batch_size,)
        return self.model(x).squeeze(-1)