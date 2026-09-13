import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Neural network that estimates Q(s, a)
    for all security actions.

    Input:
        state vector with 14 features.

    Output:
        4 Q-values:
        [LOW, MEDIUM, HIGH, CRITICAL]
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)