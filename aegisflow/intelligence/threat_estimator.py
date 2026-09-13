import torch
import torch.nn as nn


class ThreatEstimator(nn.Module):

    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                32,
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                16,
            ),
            nn.ReLU(),

            nn.Linear(
                16,
                8,
            ),
            nn.ReLU(),

            nn.Linear(
                8,
                1,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.network(x)
            .squeeze(-1)
        )