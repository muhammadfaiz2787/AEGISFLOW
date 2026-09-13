import torch
from torch import nn


class RealBaselineAutoencoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=10,
        latent_dim=4,
    ):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
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
                latent_dim,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                8,
            ),
            nn.ReLU(),

            nn.Linear(
                8,
                16,
            ),
            nn.ReLU(),

            nn.Linear(
                16,
                input_dim,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):

        latent = self.encoder(
            x
        )

        reconstruction = (
            self.decoder(
                latent
            )
        )

        return reconstruction