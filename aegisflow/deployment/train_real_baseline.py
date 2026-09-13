from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from aegisflow.deployment.real_baseline_autoencoder import (
    RealBaselineAutoencoder,
)


# ============================================================
# CONFIG
# ============================================================

BASELINE_PATH = Path(
    "data/live/"
    "real_telemetry_20260909_073601.csv"
)

MODEL_PATH = Path(
    "models/deployment/"
    "aegisflow_real_baseline_autoencoder.pt"
)


FEATURES = [
    "packet_rate",
    "failed_auth_ratio",
    "connection_burst",
    "destination_change_rate",
    "port_scan_score",
    "protocol_deviation",
    "retransmission_rate",
    "payload_irregularity",
    "device_behavior_shift",
    "request_frequency",
]


SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3


def set_seed(
    seed,
):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def main():

    print("=" * 78)
    print(
        "AEGISFLOW REAL NETWORK BASELINE TRAINING"
    )
    print("=" * 78)

    set_seed(
        SEED
    )

    if not BASELINE_PATH.exists():

        raise FileNotFoundError(
            BASELINE_PATH
        )

    df = pd.read_csv(
        BASELINE_PATH
    )

    print(
        f"Baseline rows: "
        f"{len(df):,}"
    )

    x_np = (
        df[
            FEATURES
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    if not np.isfinite(
        x_np
    ).all():

        raise ValueError(
            "Non-finite baseline values detected."
        )

    x_np = np.clip(
        x_np,
        0.0,
        1.0,
    )

    x = torch.as_tensor(
        x_np,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        x
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    model = (
        RealBaselineAutoencoder(
            input_dim=len(
                FEATURES
            ),
            latent_dim=4,
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.MSELoss()

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_losses = []

        for (batch,) in loader:

            batch = batch.to(
                device
            )

            optimizer.zero_grad()

            reconstruction = model(
                batch
            )

            loss = criterion(
                reconstruction,
                batch,
            )

            loss.backward()

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch "
                f"{epoch:3d}/{EPOCHS} | "
                f"Loss: "
                f"{np.mean(epoch_losses):.8f}"
            )

    # ========================================================
    # CALIBRATE RECONSTRUCTION ERRORS
    # ========================================================

    model.eval()

    with torch.no_grad():

        reconstruction = model(
            x.to(
                device
            )
        )

        errors = torch.mean(
            (
                reconstruction
                - x.to(
                    device
                )
            ) ** 2,
            dim=1,
        )

    errors = (
        errors
        .cpu()
        .numpy()
    )

    # Prototype baseline threshold.
    #
    # We use baseline percentiles rather than synthetic
    # threshold.
    threshold = float(
        np.quantile(
            errors,
            0.95,
        )
    )

    high_reference = float(
        np.quantile(
            errors,
            0.995,
        )
    )

    if (
        high_reference
        <= threshold
    ):

        high_reference = (
            threshold
            + max(
                float(
                    np.std(
                        errors
                    )
                ),
                1e-6,
            )
        )

    print()
    print("=" * 78)
    print(
        "BASELINE RECONSTRUCTION ERROR"
    )
    print("=" * 78)

    print(
        f"Mean : "
        f"{errors.mean():.8f}"
    )

    print(
        f"Std  : "
        f"{errors.std():.8f}"
    )

    print(
        f"P50  : "
        f"{np.quantile(errors, 0.50):.8f}"
    )

    print(
        f"P90  : "
        f"{np.quantile(errors, 0.90):.8f}"
    )

    print(
        f"P95  : "
        f"{threshold:.8f}"
    )

    print(
        f"P99  : "
        f"{np.quantile(errors, 0.99):.8f}"
    )

    print(
        f"P99.5: "
        f"{high_reference:.8f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "features":
                FEATURES,

            "input_dim":
                len(
                    FEATURES
                ),

            "latent_dim":
                4,

            "threshold":
                threshold,

            "high_reference":
                high_reference,

            "training_rows":
                len(
                    df
                ),

            "baseline_path":
                str(
                    BASELINE_PATH
                ),

            "seed":
                SEED,

            "epochs":
                EPOCHS,
        },
        MODEL_PATH,
    )

    print()
    print("=" * 78)
    print(
        "REAL BASELINE TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        "Saved to:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()