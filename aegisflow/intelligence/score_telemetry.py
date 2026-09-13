from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegisflow.intelligence.anomaly_autoencoder import (
    AnomalyAutoencoder,
)


MODEL_PATH = Path(
    "models/intelligence/"
    "aegisflow_anomaly_autoencoder.pt"
)

TRAIN_PATH = Path(
    "data/telemetry/telemetry_train.csv"
)

VALIDATION_PATH = Path(
    "data/telemetry/telemetry_validation.csv"
)

TRAIN_OUTPUT = Path(
    "data/telemetry/"
    "telemetry_train_scored.csv"
)

VALIDATION_OUTPUT = Path(
    "data/telemetry/"
    "telemetry_validation_scored.csv"
)


def calibrate_score(
    errors,
    threshold,
    high_reference,
):

    denominator = max(
        high_reference
        - threshold,
        1e-12,
    )

    score = (
        errors
        - threshold
    ) / denominator

    return np.clip(
        score,
        0.0,
        1.0,
    )


def score_file(
    input_path,
    output_path,
    model,
    device,
    features,
    threshold,
    high_reference,
):

    df = pd.read_csv(
        input_path
    )

    x_np = (
        df[
            features
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x = torch.as_tensor(
        x_np,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():

        reconstruction = model(
            x
        )

        errors = torch.mean(
            (
                reconstruction
                - x
            ) ** 2,
            dim=1,
        )

    errors = (
        errors
        .cpu()
        .numpy()
    )

    scores = calibrate_score(
        errors,
        threshold,
        high_reference,
    )

    scored = df.copy()

    scored[
        "reconstruction_error"
    ] = errors

    scored[
        "anomaly_score"
    ] = scores

    scored[
        "predicted_anomaly"
    ] = (
        errors
        > threshold
    ).astype(
        np.int64
    )

    scored.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(scored):,} rows:"
    )

    print(
        output_path
    )

    print(
        f"Mean anomaly score: "
        f"{scores.mean():.6f}"
    )

    print()


def main():

    print("=" * 78)
    print(
        "AEGISFLOW TELEMETRY ANOMALY SCORING"
    )
    print("=" * 78)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    features = checkpoint[
        "features"
    ]

    model = AnomalyAutoencoder(
        input_dim=checkpoint[
            "input_dim"
        ],
        latent_dim=checkpoint[
            "latent_dim"
        ],
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    threshold = float(
        checkpoint[
            "threshold"
        ]
    )

    high_reference = float(
        checkpoint[
            "high_reference"
        ]
    )

    print(
        "Device:",
        device,
    )

    print(
        "Threshold:",
        threshold,
    )

    print(
        "High reference:",
        high_reference,
    )

    print()

    score_file(
        TRAIN_PATH,
        TRAIN_OUTPUT,
        model,
        device,
        features,
        threshold,
        high_reference,
    )

    score_file(
        VALIDATION_PATH,
        VALIDATION_OUTPUT,
        model,
        device,
        features,
        threshold,
        high_reference,
    )

    print("=" * 78)
    print(
        "SCORING COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()