from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegisflow.intelligence.anomaly_autoencoder import (
    AnomalyAutoencoder,
)
from aegisflow.intelligence.threat_estimator import (
    ThreatEstimator,
)


TELEMETRY_PATH = Path(
    "data/telemetry/telemetry_test.csv"
)

ANOMALY_MODEL_PATH = Path(
    "models/intelligence/"
    "aegisflow_anomaly_autoencoder.pt"
)

THREAT_MODEL_PATH = Path(
    "models/intelligence/"
    "aegisflow_threat_estimator_final.pt"
)

OUTPUT_PATH = Path(
    "data/telemetry/"
    "telemetry_test_intelligence.csv"
)


def main():

    print("=" * 78)
    print(
        "AEGISFLOW FINAL TEST AI INTELLIGENCE"
    )
    print("=" * 78)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    df = pd.read_csv(
        TELEMETRY_PATH
    )

    print(
        f"Test telemetry rows: "
        f"{len(df):,}"
    )

    # ========================================================
    # AUTOENCODER
    # ========================================================

    anomaly_checkpoint = torch.load(
        ANOMALY_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    anomaly_features = list(
        anomaly_checkpoint[
            "features"
        ]
    )

    anomaly_model = AnomalyAutoencoder(
        input_dim=int(
            anomaly_checkpoint[
                "input_dim"
            ]
        ),
        latent_dim=int(
            anomaly_checkpoint[
                "latent_dim"
            ]
        ),
    ).to(
        device
    )

    anomaly_model.load_state_dict(
        anomaly_checkpoint[
            "model_state_dict"
        ]
    )

    anomaly_model.eval()

    x_anomaly_np = (
        df[
            anomaly_features
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x_anomaly = torch.as_tensor(
        x_anomaly_np,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        reconstruction = anomaly_model(
            x_anomaly
        )

        reconstruction_error = torch.mean(
            (
                reconstruction
                - x_anomaly
            ) ** 2,
            dim=1,
        )

    reconstruction_error = (
        reconstruction_error
        .cpu()
        .numpy()
    )

    threshold = float(
        anomaly_checkpoint[
            "threshold"
        ]
    )

    high_reference = float(
        anomaly_checkpoint[
            "high_reference"
        ]
    )

    denominator = max(
        high_reference
        - threshold,
        1e-12,
    )

    anomaly_score = np.clip(
        (
            reconstruction_error
            - threshold
        )
        / denominator,
        0.0,
        1.0,
    )

    predicted_anomaly = (
        reconstruction_error
        > threshold
    ).astype(
        np.int64
    )

    # ========================================================
    # THREAT ESTIMATOR
    # ========================================================

    threat_checkpoint = torch.load(
        THREAT_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    threat_features = list(
        threat_checkpoint[
            "features"
        ]
    )

    if "anomaly_score" in threat_features:
        raise ValueError(
            "Final threat estimator must not use anomaly_score."
        )

    threat_model = ThreatEstimator(
        input_dim=int(
            threat_checkpoint[
                "input_dim"
            ]
        )
    ).to(
        device
    )

    threat_model.load_state_dict(
        threat_checkpoint[
            "model_state_dict"
        ]
    )

    threat_model.eval()

    x_threat_np = (
        df[
            threat_features
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x_threat = torch.as_tensor(
        x_threat_np,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        predicted_threat = (
            threat_model(
                x_threat
            )
            .cpu()
            .numpy()
        )

    predicted_threat = np.clip(
        predicted_threat,
        0.0,
        1.0,
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_df = df.copy()

    output_df[
        "reconstruction_error"
    ] = reconstruction_error

    output_df[
        "anomaly_score"
    ] = anomaly_score

    output_df[
        "predicted_anomaly"
    ] = predicted_anomaly

    output_df[
        "predicted_network_threat"
    ] = predicted_threat

    output_df[
        "threat_absolute_error"
    ] = np.abs(
        predicted_threat
        - output_df[
            "reference_network_threat"
        ].to_numpy(
            dtype=np.float64
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Mean anomaly score:",
        f"{anomaly_score.mean():.6f}",
    )

    print(
        "Predicted anomaly rate:",
        f"{predicted_anomaly.mean() * 100:.3f}%",
    )

    print(
        "Mean predicted threat:",
        f"{predicted_threat.mean():.6f}",
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 78)
    print(
        "TEST AI INTELLIGENCE COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()