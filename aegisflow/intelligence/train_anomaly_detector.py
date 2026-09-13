from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from aegisflow.intelligence.anomaly_autoencoder import (
    AnomalyAutoencoder,
)
from aegisflow.intelligence.telemetry_generator import (
    TELEMETRY_FEATURES,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = Path(
    "data/telemetry/telemetry_train.csv"
)

VALIDATION_PATH = Path(
    "data/telemetry/telemetry_validation.csv"
)

MODEL_PATH = Path(
    "models/intelligence/"
    "aegisflow_anomaly_autoencoder.pt"
)

SCORE_OUTPUT_PATH = Path(
    "data/telemetry/"
    "telemetry_validation_scored.csv"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

LATENT_DIM = 4

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

EPOCHS = 80

NORMAL_THREAT_MAX = 0.45

NORMAL_DEVICE_TRUST_MIN = 0.55

NORMAL_DESTINATION_TRUST_MIN = 0.55


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# DATA
# ============================================================

def load_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            VALIDATION_PATH
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    missing_train = [
        feature
        for feature in TELEMETRY_FEATURES
        if feature not in train_df.columns
    ]

    missing_validation = [
        feature
        for feature in TELEMETRY_FEATURES
        if feature not in validation_df.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing train telemetry features: "
            f"{missing_train}"
        )

    if missing_validation:
        raise ValueError(
            f"Missing validation telemetry features: "
            f"{missing_validation}"
        )

    return (
        train_df,
        validation_df,
    )


# ============================================================
# NORMAL TRAINING SUBSET
# ============================================================

def select_normal_training_data(
    train_df,
):

    mask = (
        (
            train_df[
                "reference_network_threat"
            ]
            <= NORMAL_THREAT_MAX
        )
        &
        (
            train_df[
                "reference_device_trust"
            ]
            >= NORMAL_DEVICE_TRUST_MIN
        )
        &
        (
            train_df[
                "reference_destination_trust"
            ]
            >= NORMAL_DESTINATION_TRUST_MIN
        )
    )

    normal_df = (
        train_df[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(normal_df) < 1000:
        raise ValueError(
            "Normal training subset too small. "
            f"Only {len(normal_df)} rows found."
        )

    return normal_df


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

def reconstruction_error(
    model,
    x,
):

    model.eval()

    with torch.no_grad():

        reconstructed = model(
            x
        )

        errors = torch.mean(
            (
                reconstructed
                - x
            ) ** 2,
            dim=1,
        )

    return (
        errors
        .cpu()
        .numpy()
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_anomaly_score(
    errors,
    threshold,
    high_reference,
):

    denominator = max(
        high_reference
        - threshold,
        1e-12,
    )

    normalized = (
        errors
        - threshold
    ) / denominator

    score = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return score


# ============================================================
# TRAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print("=" * 78)
    print(
        "AEGISFLOW NEURAL ANOMALY DETECTOR TRAINING"
    )
    print("=" * 78)

    (
        train_df,
        validation_df,
    ) = load_data()

    print(
        f"Train telemetry rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation telemetry rows: "
        f"{len(validation_df):,}"
    )

    normal_train_df = (
        select_normal_training_data(
            train_df
        )
    )

    print(
        f"Normal training rows: "
        f"{len(normal_train_df):,}"
    )

    print(
        f"Normal percentage: "
        f"{len(normal_train_df) / len(train_df) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Training tensor
    # --------------------------------------------------------

    x_train_np = (
        normal_train_df[
            TELEMETRY_FEATURES
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x_train = torch.as_tensor(
        x_train_np,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        x_train
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AnomalyAutoencoder(
        input_dim=len(
            TELEMETRY_FEATURES
        ),
        latent_dim=LATENT_DIM,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    history = []

    for epoch in range(
        EPOCHS
    ):

        model.train()

        epoch_losses = []

        for (
            batch,
        ) in loader:

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

        mean_loss = float(
            np.mean(
                epoch_losses
            )
        )

        history.append(
            mean_loss
        )

        if (
            epoch == 0
            or (
                epoch + 1
            ) % 5 == 0
        ):

            print(
                f"Epoch "
                f"{epoch + 1:3d}/{EPOCHS} | "
                f"Loss: "
                f"{mean_loss:.8f}"
            )

    # ========================================================
    # CALIBRATION USING NORMAL TRAINING DATA
    # ========================================================

    x_normal_device = torch.as_tensor(
        x_train_np,
        dtype=torch.float32,
        device=device,
    )

    normal_errors = (
        reconstruction_error(
            model,
            x_normal_device,
        )
    )

    threshold = float(
        np.quantile(
            normal_errors,
            0.95,
        )
    )

    high_reference = float(
        np.quantile(
            normal_errors,
            0.999,
        )
    )

    if (
        high_reference
        <= threshold
    ):
        high_reference = (
            threshold
            + 1e-6
        )

    print()
    print("=" * 78)
    print(
        "CALIBRATION"
    )
    print("=" * 78)

    print(
        f"Normal reconstruction error mean: "
        f"{normal_errors.mean():.8f}"
    )

    print(
        f"Normal reconstruction error std: "
        f"{normal_errors.std():.8f}"
    )

    print(
        f"95th percentile threshold: "
        f"{threshold:.8f}"
    )

    print(
        f"99.9th percentile reference: "
        f"{high_reference:.8f}"
    )

    # ========================================================
    # VALIDATION SCORING
    # ========================================================

    x_validation_np = (
        validation_df[
            TELEMETRY_FEATURES
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x_validation = (
        torch.as_tensor(
            x_validation_np,
            dtype=torch.float32,
            device=device,
        )
    )

    validation_errors = (
        reconstruction_error(
            model,
            x_validation,
        )
    )

    anomaly_scores = (
        calibrate_anomaly_score(
            errors=validation_errors,
            threshold=threshold,
            high_reference=high_reference,
        )
    )

    scored_df = (
        validation_df
        .copy()
    )

    scored_df[
        "reconstruction_error"
    ] = (
        validation_errors
    )

    scored_df[
        "anomaly_score"
    ] = (
        anomaly_scores
    )

    scored_df[
        "predicted_anomaly"
    ] = (
        validation_errors
        > threshold
    ).astype(
        np.int64
    )

    SCORE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_df.to_csv(
        SCORE_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # SIMPLE VALIDATION METRICS
    # ========================================================

    y_true = (
        scored_df[
            "reference_anomaly"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    y_pred = (
        scored_df[
            "predicted_anomaly"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    accuracy = (
        (tp + tn)
        / len(y_true)
    )

    precision = (
        tp
        / max(
            tp + fp,
            1,
        )
    )

    recall = (
        tp
        / max(
            tp + fn,
            1,
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision
            + recall,
            1e-12,
        )
    )

    print()
    print("=" * 78)
    print(
        "VALIDATION SUMMARY"
    )
    print("=" * 78)

    print(
        "Confusion matrix:"
    )

    print(
        f"TN: {tn:,} | "
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,} | "
        f"TP: {tp:,}"
    )

    print()

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"F1: "
        f"{f1:.4f}"
    )

    print()

    print(
        f"Mean anomaly score: "
        f"{anomaly_scores.mean():.4f}"
    )

    print(
        f"Median anomaly score: "
        f"{np.median(anomaly_scores):.4f}"
    )

    print(
        f"P95 anomaly score: "
        f"{np.quantile(anomaly_scores, 0.95):.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            len(
                TELEMETRY_FEATURES
            ),

        "latent_dim":
            LATENT_DIM,

        "features":
            list(
                TELEMETRY_FEATURES
            ),

        "threshold":
            threshold,

        "high_reference":
            high_reference,

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "normal_threat_max":
            NORMAL_THREAT_MAX,

        "normal_device_trust_min":
            NORMAL_DEVICE_TRUST_MIN,

        "normal_destination_trust_min":
            NORMAL_DESTINATION_TRUST_MIN,

        "training_history":
            history,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print()
    print("=" * 78)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Scored validation telemetry saved to:"
    )

    print(
        SCORE_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()