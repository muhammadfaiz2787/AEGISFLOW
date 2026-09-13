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

from aegisflow.intelligence.telemetry_generator import (
    TELEMETRY_FEATURES,
)
from aegisflow.intelligence.threat_estimator import (
    ThreatEstimator,
)


TRAIN_PATH = Path(
    "data/telemetry/telemetry_train_scored.csv"
)

VALIDATION_PATH = Path(
    "data/telemetry/telemetry_validation_scored.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "aegisflow_anomaly_feature_ablation.csv"
)


SEED = 42
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


FEATURE_SETS = {
    "telemetry_only":
        list(TELEMETRY_FEATURES),

    "telemetry_plus_anomaly":
        list(TELEMETRY_FEATURES)
        + [
            "anomaly_score",
        ],
}


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(
    y_true,
    y_pred,
):

    error = (
        y_pred
        - y_true
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    ss_res = float(
        np.sum(
            (
                y_true
                - y_pred
            ) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_true
                - y_true.mean()
            ) ** 2
        )
    )

    r2 = float(
        1.0
        - (
            ss_res
            / max(
                ss_tot,
                1e-12,
            )
        )
    )

    correlation = float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": correlation,
    }


def train_and_evaluate(
    train_df,
    validation_df,
    feature_set_name,
    features,
    device,
):

    set_seed(
        SEED
    )

    print()
    print("=" * 78)
    print(
        f"FEATURE SET: {feature_set_name}"
    )
    print("=" * 78)

    print(
        f"Input dimensions: "
        f"{len(features)}"
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):
        print(
            f"  {index:2d}. {feature}"
        )

    x_train_np = (
        train_df[
            features
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    y_train_np = (
        train_df[
            "reference_network_threat"
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x_validation_np = (
        validation_df[
            features
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    y_validation_np = (
        validation_df[
            "reference_network_threat"
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    dataset = TensorDataset(
        torch.as_tensor(
            x_train_np,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_train_np,
            dtype=torch.float32,
        ),
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

    model = ThreatEstimator(
        input_dim=len(features)
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    for epoch in range(
        EPOCHS
    ):

        model.train()

        losses = []

        for (
            batch_x,
            batch_y,
        ) in loader:

            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad()

            prediction = model(
                batch_x
            )

            loss = criterion(
                prediction,
                batch_y,
            )

            loss.backward()
            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        if (
            epoch == 0
            or (
                epoch + 1
            ) % 20 == 0
        ):

            print(
                f"Epoch "
                f"{epoch + 1:3d}/{EPOCHS} | "
                f"Loss: "
                f"{np.mean(losses):.8f}"
            )

    model.eval()

    x_validation = torch.as_tensor(
        x_validation_np,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        predictions = (
            model(
                x_validation
            )
            .cpu()
            .numpy()
        )

    predictions = np.clip(
        predictions,
        0.0,
        1.0,
    )

    metrics = calculate_metrics(
        y_validation_np,
        predictions,
    )

    print()
    print(
        f"MAE: "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"R2: "
        f"{metrics['r2']:.6f}"
    )

    print(
        f"Correlation: "
        f"{metrics['correlation']:.6f}"
    )

    return {
        "feature_set":
            feature_set_name,

        "state_dim":
            len(features),

        **metrics,
    }


def main():

    print("=" * 78)
    print(
        "AEGISFLOW ANOMALY FEATURE ABLATION"
    )
    print("=" * 78)

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
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

    results = []

    for (
        feature_set_name,
        features,
    ) in FEATURE_SETS.items():

        result = train_and_evaluate(
            train_df=train_df,
            validation_df=validation_df,
            feature_set_name=feature_set_name,
            features=features,
            device=device,
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 78)
    print(
        "ABLATION RESULTS"
    )
    print("=" * 78)

    print(
        results_df.to_string(
            index=False
        )
    )

    telemetry_only = (
        results_df[
            results_df[
                "feature_set"
            ]
            == "telemetry_only"
        ]
        .iloc[0]
    )

    plus_anomaly = (
        results_df[
            results_df[
                "feature_set"
            ]
            == "telemetry_plus_anomaly"
        ]
        .iloc[0]
    )

    print()
    print("=" * 78)
    print(
        "ANOMALY SCORE CONTRIBUTION"
    )
    print("=" * 78)

    print(
        f"Δ MAE: "
        f"{plus_anomaly['mae'] - telemetry_only['mae']:+.6f}"
    )

    print(
        f"Δ RMSE: "
        f"{plus_anomaly['rmse'] - telemetry_only['rmse']:+.6f}"
    )

    print(
        f"Δ R2: "
        f"{plus_anomaly['r2'] - telemetry_only['r2']:+.6f}"
    )

    print(
        f"Δ Correlation: "
        f"{plus_anomaly['correlation'] - telemetry_only['correlation']:+.6f}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()