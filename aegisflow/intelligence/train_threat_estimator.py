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
    "data/telemetry/"
    "telemetry_train_scored.csv"
)

VALIDATION_PATH = Path(
    "data/telemetry/"
    "telemetry_validation_scored.csv"
)

MODEL_PATH = Path(
    "models/intelligence/"
    "aegisflow_threat_estimator_final.pt"
)

OUTPUT_PATH = Path(
    "data/telemetry/"
    "telemetry_validation_intelligence_final.csv"
)


SEED = 42

EPOCHS = 80

BATCH_SIZE = 256

LEARNING_RATE = 1e-3


INTELLIGENCE_FEATURES = list(
    list(TELEMETRY_FEATURES)
)


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


def regression_metrics(
    y_true,
    y_pred,
):

    error = (
        y_pred
        - y_true
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
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

    r2 = (
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
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "correlation":
            correlation,
    }


def main():

    set_seed(
        SEED
    )

    print("=" * 78)
    print(
        "AEGISFLOW NEURAL THREAT ESTIMATOR TRAINING"
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

    print(
        "Input features:",
        len(
            INTELLIGENCE_FEATURES
        ),
    )

    for index, feature in enumerate(
        INTELLIGENCE_FEATURES,
        start=1,
    ):

        print(
            f"  {index:2d}. "
            f"{feature}"
        )

    x_train_np = (
        train_df[
            INTELLIGENCE_FEATURES
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
            INTELLIGENCE_FEATURES
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

    print()
    print(
        "Device:",
        device,
    )

    model = ThreatEstimator(
        input_dim=len(
            INTELLIGENCE_FEATURES
        )
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    history = []

    for epoch in range(
        EPOCHS
    ):

        model.train()

        losses = []

        for (
            batch_x,
            batch_y,
        ) in loader:

            batch_x = (
                batch_x.to(
                    device
                )
            )

            batch_y = (
                batch_y.to(
                    device
                )
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

        mean_loss = float(
            np.mean(
                losses
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
    # VALIDATION
    # ========================================================

    model.eval()

    x_validation = (
        torch.as_tensor(
            x_validation_np,
            dtype=torch.float32,
            device=device,
        )
    )

    with torch.no_grad():

        prediction = (
            model(
                x_validation
            )
            .cpu()
            .numpy()
        )

    prediction = np.clip(
        prediction,
        0.0,
        1.0,
    )

    metrics = regression_metrics(
        y_validation_np,
        prediction,
    )

    print()
    print("=" * 78)
    print(
        "VALIDATION RESULTS"
    )
    print("=" * 78)

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

    print()

    print(
        f"Reference threat mean: "
        f"{y_validation_np.mean():.6f}"
    )

    print(
        f"Predicted threat mean: "
        f"{prediction.mean():.6f}"
    )

    # ========================================================
    # SAVE VALIDATION INTELLIGENCE
    # ========================================================

    output_df = (
        validation_df.copy()
    )

    output_df[
        "predicted_network_threat"
    ] = prediction

    output_df[
        "threat_absolute_error"
    ] = np.abs(
        prediction
        - y_validation_np
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
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
                INTELLIGENCE_FEATURES
            ),

        "features":
            INTELLIGENCE_FEATURES,

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "validation_metrics":
            metrics,

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
        "Validation intelligence saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()