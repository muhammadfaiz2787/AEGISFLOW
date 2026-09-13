from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


SCORED_PATH = Path(
    "data/telemetry/"
    "telemetry_validation_scored.csv"
)


def main():

    print("=" * 78)
    print(
        "AEGISFLOW ANOMALY DETECTOR EVALUATION"
    )
    print("=" * 78)

    if not SCORED_PATH.exists():
        raise FileNotFoundError(
            SCORED_PATH
        )

    df = pd.read_csv(
        SCORED_PATH
    )

    required_columns = [
        "reference_anomaly",
        "reference_network_threat",
        "reconstruction_error",
        "anomaly_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    y_true = (
        df[
            "reference_anomaly"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    score = (
        df[
            "anomaly_score"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    error = (
        df[
            "reconstruction_error"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    threat = (
        df[
            "reference_network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # RANKING METRICS
    # ========================================================

    roc_auc = roc_auc_score(
        y_true,
        error,
    )

    pr_auc = average_precision_score(
        y_true,
        error,
    )

    # ========================================================
    # SCORE SEPARATION
    # ========================================================

    normal_score = score[
        y_true == 0
    ]

    anomaly_score = score[
        y_true == 1
    ]

    normal_error = error[
        y_true == 0
    ]

    anomaly_error = error[
        y_true == 1
    ]

    # ========================================================
    # CORRELATION
    # ========================================================

    corr_score_threat = (
        np.corrcoef(
            score,
            threat,
        )[0, 1]
    )

    corr_error_threat = (
        np.corrcoef(
            error,
            threat,
        )[0, 1]
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Reference anomaly rate: "
        f"{y_true.mean() * 100:.2f}%"
    )

    print()

    print(
        f"ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{pr_auc:.6f}"
    )

    print()
    print(
        "ANOMALY SCORE SEPARATION"
    )

    print(
        f"Normal mean score: "
        f"{normal_score.mean():.6f}"
    )

    print(
        f"Anomaly mean score: "
        f"{anomaly_score.mean():.6f}"
    )

    print(
        f"Normal median score: "
        f"{np.median(normal_score):.6f}"
    )

    print(
        f"Anomaly median score: "
        f"{np.median(anomaly_score):.6f}"
    )

    print()

    print(
        "RECONSTRUCTION ERROR SEPARATION"
    )

    print(
        f"Normal mean error: "
        f"{normal_error.mean():.8f}"
    )

    print(
        f"Anomaly mean error: "
        f"{anomaly_error.mean():.8f}"
    )

    print()

    print(
        "CORRELATION WITH "
        "REFERENCE NETWORK THREAT"
    )

    print(
        f"Anomaly score correlation: "
        f"{corr_score_threat:.6f}"
    )

    print(
        f"Reconstruction error correlation: "
        f"{corr_error_threat:.6f}"
    )

    # ========================================================
    # BEST F1 THRESHOLD, EVALUATION ONLY
    # ========================================================

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            error,
        )
    )

    f1_scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(
            precision[:-1]
            + recall[:-1],
            1e-12,
        )
    )

    best_index = int(
        np.argmax(
            f1_scores
        )
    )

    best_threshold = float(
        thresholds[
            best_index
        ]
    )

    best_f1 = float(
        f1_scores[
            best_index
        ]
    )

    print()

    print(
        "BEST VALIDATION THRESHOLD "
        "(DIAGNOSTIC ONLY)"
    )

    print(
        f"Best error threshold: "
        f"{best_threshold:.8f}"
    )

    print(
        f"Best validation F1: "
        f"{best_f1:.6f}"
    )


if __name__ == "__main__":
    main()