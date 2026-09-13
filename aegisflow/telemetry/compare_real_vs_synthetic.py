from pathlib import Path

import numpy as np
import pandas as pd


REAL_PATH = Path(
    "data/live/real_telemetry_20260909_073601.csv"
)

SYNTHETIC_PATH = Path(
    "data/telemetry/telemetry_train.csv"
)

OUTPUT_PATH = Path(
    "data/live/real_vs_synthetic_comparison.csv"
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


def summarize(
    df,
    prefix,
):

    rows = []

    for feature in FEATURES:

        values = (
            df[feature]
            .to_numpy(
                dtype=np.float64
            )
        )

        rows.append(
            {
                "feature":
                    feature,

                f"{prefix}_mean":
                    float(
                        np.mean(values)
                    ),

                f"{prefix}_std":
                    float(
                        np.std(values)
                    ),

                f"{prefix}_p05":
                    float(
                        np.quantile(
                            values,
                            0.05,
                        )
                    ),

                f"{prefix}_p25":
                    float(
                        np.quantile(
                            values,
                            0.25,
                        )
                    ),

                f"{prefix}_median":
                    float(
                        np.median(values)
                    ),

                f"{prefix}_p75":
                    float(
                        np.quantile(
                            values,
                            0.75,
                        )
                    ),

                f"{prefix}_p95":
                    float(
                        np.quantile(
                            values,
                            0.95,
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():

    print("=" * 90)
    print(
        "AEGISFLOW REAL VS SYNTHETIC TELEMETRY COMPARISON"
    )
    print("=" * 90)

    if not REAL_PATH.exists():
        raise FileNotFoundError(
            REAL_PATH
        )

    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(
            SYNTHETIC_PATH
        )

    real_df = pd.read_csv(
        REAL_PATH
    )

    synthetic_df = pd.read_csv(
        SYNTHETIC_PATH
    )

    print(
        f"Real rows      : "
        f"{len(real_df):,}"
    )

    print(
        f"Synthetic rows : "
        f"{len(synthetic_df):,}"
    )

    missing_real = [
        feature
        for feature in FEATURES
        if feature not in real_df.columns
    ]

    missing_synthetic = [
        feature
        for feature in FEATURES
        if feature not in synthetic_df.columns
    ]

    if missing_real:
        raise ValueError(
            f"Missing real features: "
            f"{missing_real}"
        )

    if missing_synthetic:
        raise ValueError(
            f"Missing synthetic features: "
            f"{missing_synthetic}"
        )

    real_summary = summarize(
        real_df,
        "real",
    )

    synthetic_summary = summarize(
        synthetic_df,
        "synthetic",
    )

    comparison = real_summary.merge(
        synthetic_summary,
        on="feature",
    )

    comparison[
        "mean_delta"
    ] = (
        comparison[
            "real_mean"
        ]
        - comparison[
            "synthetic_mean"
        ]
    )

    comparison[
        "mean_abs_delta"
    ] = np.abs(
        comparison[
            "mean_delta"
        ]
    )

    comparison[
        "std_ratio_real_to_synthetic"
    ] = (
        comparison[
            "real_std"
        ]
        / np.maximum(
            comparison[
                "synthetic_std"
            ],
            1e-12,
        )
    )

    comparison = (
        comparison
        .sort_values(
            "mean_abs_delta",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        comparison[
            [
                "feature",
                "real_mean",
                "synthetic_mean",
                "mean_delta",
                "real_std",
                "synthetic_std",
                "std_ratio_real_to_synthetic",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "COMPARISON COMPLETE"
    )
    print("=" * 90)

    print(
        "Saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()