from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRAIN_CONTEXT_PATH = Path(
    "data/processed/splits/train.csv"
)

VALIDATION_CONTEXT_PATH = Path(
    "data/processed/splits/validation.csv"
)

OUTPUT_DIR = Path(
    "data/telemetry"
)


# ============================================================
# TELEMETRY FEATURES
# ============================================================

TELEMETRY_FEATURES = [
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


# ============================================================
# HELPER
# ============================================================

def clip01(x):
    return np.clip(
        x,
        0.0,
        1.0,
    )


# ============================================================
# GENERATOR
# ============================================================

def generate_telemetry(
    context_df,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    n = len(
        context_df
    )

    threat = (
        context_df[
            "network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    device_trust = (
        context_df[
            "device_trust"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    destination_trust = (
        context_df[
            "destination_trust"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    transmission_frequency = (
        context_df[
            "transmission_frequency"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    data_volume = (
        context_df[
            "data_volume"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    authenticity = (
        context_df[
            "authenticity"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # Random noise
    # --------------------------------------------------------

    noise = lambda scale: rng.normal(
        loc=0.0,
        scale=scale,
        size=n,
    )

    # --------------------------------------------------------
    # Simulated raw-ish telemetry
    # --------------------------------------------------------

    packet_rate = clip01(
        0.45 * transmission_frequency
        + 0.30 * data_volume
        + 0.20 * threat
        + noise(0.08)
    )

    failed_auth_ratio = clip01(
        0.50 * threat
        + 0.25 * (1.0 - device_trust)
        + 0.15 * (1.0 - authenticity)
        + noise(0.07)
    )

    connection_burst = clip01(
        0.45 * threat
        + 0.35 * transmission_frequency
        + noise(0.09)
    )

    destination_change_rate = clip01(
        0.45 * threat
        + 0.35 * (1.0 - destination_trust)
        + noise(0.08)
    )

    port_scan_score = clip01(
        0.70 * threat
        + 0.15 * (1.0 - destination_trust)
        + noise(0.08)
    )

    protocol_deviation = clip01(
        0.60 * threat
        + 0.20 * (1.0 - device_trust)
        + noise(0.09)
    )

    retransmission_rate = clip01(
        0.35 * threat
        + 0.30 * data_volume
        + 0.20 * transmission_frequency
        + noise(0.08)
    )

    payload_irregularity = clip01(
        0.50 * threat
        + 0.25 * data_volume
        + noise(0.09)
    )

    device_behavior_shift = clip01(
        0.45 * threat
        + 0.40 * (1.0 - device_trust)
        + noise(0.07)
    )

    request_frequency = clip01(
        0.55 * transmission_frequency
        + 0.25 * threat
        + 0.10 * data_volume
        + noise(0.08)
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    telemetry_df = pd.DataFrame(
        {
            "packet_rate":
                packet_rate,

            "failed_auth_ratio":
                failed_auth_ratio,

            "connection_burst":
                connection_burst,

            "destination_change_rate":
                destination_change_rate,

            "port_scan_score":
                port_scan_score,

            "protocol_deviation":
                protocol_deviation,

            "retransmission_rate":
                retransmission_rate,

            "payload_irregularity":
                payload_irregularity,

            "device_behavior_shift":
                device_behavior_shift,

            "request_frequency":
                request_frequency,

            # Evaluation metadata only
            "reference_network_threat":
                threat,

            "reference_device_trust":
                device_trust,

            "reference_destination_trust":
                destination_trust,
        }
    )

    # --------------------------------------------------------
    # Reference anomaly label
    #
    # IMPORTANT:
    # NOT used for Autoencoder training.
    # Only used for offline validation.
    # --------------------------------------------------------

    telemetry_df[
        "reference_anomaly"
    ] = (
        (
            threat >= 0.70
        )
        | (
            (
                threat >= 0.55
            )
            & (
                device_trust <= 0.35
            )
        )
        | (
            (
                threat >= 0.55
            )
            & (
                destination_trust <= 0.35
            )
        )
    ).astype(
        np.int64
    )

    return telemetry_df


# ============================================================
# SAVE
# ============================================================

def save_split(
    input_path,
    output_path,
    seed,
):

    print(
        "Loading:",
        input_path,
    )

    context_df = pd.read_csv(
        input_path
    )

    telemetry_df = (
        generate_telemetry(
            context_df=context_df,
            seed=seed,
        )
    )

    telemetry_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(telemetry_df):,} rows to:"
    )

    print(
        output_path
    )

    print(
        "Reference anomaly distribution:"
    )

    print(
        telemetry_df[
            "reference_anomaly"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "AEGISFLOW TELEMETRY GENERATOR"
    )
    print("=" * 78)

    save_split(
        input_path=TRAIN_CONTEXT_PATH,
        output_path=(
            OUTPUT_DIR
            / "telemetry_train.csv"
        ),
        seed=42,
    )

    save_split(
        input_path=VALIDATION_CONTEXT_PATH,
        output_path=(
            OUTPUT_DIR
            / "telemetry_validation.csv"
        ),
        seed=84,
    )

    print("=" * 78)
    print(
        "TELEMETRY GENERATION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()