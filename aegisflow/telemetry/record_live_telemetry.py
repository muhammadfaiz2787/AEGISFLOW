from pathlib import Path
from datetime import datetime

import pandas as pd

from aegisflow.telemetry.collector import (
    LiveNetworkCollector,
)

from aegisflow.telemetry.live_features import (
    LiveFeatureExtractor,
)


OUTPUT_DIR = Path(
    "data/live"
)

INTERVAL_SECONDS = 2.0

NUM_WINDOWS = 300
# 300 x 2 sec = sekitar 10 menit


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    collector = LiveNetworkCollector(
        interval_seconds=INTERVAL_SECONDS
    )

    extractor = LiveFeatureExtractor()

    rows = []

    print("=" * 78)
    print(
        "AEGISFLOW REAL NETWORK TELEMETRY RECORDER"
    )
    print("=" * 78)

    print(
        f"Recording {NUM_WINDOWS} windows "
        f"({NUM_WINDOWS * INTERVAL_SECONDS / 60:.1f} minutes)"
    )

    print()

    for index in range(
        NUM_WINDOWS
    ):

        metrics = collector.collect_window()

        features = extractor.extract(
            metrics
        )

        row = {
            **metrics,
            **features,
        }

        rows.append(
            row
        )

        if (
            index == 0
            or (index + 1) % 10 == 0
        ):

            print(
                f"{index + 1:3d}/{NUM_WINDOWS} | "
                f"packet={metrics['packet_rate_raw']:.2f} pkt/s | "
                f"throughput={metrics['byte_rate_raw']:.2f} B/s"
            )

    df = pd.DataFrame(
        rows
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"real_telemetry_{timestamp}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "RECORDING COMPLETE"
    )
    print("=" * 78)

    print(
        "Saved to:"
    )

    print(
        output_path
    )

    print()

    print(
        "Feature summary:"
    )

    feature_columns = [
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

    print(
        df[
            feature_columns
        ]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
        .round(4)
    )


if __name__ == "__main__":
    main()