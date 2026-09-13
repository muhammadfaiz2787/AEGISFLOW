from aegisflow.telemetry.collector import (
    LiveNetworkCollector,
)

from aegisflow.telemetry.live_features import (
    LiveFeatureExtractor,
)


def main():

    collector = LiveNetworkCollector(
        interval_seconds=2.0
    )

    extractor = LiveFeatureExtractor()

    print("=" * 78)
    print(
        "AEGISFLOW LIVE NETWORK TELEMETRY TEST"
    )
    print("=" * 78)

    print(
        "Monitoring current host network..."
    )

    print(
        "Press CTRL+C to stop."
    )

    print()

    try:

        while True:

            metrics = (
                collector.collect_window()
            )

            features = (
                extractor.extract(
                    metrics
                )
            )

            print("-" * 78)

            print(
                f"Window: "
                f"{metrics['window_seconds']:.2f} s"
            )

            print(
                f"Packet rate raw: "
                f"{metrics['packet_rate_raw']:.2f} pkt/s"
            )

            print(
                f"Byte rate raw: "
                f"{metrics['byte_rate_raw']:.2f} B/s"
            )

            print(
                f"Connections: "
                f"{metrics['connection_count']}"
            )

            print(
                f"Destinations: "
                f"{metrics['unique_destinations']}"
            )

            print()

            for (
                name,
                value,
            ) in features.items():

                print(
                    f"{name:28s}: "
                    f"{value:.4f}"
                )

    except KeyboardInterrupt:

        print()

        print(
            "Monitoring stopped."
        )


if __name__ == "__main__":
    main()