from aegisflow.service.live_intelligence import (
    AegisFlowLiveService,
)


def main():

    print("=" * 78)
    print("AEGISFLOW LIVE AI")
    print("=" * 78)

    service = AegisFlowLiveService(
        interval_seconds=2.0
    )

    print(
        "Models loaded successfully."
    )

    print(
        "Monitoring current network..."
    )

    print(
        "Press CTRL+C to stop."
    )

    print()

    try:

        while True:

            result = service.run_once(
                profile_name="general",
                device_trust=0.75,
                destination_trust=0.75,
                latency_sensitivity=0.50,
            )

            metrics = result[
                "metrics"
            ]

            telemetry = result[
                "telemetry"
            ]

            intelligence = result[
                "intelligence"
            ]

            policy_context = result[
                "policy_context"
            ]

            decision = result[
                "decision"
            ]

            print(
                "=" * 78
            )

            print(
                f"Window              : "
                f"{metrics['window_seconds']:.2f} s"
            )

            print(
                f"Packet Rate         : "
                f"{metrics['packet_rate_raw']:.2f} pkt/s"
            )

            print(
                f"Network Throughput  : "
                f"{metrics['byte_rate_raw']:.2f} B/s"
            )

            print(
                f"Connections         : "
                f"{metrics['connection_count']}"
            )

            print(
                f"Destinations        : "
                f"{metrics['unique_destinations']}"
            )

            print()

            print(
                f"Threat Score        : "
                f"{intelligence['threat_score']:.4f} "
                f"(experimental)"
            )

            print(
                f"Anomaly Score       : "
                f"{intelligence['anomaly_score']:.4f}"
            )

            print(
                f"Reconstruction Error: "
                f"{intelligence['reconstruction_error']:.8f}"
            )

            print(
                f"Anomaly Threshold   : "
                f"{intelligence['anomaly_threshold']:.8f}"
            )

            print(
                f"Anomaly Detected    : "
                f"{intelligence['predicted_anomaly']}"
            )

            print(
                f"Anomaly Mode        : "
                f"{intelligence['anomaly_mode']}"
            )

            print()

            print(
                "Telemetry:"
            )

            for (
                name,
                value,
            ) in telemetry.items():

                print(
                    f"  {name:28s}: "
                    f"{value:.4f}"
                )

            print()

            print(
                "Policy Context:"
            )

            for (
                name,
                value,
            ) in policy_context.items():

                print(
                    f"  {name:28s}: "
                    f"{value:.4f}"
                )

            print()

            print(
                f"Recommended Policy  : "
                f"{decision['policy']}"
            )

            print()

            print(
                "Policy Q-values:"
            )

            for (
                policy_name,
                value,
            ) in (
                decision[
                    "q_values"
                ].items()
            ):

                print(
                    f"  {policy_name:10s}: "
                    f"{value:.4f}"
                )

            print()

    except KeyboardInterrupt:

        print()

        print(
            "AegisFlow monitoring stopped."
        )


if __name__ == "__main__":
    main()