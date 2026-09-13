import numpy as np

from aegisflow.telemetry.normalizer import (
    TelemetryNormalizer,
)


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


class LiveFeatureExtractor:

    def __init__(self):

        self.normalizer = (
            TelemetryNormalizer()
        )

        self.previous_connection_count = 0

    @staticmethod
    def clip01(value):

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    def extract(
        self,
        metrics,
    ):

        packet_rate = (
            self.normalizer
            .normalize_log(
                metrics[
                    "packet_rate_raw"
                ],
                "packet_rate_raw",
            )
        )

        byte_activity = (
            self.normalizer
            .normalize_log(
                metrics[
                    "byte_rate_raw"
                ],
                "byte_rate_raw",
            )
        )

        connection_count = (
            self.normalizer
            .normalize_log(
                metrics[
                    "connection_count"
                ],
                "connection_count",
            )
        )

        unique_destinations = (
            self.normalizer
            .normalize_log(
                metrics[
                    "unique_destinations"
                ],
                "unique_destinations",
            )
        )

        unique_ports = (
            self.normalizer
            .normalize_log(
                metrics[
                    "unique_ports"
                ],
                "unique_ports",
            )
        )

        syn_activity = (
            self.normalizer
            .normalize_log(
                metrics[
                    "syn_count"
                ],
                "syn_count",
            )
        )

        time_wait = (
            self.normalizer
            .normalize_log(
                metrics[
                    "time_wait_count"
                ],
                "time_wait_count",
            )
        )

        new_destinations = (
            self.normalizer
            .normalize_log(
                metrics[
                    "new_destinations"
                ],
                "new_destinations",
            )
        )

        connection_delta = abs(
            metrics[
                "connection_count"
            ]
            - self.previous_connection_count
        )

        connection_burst = (
            self.clip01(
                connection_delta
                / 20.0
            )
        )

        destination_change_rate = (
            self.clip01(
                0.7
                * new_destinations
                + 0.3
                * unique_destinations
            )
        )

        port_scan_score = (
            self.clip01(
                0.45
                * unique_ports
                + 0.35
                * syn_activity
                + 0.20
                * new_destinations
            )
        )

        protocol_deviation = (
            self.clip01(
                0.5
                * syn_activity
                + 0.3
                * time_wait
                + 0.2
                * connection_burst
            )
        )

        retransmission_rate = (
            self.clip01(
                0.55
                * time_wait
                + 0.25
                * syn_activity
                + 0.20
                * connection_burst
            )
        )

        payload_irregularity = (
            self.clip01(
                abs(
                    packet_rate
                    - byte_activity
                )
            )
        )

        device_behavior_shift = (
            self.clip01(
                0.5
                * connection_burst
                + 0.3
                * destination_change_rate
                + 0.2
                * protocol_deviation
            )
        )

        request_frequency = (
            self.clip01(
                0.5
                * packet_rate
                + 0.3
                * connection_count
                + 0.2
                * byte_activity
            )
        )

        # Host-level collector tidak punya
        # authentication log yang reliabel.
        # Untuk prototype awal kita gunakan
        # proxy dari SYN / connection behavior.
        failed_auth_ratio = (
            self.clip01(
                0.65
                * syn_activity
                + 0.35
                * protocol_deviation
            )
        )

        features = {

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
        }

        self.previous_connection_count = (
            metrics[
                "connection_count"
            ]
        )

        return features