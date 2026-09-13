from pathlib import Path

import numpy as np
import pandas as pd


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


class DeploymentCalibrator:

    def __init__(
        self,
        real_reference_path,
        synthetic_reference_path,
    ):

        self.real_df = pd.read_csv(
            real_reference_path
        )

        self.synthetic_df = pd.read_csv(
            synthetic_reference_path
        )

        self._validate()

        self.real_sorted = {}
        self.synthetic_sorted = {}

        for feature in FEATURES:

            self.real_sorted[
                feature
            ] = np.sort(
                self.real_df[
                    feature
                ].to_numpy(
                    dtype=np.float64
                )
            )

            self.synthetic_sorted[
                feature
            ] = np.sort(
                self.synthetic_df[
                    feature
                ].to_numpy(
                    dtype=np.float64
                )
            )

    def _validate(self):

        missing_real = [
            feature
            for feature in FEATURES
            if feature
            not in self.real_df.columns
        ]

        missing_synthetic = [
            feature
            for feature in FEATURES
            if feature
            not in self.synthetic_df.columns
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

    @staticmethod
    def _empirical_quantile(
        sorted_values,
        value,
    ):

        index = np.searchsorted(
            sorted_values,
            value,
            side="right",
        )

        quantile = (
            index
            / len(
                sorted_values
            )
        )

        return float(
            np.clip(
                quantile,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _quantile_value(
        sorted_values,
        quantile,
    ):

        quantile = float(
            np.clip(
                quantile,
                0.0,
                1.0,
            )
        )

        index = int(
            round(
                quantile
                * (
                    len(
                        sorted_values
                    )
                    - 1
                )
            )
        )

        return float(
            sorted_values[
                index
            ]
        )

    def calibrate_feature(
        self,
        feature,
        value,
    ):

        if feature not in FEATURES:

            raise ValueError(
                f"Unknown feature: "
                f"{feature}"
            )

        real_values = (
            self.real_sorted[
                feature
            ]
        )

        synthetic_values = (
            self.synthetic_sorted[
                feature
            ]
        )

        quantile = (
            self._empirical_quantile(
                real_values,
                float(
                    value
                ),
            )
        )

        calibrated = (
            self._quantile_value(
                synthetic_values,
                quantile,
            )
        )

        return float(
            np.clip(
                calibrated,
                0.0,
                1.0,
            )
        )

    def calibrate(
        self,
        features,
    ):

        calibrated = {}

        for feature in FEATURES:

            calibrated[
                feature
            ] = (
                self.calibrate_feature(
                    feature,
                    features[
                        feature
                    ],
                )
            )

        return calibrated