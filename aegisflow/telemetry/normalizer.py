import numpy as np


class TelemetryNormalizer:

    def __init__(self):

        self.scale = {

            "packet_rate_raw":
                1000.0,

            "byte_rate_raw":
                5_000_000.0,

            "connection_count":
                100.0,

            "unique_destinations":
                50.0,

            "unique_ports":
                30.0,

            "syn_count":
                20.0,

            "time_wait_count":
                50.0,

            "new_destinations":
                20.0,
        }

    @staticmethod
    def clip01(value):

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    def normalize_linear(
        self,
        value,
        key,
    ):

        scale = float(
            self.scale[key]
        )

        return self.clip01(
            float(value)
            / max(
                scale,
                1e-12,
            )
        )

    def normalize_log(
        self,
        value,
        key,
    ):

        scale = float(
            self.scale[key]
        )

        value = max(
            float(value),
            0.0,
        )

        return self.clip01(
            np.log1p(
                value
            )
            / np.log1p(
                scale
            )
        )