from collections import deque


class TemporalAnomalyState:

    def __init__(
        self,
        window_size=5,
    ):
        self.window_size = int(
            window_size
        )

        self.history = deque(
            maxlen=self.window_size
        )

    def update(
        self,
        predicted_anomaly,
        anomaly_score,
        reconstruction_error,
    ):
        event = {
            "anomaly":
                bool(predicted_anomaly),

            "score":
                float(anomaly_score),

            "reconstruction_error":
                float(reconstruction_error),
        }

        self.history.append(
            event
        )

        anomaly_count = sum(
            1
            for item in self.history
            if item["anomaly"]
        )

        observed_windows = len(
            self.history
        )

        ratio = (
            anomaly_count
            / observed_windows
            if observed_windows > 0
            else 0.0
        )

        # ====================================================
        # TEMPORAL STATE
        # ====================================================

        if anomaly_count <= 1:

            state = "NORMAL"

            severity = 0

        elif anomaly_count == 2:

            state = "OBSERVING"

            severity = 1

        elif anomaly_count < self.window_size:

            state = "WARNING"

            severity = 2

        else:

            state = (
                "PERSISTENT_ANOMALY"
            )

            severity = 3

        mean_score = (
            sum(
                item["score"]
                for item in self.history
            )
            / observed_windows
        )

        max_score = max(
            (
                item["score"]
                for item in self.history
            ),
            default=0.0,
        )

        return {
            "state":
                state,

            "severity":
                severity,

            "anomaly_count":
                anomaly_count,

            "observed_windows":
                observed_windows,

            "window_size":
                self.window_size,

            "anomaly_ratio":
                float(ratio),

            "mean_anomaly_score":
                float(mean_score),

            "max_anomaly_score":
                float(max_score),

            "persistent":
                bool(
                    state
                    == "PERSISTENT_ANOMALY"
                ),
        }

    def reset(
        self,
    ):
        self.history.clear()