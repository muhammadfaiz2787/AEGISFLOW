from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegisflow.config import (
    FEATURES,
    TrainConfig,
)

from aegisflow.service.anomaly_state import (
    TemporalAnomalyState,
)

from aegisflow.deployment.real_baseline_autoencoder import (
    RealBaselineAutoencoder,
)

from aegisflow.intelligence.threat_estimator import (
    ThreatEstimator,
)

from aegisflow.rl.networks import (
    QNetwork,
)

from aegisflow.telemetry.collector import (
    LiveNetworkCollector,
)

from aegisflow.telemetry.live_features import (
    LiveFeatureExtractor,
)

from aegisflow.telemetry.normalizer import (
    TelemetryNormalizer,
)

from aegisflow.service.context_builder import (
    build_policy_context,
)


# ============================================================
# MODEL PATHS
# ============================================================

REAL_ANOMALY_MODEL_PATH = Path(
    "models/deployment/"
    "aegisflow_real_baseline_autoencoder.pt"
)

THREAT_MODEL_PATH = Path(
    "models/intelligence/"
    "aegisflow_threat_estimator_final.pt"
)

POLICY_MODEL_PATH = Path(
    "models/"
    "aegisflow_ddqn_full_gamma0p0_seed42.pt"
)


POLICY_NAMES = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL",
}


class AegisFlowLiveService:

    def __init__(
        self,
        interval_seconds=2.0,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.collector = (
            LiveNetworkCollector(
                interval_seconds=interval_seconds
            )
        )

        self.extractor = (
            LiveFeatureExtractor()
        )

        self.normalizer = (
            TelemetryNormalizer()
        )

        self.temporal_anomaly = (
            TemporalAnomalyState(
                window_size=5
            )
        )

        self._load_anomaly_model()
        self._load_threat_model()
        self._load_policy_model()

    # ========================================================
    # LOAD REAL-NETWORK ANOMALY MODEL
    # ========================================================

    def _load_anomaly_model(
        self,
    ):

        checkpoint = torch.load(
            REAL_ANOMALY_MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.anomaly_features = list(
            checkpoint[
                "features"
            ]
        )

        self.anomaly_threshold = float(
            checkpoint[
                "threshold"
            ]
        )

        self.anomaly_high_reference = float(
            checkpoint[
                "high_reference"
            ]
        )

        self.anomaly_model = (
            RealBaselineAutoencoder(
                input_dim=int(
                    checkpoint[
                        "input_dim"
                    ]
                ),
                latent_dim=int(
                    checkpoint[
                        "latent_dim"
                    ]
                ),
            )
            .to(
                self.device
            )
        )

        self.anomaly_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.anomaly_model.eval()

        print(
            "Real-network anomaly baseline loaded."
        )

        print(
            f"Anomaly threshold: "
            f"{self.anomaly_threshold:.8f}"
        )

    # ========================================================
    # LOAD THREAT ESTIMATOR
    # ========================================================

    def _load_threat_model(
        self,
    ):

        checkpoint = torch.load(
            THREAT_MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.threat_features = list(
            checkpoint[
                "features"
            ]
        )

        if (
            "anomaly_score"
            in self.threat_features
        ):

            raise ValueError(
                "Final Threat Estimator "
                "must not use anomaly_score."
            )

        self.threat_model = (
            ThreatEstimator(
                input_dim=int(
                    checkpoint[
                        "input_dim"
                    ]
                )
            )
            .to(
                self.device
            )
        )

        self.threat_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.threat_model.eval()

    # ========================================================
    # LOAD POLICY MODEL
    # ========================================================

    def _load_policy_model(
        self,
    ):

        checkpoint = torch.load(
            POLICY_MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        if not np.isclose(
            float(
                checkpoint[
                    "gamma"
                ]
            ),
            0.0,
            atol=1e-12,
        ):

            raise ValueError(
                "Expected final gamma = 0."
            )

        if (
            int(
                checkpoint[
                    "state_dim"
                ]
            )
            != len(
                FEATURES
            )
        ):

            raise ValueError(
                "Policy state dimension mismatch."
            )

        if (
            list(
                checkpoint[
                    "features"
                ]
            )
            != list(
                FEATURES
            )
        ):

            raise ValueError(
                "Policy feature ordering mismatch."
            )

        config = TrainConfig()

        self.policy_model = (
            QNetwork(
                state_dim=len(
                    FEATURES
                ),
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
            )
            .to(
                self.device
            )
        )

        self.policy_model.load_state_dict(
            checkpoint[
                "online_net"
            ]
        )

        self.policy_model.eval()

    # ========================================================
    # REAL-NETWORK ANOMALY INFERENCE
    # ========================================================

    def infer_anomaly(
        self,
        telemetry_features,
    ):

        x_np = np.asarray(
            [
                telemetry_features[
                    feature
                ]
                for feature
                in self.anomaly_features
            ],
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        x = torch.as_tensor(
            x_np,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            reconstruction = (
                self.anomaly_model(
                    x
                )
            )

            reconstruction_error_tensor = (
                torch.mean(
                    (
                        reconstruction
                        - x
                    ) ** 2,
                    dim=1,
                )
            )

        reconstruction_error = float(
            reconstruction_error_tensor[
                0
            ].item()
        )

        denominator = max(
            self.anomaly_high_reference
            - self.anomaly_threshold,
            1e-12,
        )

        anomaly_score = float(
            np.clip(
                (
                    reconstruction_error
                    - self.anomaly_threshold
                )
                / denominator,
                0.0,
                1.0,
            )
        )

        predicted_anomaly = bool(
            reconstruction_error
            > self.anomaly_threshold
        )

        return {
            "anomaly_score":
                anomaly_score,

            "reconstruction_error":
                reconstruction_error,

            "predicted_anomaly":
                predicted_anomaly,
        }

    # ========================================================
    # THREAT ESTIMATOR
    # ========================================================

    def infer_threat(
        self,
        telemetry_features,
    ):

        x_np = np.asarray(
            [
                telemetry_features[
                    feature
                ]
                for feature
                in self.threat_features
            ],
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        x = torch.as_tensor(
            x_np,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            predicted_threat = (
                self.threat_model(
                    x
                )
                .cpu()
                .numpy()
                .reshape(
                    -1
                )
            )

        return float(
            np.clip(
                predicted_threat[
                    0
                ],
                0.0,
                1.0,
            )
        )

    # ========================================================
    # POLICY INFERENCE
    # ========================================================

    def infer_policy(
        self,
        context,
    ):

        row = pd.DataFrame(
            [
                context
            ]
        )

        x_np = (
            row[
                FEATURES
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        x = torch.as_tensor(
            x_np,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            q_values = (
                self.policy_model(
                    x
                )
                .cpu()
                .numpy()[0]
            )

        action = int(
            np.argmax(
                q_values
            )
        )

        return {
            "action":
                action,

            "policy":
                POLICY_NAMES[
                    action
                ],

            "q_values": {
                POLICY_NAMES[i]:
                    float(
                        q_values[i]
                    )
                for i in range(
                    len(
                        q_values
                    )
                )
            },
        }

    # ========================================================
    # COMPLETE LIVE WINDOW
    # ========================================================

    def run_once(
        self,
        profile_name="general",
        device_trust=0.75,
        destination_trust=0.75,
        latency_sensitivity=0.50,
    ):

        # ----------------------------------------------------
        # COLLECT LIVE NETWORK DATA
        # ----------------------------------------------------

        metrics = (
            self.collector
            .collect_window()
        )

        # ----------------------------------------------------
        # EXTRACT REAL TELEMETRY FEATURES
        # ----------------------------------------------------

        telemetry = (
            self.extractor
            .extract(
                metrics
            )
        )

        # ----------------------------------------------------
        # REAL BASELINE ANOMALY DETECTION
        # ----------------------------------------------------

        anomaly = (
            self.infer_anomaly(
                telemetry
            )
        )

        temporal_anomaly = (
            self.temporal_anomaly.update(
                predicted_anomaly=
                    anomaly[
                        "predicted_anomaly"
                    ],

                anomaly_score=
                    anomaly[
                        "anomaly_score"
                    ],

                reconstruction_error=
                    anomaly[
                        "reconstruction_error"
                    ],
            )
        )

        # ----------------------------------------------------
        # EXPERIMENTAL THREAT ESTIMATION
        # ----------------------------------------------------

        threat_score = (
            self.infer_threat(
                telemetry
            )
        )

        # ----------------------------------------------------
        # DYNAMIC POLICY CONTEXT
        # ----------------------------------------------------

        data_volume = (
            self.normalizer
            .normalize_log(
                metrics[
                    "byte_rate_raw"
                ],
                "byte_rate_raw",
            )
        )

        transmission_frequency = (
            telemetry[
                "request_frequency"
            ]
        )

        # ----------------------------------------------------
        # BUILD POLICY CONTEXT
        # ----------------------------------------------------

        policy_context = (
            build_policy_context(
                profile_name=
                    profile_name,

                network_threat=
                    threat_score,

                data_volume=
                    data_volume,

                transmission_frequency=
                    transmission_frequency,

                device_trust=
                    device_trust,

                destination_trust=
                    destination_trust,

                latency_sensitivity=
                    latency_sensitivity,
            )
        )

        # ----------------------------------------------------
        # POLICY DECISION
        # ----------------------------------------------------

        policy = (
            self.infer_policy(
                policy_context
            )
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        return {
            "metrics":
                metrics,

            "telemetry":
                telemetry,

            "intelligence": {
                "threat_score":
                    threat_score,

                "threat_status":
                    "experimental",

                "anomaly_score":
                    anomaly[
                        "anomaly_score"
                    ],

                "reconstruction_error":
                    anomaly[
                        "reconstruction_error"
                    ],

                "predicted_anomaly":
                    anomaly[
                        "predicted_anomaly"
                    ],

                "anomaly_threshold":
                    self.anomaly_threshold,

                "anomaly_high_reference":
                    self.anomaly_high_reference,

                "anomaly_mode":
                    "real_network_baseline",

                "temporal_anomaly":
                    temporal_anomaly,
            },

            "policy_context":
                policy_context,

            "decision":
                policy,
        }