from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


FEATURES = [
    "sensitivity",
    "confidentiality",
    "integrity",
    "authenticity",
    "privacy",
    "regulatory_requirement",
    "data_volume",
    "transmission_frequency",
    "network_threat",
    "destination_trust",
    "device_trust",
    "latency_sensitivity",
    "replay_freshness",
    "availability_criticality",
]

FEATURE_SETS = {
    "full": FEATURES,

    "no-threat": [
        feature
        for feature in FEATURES
        if feature != "network_threat"
    ],

    "no-trust": [
        feature
        for feature in FEATURES
        if feature not in {
            "destination_trust",
            "device_trust",
        }
    ],

    "no-latency": [
        feature
        for feature in FEATURES
        if feature != "latency_sensitivity"
    ],
}

def get_feature_set(name):

    if name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {name}. "
            f"Available: {list(FEATURE_SETS.keys())}"
        )

    return FEATURE_SETS[name]


ACTIONS = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]


# Simulator-only security scores.
SECURITY_SCORE = [
    0.445,
    0.680,
    0.863,
    0.990,
]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    state_dim: int = 14
    action_dim: int = 4
    hidden_dim: int = 128

    gamma: float = 0.95
    lr: float = 1e-3

    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay: int = 2_000

    episodes: int = 2000
    steps_per_episode: int = 32

    target_update_every: int = 250

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 20_000.0