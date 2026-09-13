import numpy as np


# ============================================================
# LEGACY / DEBUG PRESETS
#
# These remain available for simulation and developer override.
# Secure Transfer uses automatic content requirements instead.
# ============================================================

DATA_PROFILES = {
    "general": {
        "sensitivity": 0.45,
        "confidentiality": 0.50,
        "integrity": 0.55,
        "authenticity": 0.50,
        "privacy": 0.45,
        "regulatory_requirement": 0.30,
        "replay_freshness": 0.40,
        "availability_criticality": 0.50,
    },
    "personal": {
        "sensitivity": 0.65,
        "confidentiality": 0.70,
        "integrity": 0.65,
        "authenticity": 0.65,
        "privacy": 0.80,
        "regulatory_requirement": 0.60,
        "replay_freshness": 0.55,
        "availability_criticality": 0.55,
    },
    "financial": {
        "sensitivity": 0.90,
        "confidentiality": 0.90,
        "integrity": 0.95,
        "authenticity": 0.95,
        "privacy": 0.85,
        "regulatory_requirement": 0.90,
        "replay_freshness": 0.90,
        "availability_criticality": 0.85,
    },
    "medical": {
        "sensitivity": 0.90,
        "confidentiality": 0.95,
        "integrity": 0.90,
        "authenticity": 0.85,
        "privacy": 0.95,
        "regulatory_requirement": 0.95,
        "replay_freshness": 0.70,
        "availability_criticality": 0.90,
    },
    "iot": {
        "sensitivity": 0.55,
        "confidentiality": 0.50,
        "integrity": 0.80,
        "authenticity": 0.80,
        "privacy": 0.45,
        "regulatory_requirement": 0.40,
        "replay_freshness": 0.85,
        "availability_criticality": 0.80,
    },
}


REQUIREMENT_KEYS = (
    "sensitivity",
    "confidentiality",
    "integrity",
    "authenticity",
    "privacy",
    "regulatory_requirement",
    "replay_freshness",
    "availability_criticality",
)


def clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def build_policy_context_from_requirements(
    requirements,
    network_threat,
    data_volume,
    transmission_frequency,
    device_trust=0.75,
    destination_trust=0.75,
    latency_sensitivity=0.50,
):
    missing = [key for key in REQUIREMENT_KEYS if key not in requirements]
    if missing:
        raise ValueError(f"Missing automatic content requirements: {missing}")

    return {
        "sensitivity": clip01(requirements["sensitivity"]),
        "confidentiality": clip01(requirements["confidentiality"]),
        "integrity": clip01(requirements["integrity"]),
        "authenticity": clip01(requirements["authenticity"]),
        "privacy": clip01(requirements["privacy"]),
        "regulatory_requirement": clip01(requirements["regulatory_requirement"]),
        "data_volume": clip01(data_volume),
        "transmission_frequency": clip01(transmission_frequency),
        "network_threat": clip01(network_threat),
        "destination_trust": clip01(destination_trust),
        "device_trust": clip01(device_trust),
        "latency_sensitivity": clip01(latency_sensitivity),
        "replay_freshness": clip01(requirements["replay_freshness"]),
        "availability_criticality": clip01(requirements["availability_criticality"]),
    }


def build_policy_context(
    profile_name,
    network_threat,
    data_volume,
    transmission_frequency,
    device_trust=0.75,
    destination_trust=0.75,
    latency_sensitivity=0.50,
):
    if profile_name not in DATA_PROFILES:
        raise ValueError(
            f"Unknown profile: {profile_name}. Available profiles: {list(DATA_PROFILES.keys())}"
        )

    return build_policy_context_from_requirements(
        requirements=DATA_PROFILES[profile_name],
        network_threat=network_threat,
        data_volume=data_volume,
        transmission_frequency=transmission_frequency,
        device_trust=device_trust,
        destination_trust=destination_trust,
        latency_sensitivity=latency_sensitivity,
    )
