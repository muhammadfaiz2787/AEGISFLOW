import numpy as np


# ============================================================
# DEMO DATA PROFILES
#
# These are configurable application-context presets.
# They are NOT outputs of the AI model.
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


def clip01(value):

    return float(
        np.clip(
            float(value),
            0.0,
            1.0,
        )
    )


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
            f"Unknown profile: {profile_name}. "
            f"Available profiles: "
            f"{list(DATA_PROFILES.keys())}"
        )

    profile = DATA_PROFILES[
        profile_name
    ]

    context = {

        "sensitivity":
            profile["sensitivity"],

        "confidentiality":
            profile["confidentiality"],

        "integrity":
            profile["integrity"],

        "authenticity":
            profile["authenticity"],

        "privacy":
            profile["privacy"],

        "regulatory_requirement":
            profile[
                "regulatory_requirement"
            ],

        "data_volume":
            clip01(
                data_volume
            ),

        "transmission_frequency":
            clip01(
                transmission_frequency
            ),

        "network_threat":
            clip01(
                network_threat
            ),

        "destination_trust":
            clip01(
                destination_trust
            ),

        "device_trust":
            clip01(
                device_trust
            ),

        "latency_sensitivity":
            clip01(
                latency_sensitivity
            ),

        "replay_freshness":
            profile[
                "replay_freshness"
            ],

        "availability_criticality":
            profile[
                "availability_criticality"
            ],
    }

    return context