import os
import numpy as np
import pandas as pd


SEED = 42
N_SAMPLES = 100_000

OUTPUT_PATH = "data/processed/aegisflow_100k.csv"


# ============================================================
# SECURITY SCENARIO PROFILES
# ============================================================

SCENARIOS = {
    # ========================================================
    # 1. PUBLIC CONTENT
    # Target: dominantly LOW, but allow some MEDIUM cases.
    # ========================================================
    "PUBLIC_CONTENT": {
        "weight": 0.15,

        "confidentiality": (0.05, 0.35),
        "integrity": (0.10, 0.45),
        "authenticity": (0.05, 0.40),
        "privacy": (0.05, 0.35),
        "regulatory_requirement": (0.05, 0.30),
        "availability_criticality": (0.05, 0.40),
        "replay_freshness": (0.05, 0.40),

        "sensitivity": (0.05, 0.40),
        "data_volume": (0.20, 0.90),
        "transmission_frequency": (0.20, 0.90),
        "network_threat": (0.05, 0.50),
        "destination_trust": (0.55, 0.95),
        "device_trust": (0.55, 0.95),
        "latency_sensitivity": (0.40, 0.90),
    },


    # ========================================================
    # 2. IoT TELEMETRY
    # Allow LOW + MEDIUM + small HIGH tail.
    # ========================================================
    "IOT_TELEMETRY": {
        "weight": 0.15,

        "confidentiality": (0.15, 0.55),
        "integrity": (0.30, 0.70),
        "authenticity": (0.25, 0.65),
        "privacy": (0.10, 0.50),
        "regulatory_requirement": (0.10, 0.45),
        "availability_criticality": (0.30, 0.75),
        "replay_freshness": (0.35, 0.85),

        "sensitivity": (0.15, 0.55),
        "data_volume": (0.10, 0.70),
        "transmission_frequency": (0.40, 0.95),
        "network_threat": (0.20, 0.70),
        "destination_trust": (0.40, 0.90),
        "device_trust": (0.35, 0.90),
        "latency_sensitivity": (0.55, 0.95),
    },


    # ========================================================
    # 3. INTERNAL BUSINESS
    # Strong MEDIUM center with LOW and HIGH overlap.
    # ========================================================
    "INTERNAL_BUSINESS": {
        "weight": 0.15,

        "confidentiality": (0.25, 0.75),
        "integrity": (0.35, 0.80),
        "authenticity": (0.30, 0.75),
        "privacy": (0.20, 0.70),
        "regulatory_requirement": (0.15, 0.65),
        "availability_criticality": (0.25, 0.75),
        "replay_freshness": (0.20, 0.70),

        "sensitivity": (0.25, 0.70),
        "data_volume": (0.20, 0.80),
        "transmission_frequency": (0.20, 0.80),
        "network_threat": (0.20, 0.70),
        "destination_trust": (0.45, 0.90),
        "device_trust": (0.45, 0.90),
        "latency_sensitivity": (0.30, 0.80),
    },


    # ========================================================
    # 4. AUTHENTICATED SERVICE
    # Medium/High with overlap into neighboring regimes.
    # ========================================================
    "AUTHENTICATED_SERVICE": {
        "weight": 0.10,

        "confidentiality": (0.35, 0.80),
        "integrity": (0.45, 0.85),
        "authenticity": (0.50, 0.90),
        "privacy": (0.30, 0.75),
        "regulatory_requirement": (0.25, 0.75),
        "availability_criticality": (0.35, 0.85),
        "replay_freshness": (0.40, 0.90),

        "sensitivity": (0.30, 0.80),
        "data_volume": (0.15, 0.75),
        "transmission_frequency": (0.30, 0.90),
        "network_threat": (0.25, 0.75),
        "destination_trust": (0.55, 0.95),
        "device_trust": (0.50, 0.95),
        "latency_sensitivity": (0.45, 0.90),
    },


    # ========================================================
    # 5. PERSONAL DATA
    # Mostly HIGH, but intentionally overlaps MEDIUM.
    # ========================================================
    "PERSONAL_DATA": {
        "weight": 0.15,

        "confidentiality": (0.50, 0.95),
        "integrity": (0.45, 0.85),
        "authenticity": (0.40, 0.85),
        "privacy": (0.60, 0.99),
        "regulatory_requirement": (0.45, 0.95),
        "availability_criticality": (0.30, 0.80),
        "replay_freshness": (0.30, 0.80),

        "sensitivity": (0.55, 0.98),
        "data_volume": (0.10, 0.65),
        "transmission_frequency": (0.15, 0.70),
        "network_threat": (0.30, 0.85),
        "destination_trust": (0.30, 0.85),
        "device_trust": (0.30, 0.85),
        "latency_sensitivity": (0.20, 0.75),
    },


    # ========================================================
    # 6. MEDICAL DATA
    # High/Critical overlap.
    # ========================================================
    "MEDICAL_DATA": {
        "weight": 0.10,

        "confidentiality": (0.65, 0.99),
        "integrity": (0.60, 0.98),
        "authenticity": (0.55, 0.95),
        "privacy": (0.70, 0.99),
        "regulatory_requirement": (0.55, 0.98),
        "availability_criticality": (0.45, 0.95),
        "replay_freshness": (0.45, 0.95),

        "sensitivity": (0.65, 0.99),
        "data_volume": (0.10, 0.70),
        "transmission_frequency": (0.15, 0.75),
        "network_threat": (0.30, 0.90),
        "destination_trust": (0.25, 0.85),
        "device_trust": (0.25, 0.85),
        "latency_sensitivity": (0.40, 0.90),
    },


    # ========================================================
    # 7. FINANCIAL DATA
    # Strong HIGH/CRITICAL overlap.
    # ========================================================
    "FINANCIAL_DATA": {
        "weight": 0.10,

        "confidentiality": (0.60, 0.99),
        "integrity": (0.65, 0.99),
        "authenticity": (0.60, 0.99),
        "privacy": (0.55, 0.97),
        "regulatory_requirement": (0.60, 0.99),
        "availability_criticality": (0.45, 0.95),
        "replay_freshness": (0.55, 0.99),

        "sensitivity": (0.65, 0.99),
        "data_volume": (0.05, 0.60),
        "transmission_frequency": (0.20, 0.80),
        "network_threat": (0.35, 0.95),
        "destination_trust": (0.20, 0.85),
        "device_trust": (0.25, 0.85),
        "latency_sensitivity": (0.50, 0.95),
    },


    # ========================================================
    # 8. CRITICAL OPERATIONAL
    # Strongly CRITICAL, but no longer deterministic.
    # ========================================================
    "CRITICAL_OPERATIONAL": {
        "weight": 0.10,

        "confidentiality": (0.70, 1.00),
        "integrity": (0.75, 1.00),
        "authenticity": (0.70, 1.00),
        "privacy": (0.55, 0.99),
        "regulatory_requirement": (0.60, 1.00),
        "availability_criticality": (0.75, 1.00),
        "replay_freshness": (0.70, 1.00),

        "sensitivity": (0.70, 1.00),
        "data_volume": (0.05, 0.50),
        "transmission_frequency": (0.30, 0.90),
        "network_threat": (0.45, 0.99),
        "destination_trust": (0.15, 0.75),
        "device_trust": (0.20, 0.80),
        "latency_sensitivity": (0.65, 1.00),
    },
}

# ============================================================
# SECURITY REQUIREMENT
# ============================================================

def calculate_security_requirement(df):
    return (
        0.30 * df["confidentiality"]
        + 0.25 * df["integrity"]
        + 0.20 * df["authenticity"]
        + 0.10 * df["regulatory_requirement"]
        + 0.10 * df["availability_criticality"]
        + 0.05 * df["replay_freshness"]
    )


def assign_security_level(rs):
    if rs < 0.45:
        return 0
    elif rs < 0.65:
        return 1
    elif rs < 0.83:
        return 2
    else:
        return 3


# ============================================================
# GENERATOR
# ============================================================

def generate_dataset(n_samples=N_SAMPLES, seed=SEED):

    rng = np.random.default_rng(seed)

    scenario_names = list(SCENARIOS.keys())

    weights = np.array([
        SCENARIOS[name]["weight"]
        for name in scenario_names
    ])

    weights = weights / weights.sum()

    selected_scenarios = rng.choice(
        scenario_names,
        size=n_samples,
        p=weights
    )

    rows = []

    feature_names = [
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

    for scenario in selected_scenarios:

        profile = SCENARIOS[scenario]

        row = {
            "scenario": scenario
        }

        for feature in feature_names:

            low, high = profile[feature]

            value = rng.uniform(low, high)

            row[feature] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    df["security_requirement"] = calculate_security_requirement(df)

    df["required_level"] = df["security_requirement"].apply(
        assign_security_level
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AEGISFLOW DATASET GENERATOR V2")
    print("=" * 60)

    df = generate_dataset()

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nGenerated {len(df):,} samples")

    print("\nScenario distribution:")
    print(
        df["scenario"]
        .value_counts()
        .sort_index()
    )

    print("\nSecurity level distribution:")
    print(
        df["required_level"]
        .value_counts()
        .sort_index()
    )

    print("\nSecurity level percentage:")
    print(
        (
            df["required_level"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(3)
    )

    print("\nSecurity requirement:")
    print(
        df["security_requirement"]
        .describe()
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)