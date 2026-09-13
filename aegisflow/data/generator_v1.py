import numpy as np
import pandas as pd
from aegisflow.config import DATA_DIR

N = 100_000
SEED = 42

def beta(rng, a, b, n):
    return rng.beta(a, b, size=n).astype(np.float32)

def clip01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def generate_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    sensitivity = beta(rng, 2.2, 2.0, n)
    confidentiality = clip01(0.35*sensitivity + 0.65*beta(rng, 2.5, 2.0, n))
    integrity = clip01(0.25*sensitivity + 0.75*beta(rng, 2.3, 2.1, n))
    authenticity = clip01(0.20*sensitivity + 0.80*beta(rng, 2.0, 2.2, n))
    privacy = clip01(0.30*sensitivity + 0.70*beta(rng, 2.2, 2.0, n))

    regulatory_requirement = clip01(
        0.55*privacy + 0.45*beta(rng, 1.8, 2.6, n)
    )

    data_volume = beta(rng, 1.5, 4.0, n)
    transmission_frequency = beta(rng, 1.8, 3.0, n)
    network_threat = beta(rng, 2.0, 2.4, n)

    destination_trust = clip01(
        0.65*beta(rng, 4.0, 1.8, n) + 0.35*(1.0-network_threat)
    )
    device_trust = clip01(
        0.65*beta(rng, 4.2, 1.8, n) + 0.35*(1.0-network_threat)
    )

    latency_sensitivity = beta(rng, 2.1, 2.3, n)
    replay_freshness = beta(rng, 2.0, 2.0, n)

    availability_criticality = clip01(
        0.45*sensitivity + 0.30*integrity
        + 0.25*beta(rng, 2.4, 2.0, n)
    )

    rs = (
        0.30*confidentiality
        + 0.25*integrity
        + 0.20*authenticity
        + 0.10*regulatory_requirement
        + 0.10*availability_criticality
        + 0.05*replay_freshness
    )

    # Evaluation label only; required_level is NOT fed to the agent.
    required_level = np.select(
        [rs < 0.45, rs < 0.65, rs < 0.83],
        [0, 1, 2],
        default=3,
    ).astype(np.int64)

    return pd.DataFrame({
        "sensitivity": sensitivity,
        "confidentiality": confidentiality,
        "integrity": integrity,
        "authenticity": authenticity,
        "privacy": privacy,
        "regulatory_requirement": regulatory_requirement,
        "data_volume": data_volume,
        "transmission_frequency": transmission_frequency,
        "network_threat": network_threat,
        "destination_trust": destination_trust,
        "device_trust": device_trust,
        "latency_sensitivity": latency_sensitivity,
        "replay_freshness": replay_freshness,
        "availability_criticality": availability_criticality,
        "security_requirement": rs.astype(np.float32),
        "required_level": required_level,
    })

if __name__ == "__main__":
    out = DATA_DIR / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    path = out / "aegisflow_100k.csv"
    df.to_csv(path, index=False)
    print(f"Generated {len(df):,} rows -> {path}")
    print(df["required_level"].value_counts(normalize=True).sort_index())
