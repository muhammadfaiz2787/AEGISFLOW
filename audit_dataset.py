import pandas as pd
import numpy as np

DATA_PATH = "data/processed/aegisflow_100k.csv"

df = pd.read_csv(DATA_PATH)

features = [
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


print("=" * 70)
print("AEGISFLOW DATASET V2 — FORMAL AUDIT")
print("=" * 70)


# ============================================================
# 1. BASIC DATASET
# ============================================================

print("\n[1] DATASET SHAPE")
print(df.shape)


print("\n[2] COLUMNS")
print(df.columns.tolist())


print("\n[3] MISSING VALUES")
print(df.isnull().sum().sum())


# ============================================================
# 2. FEATURE RANGE
# ============================================================

print("\n[4] FEATURE RANGE")

feature_range = pd.DataFrame({
    "min": df[features].min(),
    "max": df[features].max(),
    "mean": df[features].mean(),
    "std": df[features].std()
})

print(feature_range)


# ============================================================
# 3. SECURITY LEVEL DISTRIBUTION
# ============================================================

print("\n[5] REQUIRED LEVEL DISTRIBUTION")

level_counts = (
    df["required_level"]
    .value_counts()
    .sort_index()
)

print(level_counts)


print("\n[6] REQUIRED LEVEL PERCENTAGE")

level_percent = (
    df["required_level"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(level_percent.round(3))


# ============================================================
# 4. SECURITY REQUIREMENT
# ============================================================

print("\n[7] SECURITY REQUIREMENT STATISTICS")

print(
    df["security_requirement"].describe()
)


# ============================================================
# 5. FORMULA VALIDATION
# ============================================================

print("\n[8] SECURITY REQUIREMENT FORMULA VALIDATION")

expected_rs = (
    0.30 * df["confidentiality"]
    + 0.25 * df["integrity"]
    + 0.20 * df["authenticity"]
    + 0.10 * df["regulatory_requirement"]
    + 0.10 * df["availability_criticality"]
    + 0.05 * df["replay_freshness"]
)

error = (
    expected_rs
    - df["security_requirement"]
).abs()

print("Maximum absolute error:", error.max())
print("Mean absolute error:", error.mean())


# ============================================================
# 6. LEVEL THRESHOLD VALIDATION
# ============================================================

print("\n[9] SECURITY LEVEL THRESHOLD VALIDATION")

invalid_low = df[
    (df["required_level"] == 0)
    & (df["security_requirement"] >= 0.45)
]

invalid_medium = df[
    (df["required_level"] == 1)
    & (
        (df["security_requirement"] < 0.45)
        | (df["security_requirement"] >= 0.65)
    )
]

invalid_high = df[
    (df["required_level"] == 2)
    & (
        (df["security_requirement"] < 0.65)
        | (df["security_requirement"] >= 0.83)
    )
]

invalid_critical = df[
    (df["required_level"] == 3)
    & (df["security_requirement"] < 0.83)
]

print("Invalid LOW:", len(invalid_low))
print("Invalid MEDIUM:", len(invalid_medium))
print("Invalid HIGH:", len(invalid_high))
print("Invalid CRITICAL:", len(invalid_critical))


# ============================================================
# 7. SECURITY REQUIREMENT BY LEVEL
# ============================================================

print("\n[10] SECURITY REQUIREMENT BY LEVEL")

level_stats = (
    df.groupby("required_level")["security_requirement"]
    .agg(["count", "min", "mean", "max"])
)

print(level_stats)


# ============================================================
# 8. SCENARIO DISTRIBUTION
# ============================================================

print("\n[11] SCENARIO DISTRIBUTION")

print(
    df["scenario"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. SCENARIO × SECURITY LEVEL
# ============================================================

print("\n[12] SCENARIO × SECURITY LEVEL")

scenario_level = pd.crosstab(
    df["scenario"],
    df["required_level"],
    normalize="index"
) * 100

print(
    scenario_level.round(2)
)


# ============================================================
# 10. SCENARIO SECURITY REQUIREMENT
# ============================================================

print("\n[13] SECURITY REQUIREMENT BY SCENARIO")

scenario_rs = (
    df.groupby("scenario")["security_requirement"]
    .agg(["count", "mean", "std", "min", "max"])
    .sort_values("mean")
)

print(
    scenario_rs.round(4)
)


# ============================================================
# 11. IMPORTANT SECURITY FEATURES BY LEVEL
# ============================================================

security_features = [
    "confidentiality",
    "integrity",
    "authenticity",
    "regulatory_requirement",
    "availability_criticality",
    "replay_freshness",
]

print("\n[14] SECURITY FEATURES BY LEVEL")

security_by_level = (
    df.groupby("required_level")[security_features]
    .mean()
)

print(
    security_by_level.round(4)
)


# ============================================================
# 12. CORRELATION WITH SECURITY REQUIREMENT
# ============================================================

print("\n[15] FEATURE CORRELATION WITH SECURITY REQUIREMENT")

correlation = (
    df[features + ["security_requirement"]]
    .corr()["security_requirement"]
    .drop("security_requirement")
    .sort_values(ascending=False)
)

print(
    correlation.round(4)
)


# ============================================================
# 13. SANITY CHECK
# ============================================================

print("\n[16] SANITY CHECK")

# CRITICAL should generally have high security features
critical = df[df["required_level"] == 3]

print(
    "CRITICAL samples:",
    len(critical)
)

print(
    "\nCRITICAL feature means:"
)

print(
    critical[security_features]
    .mean()
    .round(4)
)


# LOW should generally have lower security features
low = df[df["required_level"] == 0]

print(
    "\nLOW feature means:"
)

print(
    low[security_features]
    .mean()
    .round(4)
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)