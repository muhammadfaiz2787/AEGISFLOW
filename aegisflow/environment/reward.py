import pandas as pd

from aegisflow.environment.policy import get_policy
from aegisflow.environment.cost_model import (
    calculate_latency,
    calculate_compute_cost,
    calculate_bandwidth_cost,
    calculate_environment_risk,
)


# ============================================================
# SECURITY REQUIREMENT WEIGHTS
# ============================================================

REQUIREMENT_WEIGHTS = {
    "confidentiality": 0.25,
    "integrity": 0.20,
    "authenticity": 0.15,
    "privacy": 0.15,
    "regulatory_requirement": 0.10,
    "replay_freshness": 0.05,
    "availability_criticality": 0.10,
}


# ============================================================
# REWARD WEIGHTS
# ============================================================

FULFILLMENT_WEIGHT = 4.0
SECURITY_RISK_WEIGHT = 8.0
LATENCY_WEIGHT = 2.0
COMPUTE_WEIGHT = 1.0
BANDWIDTH_WEIGHT = 1.0


# ============================================================
# SECURITY FULFILLMENT
# ============================================================

def calculate_security_fulfillment(
    state: pd.Series,
    action: int,
) -> float:
    """
    Calculate how well a security action satisfies
    the multidimensional security requirements.

    Result is normalized approximately to [0, 1].
    """

    policy = get_policy(action)

    capability = {
        "confidentiality": policy.confidentiality,
        "integrity": policy.integrity,
        "authenticity": policy.authenticity,
        "privacy": policy.privacy,
        "regulatory_requirement": policy.regulatory_requirement,
        "replay_freshness": policy.replay_freshness,
        "availability_criticality": policy.availability_criticality,
    }

    fulfillment = 0.0

    for feature, weight in REQUIREMENT_WEIGHTS.items():

        requirement = float(state[feature])

        if requirement <= 0:
            component = 1.0

        else:
            component = min(
                capability[feature] / requirement,
                1.0,
            )

        fulfillment += weight * component

    return fulfillment


# ============================================================
# SECURITY VIOLATION
# ============================================================

def calculate_security_violation(
    state: pd.Series,
    action: int,
) -> float:
    """
    Calculate weighted unmet security requirements.

    Higher value means greater security deficiency.
    """

    policy = get_policy(action)

    capability = {
        "confidentiality": policy.confidentiality,
        "integrity": policy.integrity,
        "authenticity": policy.authenticity,
        "privacy": policy.privacy,
        "regulatory_requirement": policy.regulatory_requirement,
        "replay_freshness": policy.replay_freshness,
        "availability_criticality": policy.availability_criticality,
    }

    violation = 0.0

    for feature, weight in REQUIREMENT_WEIGHTS.items():

        requirement = float(state[feature])

        gap = max(
            requirement - capability[feature],
            0.0,
        )

        violation += weight * gap

    return violation


# ============================================================
# TOTAL REWARD
# ============================================================

def calculate_reward(
    state: pd.Series,
    action: int,
) -> dict:
    """
    Calculate complete reward for one state-action pair.
    """

    # --------------------------------------------------------
    # 1. Security fulfillment
    # --------------------------------------------------------

    fulfillment = calculate_security_fulfillment(
        state,
        action,
    )

    # --------------------------------------------------------
    # 2. Security violation
    # --------------------------------------------------------

    violation = calculate_security_violation(
        state,
        action,
    )

    # --------------------------------------------------------
    # 3. Environmental risk
    # --------------------------------------------------------

    environment_risk = calculate_environment_risk(
        float(state["network_threat"]),
        float(state["device_trust"]),
        float(state["destination_trust"]),
    )

    # Security violation becomes more costly
    # in dangerous environments.

    risk = violation * (
        1.0 + 2.0 * environment_risk
    )

    # --------------------------------------------------------
    # 4. Latency
    # --------------------------------------------------------

    latency_ms = calculate_latency(
        action,
        float(state["data_volume"]),
        float(state["transmission_frequency"]),
    )

    latency_penalty = (
        latency_ms / 10.0
        * float(state["latency_sensitivity"])
    )

    # --------------------------------------------------------
    # 5. Compute cost
    # --------------------------------------------------------

    compute_cost = calculate_compute_cost(
        action,
        float(state["data_volume"]),
    )

    # --------------------------------------------------------
    # 6. Bandwidth cost
    # --------------------------------------------------------

    bandwidth_cost = calculate_bandwidth_cost(
        action,
        float(state["data_volume"]),
    )

    # --------------------------------------------------------
    # 7. Final reward
    # --------------------------------------------------------

    reward = (
        FULFILLMENT_WEIGHT * fulfillment
        - SECURITY_RISK_WEIGHT * risk
        - LATENCY_WEIGHT * latency_penalty
        - COMPUTE_WEIGHT * compute_cost
        - BANDWIDTH_WEIGHT * bandwidth_cost
    )

    return {
        "reward": reward,
        "fulfillment": fulfillment,
        "violation": violation,
        "environment_risk": environment_risk,
        "risk": risk,
        "latency_ms": latency_ms,
        "latency_penalty": latency_penalty,
        "compute_cost": compute_cost,
        "bandwidth_cost": bandwidth_cost,
    }