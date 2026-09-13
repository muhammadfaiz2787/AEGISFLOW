from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityCost:
    """
    Baseline transmission cost for each security level.

    latency_ms:
        Baseline additional latency in milliseconds.

    compute_multiplier:
        Relative computational cost.

    bandwidth_multiplier:
        Relative bandwidth usage.
    """

    latency_ms: float
    compute_multiplier: float
    bandwidth_multiplier: float


SECURITY_COSTS = {
    0: SecurityCost(
        latency_ms=1.0,
        compute_multiplier=1.00,
        bandwidth_multiplier=1.05,
    ),

    1: SecurityCost(
        latency_ms=2.0,
        compute_multiplier=1.25,
        bandwidth_multiplier=1.10,
    ),

    2: SecurityCost(
        latency_ms=4.0,
        compute_multiplier=1.70,
        bandwidth_multiplier=1.18,
    ),

    3: SecurityCost(
        latency_ms=7.0,
        compute_multiplier=2.30,
        bandwidth_multiplier=1.30,
    ),
}


def calculate_latency(
    action: int,
    data_volume: float,
    transmission_frequency: float,
) -> float:
    """
    Calculate estimated transmission latency.

    data_volume and transmission_frequency are normalized
    values approximately within [0, 1].
    """

    if action not in SECURITY_COSTS:
        raise ValueError(f"Invalid action: {action}")

    cost = SECURITY_COSTS[action]

    alpha = 0.40
    beta = 0.25

    latency = (
        cost.latency_ms
        * (
            1.0
            + alpha * data_volume
            + beta * transmission_frequency
        )
    )

    return latency


def calculate_compute_cost(
    action: int,
    data_volume: float,
) -> float:
    """
    Calculate normalized computational cost.
    """

    if action not in SECURITY_COSTS:
        raise ValueError(f"Invalid action: {action}")

    cost = SECURITY_COSTS[action]

    return cost.compute_multiplier * (
        0.5 + 0.5 * data_volume
    )


def calculate_bandwidth_cost(
    action: int,
    data_volume: float,
) -> float:
    """
    Calculate normalized bandwidth overhead.
    """

    if action not in SECURITY_COSTS:
        raise ValueError(f"Invalid action: {action}")

    cost = SECURITY_COSTS[action]

    return data_volume * (
        cost.bandwidth_multiplier - 1.0
    )


def calculate_environment_risk(
    network_threat: float,
    device_trust: float,
    destination_trust: float,
) -> float:
    """
    Estimate transmission environment risk.

    Higher network threat increases risk.

    Lower device trust and destination trust
    increase environmental risk.
    """

    risk = (
        0.50 * network_threat
        + 0.25 * (1.0 - device_trust)
        + 0.25 * (1.0 - destination_trust)
    )

    return max(0.0, min(1.0, risk))