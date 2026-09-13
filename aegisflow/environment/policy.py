from dataclasses import dataclass
from typing import Dict


LOW = 0
MEDIUM = 1
HIGH = 2
CRITICAL = 3


LEVEL_NAMES = {
    LOW: "LOW",
    MEDIUM: "MEDIUM",
    HIGH: "HIGH",
    CRITICAL: "CRITICAL",
}


@dataclass(frozen=True)
class SecurityPolicy:
    """
    Normalized security capability for each security level.

    These values represent simulation capabilities,
    NOT real-world probabilities of security.
    """

    confidentiality: float
    integrity: float
    authenticity: float
    privacy: float
    regulatory_requirement: float
    replay_freshness: float
    availability_criticality: float


SECURITY_POLICIES: Dict[int, SecurityPolicy] = {

    LOW: SecurityPolicy(
        confidentiality=0.35,
        integrity=0.30,
        authenticity=0.35,
        privacy=0.25,
        regulatory_requirement=0.25,
        replay_freshness=0.10,
        availability_criticality=0.30,
    ),

    MEDIUM: SecurityPolicy(
        confidentiality=0.60,
        integrity=0.60,
        authenticity=0.65,
        privacy=0.55,
        regulatory_requirement=0.55,
        replay_freshness=0.50,
        availability_criticality=0.55,
    ),

    HIGH: SecurityPolicy(
        confidentiality=0.82,
        integrity=0.85,
        authenticity=0.88,
        privacy=0.80,
        regulatory_requirement=0.85,
        replay_freshness=0.85,
        availability_criticality=0.78,
    ),

    CRITICAL: SecurityPolicy(
        confidentiality=0.96,
        integrity=0.97,
        authenticity=0.98,
        privacy=0.95,
        regulatory_requirement=0.98,
        replay_freshness=0.98,
        availability_criticality=0.92,
    ),
}


def get_policy(action: int) -> SecurityPolicy:
    """Return security policy for an action."""

    if action not in SECURITY_POLICIES:
        raise ValueError(f"Invalid security action: {action}")

    return SECURITY_POLICIES[action]


def get_level_name(action: int) -> str:
    """Convert action number into security level name."""

    if action not in LEVEL_NAMES:
        raise ValueError(f"Invalid security action: {action}")

    return LEVEL_NAMES[action]