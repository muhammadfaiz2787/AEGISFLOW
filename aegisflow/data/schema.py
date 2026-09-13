from dataclasses import dataclass, asdict
from typing import Dict
from aegisflow.config import FEATURES

@dataclass
class DataRecord:
    sensitivity: float
    confidentiality: float
    integrity: float
    authenticity: float
    privacy: float
    regulatory_requirement: float
    data_volume: float
    transmission_frequency: float
    network_threat: float
    destination_trust: float
    device_trust: float
    latency_sensitivity: float
    replay_freshness: float
    availability_criticality: float
    security_requirement: float
    required_level: int

    def state(self):
        return [getattr(self, f) for f in FEATURES]

    def to_dict(self) -> Dict:
        return asdict(self)
