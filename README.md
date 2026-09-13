# AegisFlow v0.1 — Adaptive Security Policy with Double DQN

A research/demo implementation for AI HackFest 2026.

## Core idea
AegisFlow receives a 14-feature data-flow context and selects one of four simulated security profiles:
LOW, MEDIUM, HIGH, CRITICAL.

The agent optimizes security sufficiency against latency and computational/communication cost.

Important: security scores and overhead values are simulation abstractions, not claims of real-world cryptographic security guarantees.

## State
14 normalized [0,1] features:
sensitivity, confidentiality, integrity, authenticity, privacy,
regulatory_requirement, data_volume, transmission_frequency,
network_threat, destination_trust, device_trust, latency_sensitivity,
replay_freshness, availability_criticality.

## Security requirement
Rs = 0.30*C + 0.25*I + 0.20*O + 0.10*Reg + 0.10*K + 0.05*F

C=confidentiality, I=integrity, O=authenticity,
Reg=regulatory requirement, K=availability/operational criticality,
F=replay/freshness requirement.

## Reward
U = max(0, Rs - Sa)
O = max(0, Sa - Rs)
L = max(0, T/T_deadline - 1)
C_cost = normalized cost

reward = 2 - 12*U^2 - 2*O - 3*L - 0.5*C_cost

## Run
pip install -r requirements.txt
python -m aegisflow.data.generator
python -m aegisflow.training.train
python -m aegisflow.evaluation.evaluate

The v0.1 environment samples independent data-flow contexts per step. Technically this is a contextual-bandit-like problem implemented with a Gymnasium interface; that is more honest than pretending the first prototype is a long-horizon MDP.
