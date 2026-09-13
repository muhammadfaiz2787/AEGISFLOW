# AegisFlow — Adaptive Security Intelligence

A research/demo implementation for AI HackFest 2026.

## Core idea
AegisFlow receives a normalized 14-feature data-flow context and selects one of four security profiles:
LOW, MEDIUM, HIGH, CRITICAL.

The policy optimizer balances security sufficiency against latency and computational/communication cost.

Important: the original policy capability scores and overhead values are simulation abstractions, not claims of real-world cryptographic security guarantees.

## State
14 normalized [0,1] features:
sensitivity, confidentiality, integrity, authenticity, privacy,
regulatory_requirement, data_volume, transmission_frequency,
network_threat, destination_trust, device_trust, latency_sensitivity,
replay_freshness, availability_criticality.

## Research positioning
The v0.1 environment samples independent data-flow contexts per step. Technically this is a contextual-bandit-like problem implemented with a Gymnasium interface; this is more accurate than describing the first prototype as a long-horizon MDP.

## Live network monitoring
The deployment prototype includes host-network telemetry, a network-specific anomaly baseline, experimental threat estimation, temporal anomaly smoothing, and live LOW/MEDIUM/HIGH/CRITICAL policy decisions.

The live telemetry collector observes aggregate network metadata/statistics. It does not decrypt arbitrary HTTPS/TLS traffic or inspect unrelated application payloads.

## AegisFlow Secure Transfer
Secure Transfer is the first real enforcement path. Files explicitly submitted to AegisFlow are:

1. analyzed automatically for content context,
2. combined with live network intelligence,
3. evaluated by the existing policy network,
4. mapped to a deployable cryptographic profile, and
5. protected with authenticated encryption.

Current enforcement uses AES-GCM for file data, ephemeral X25519 key agreement, and HKDF-SHA256 key derivation through the `cryptography` library.

### Content Intelligence v2
Content Intelligence v2 fixes an important limitation in the first prototype:

- binary images are no longer decoded as if they were plaintext,
- public posters/flyers/announcements are represented as public content instead of automatically receiving a personal-data bias,
- images can optionally be interpreted by a local zero-shot vision model,
- sensitive text patterns and credential/key formats remain strong signals,
- the UI reports whether local visual intelligence was active or whether the safer metadata/text fallback was used.

The optional local image model does not send image bytes to a cloud API.

Install the normal runtime:

```powershell
pip install -r requirements.txt
```

To enable visual semantic understanding for images:

```powershell
pip install -r requirements-vision.txt
```

The first image analysis may download the configured local model (`openai/clip-vit-base-patch32` by default). After installation, restart the FastAPI backend before testing images again.

You can disable local vision explicitly with:

```powershell
$env:AEGISFLOW_ENABLE_LOCAL_VISION="0"
```

## Run the deployment prototype
Backend:

```powershell
uvicorn backend.main:app --reload
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Live dashboard:

```text
http://localhost:5173/
```

Secure Transfer:

```text
http://localhost:5173/secure.html
```

## Validation note
Held-out policy/threat/anomaly metrics in the dashboard come from the synthetic evaluation pipeline unless explicitly labeled otherwise. Real-network anomaly behavior is network-baseline-specific, and the current threat estimator remains experimental on real traffic.
