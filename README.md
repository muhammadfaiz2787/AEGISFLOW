# AegisFlow — Adaptive Security Intelligence

AegisFlow is an AI-driven adaptive security prototype for AI HackFest 2026. It combines:

1. live host-network telemetry,
2. network-behaviour anomaly intelligence,
3. automatic file/content context intelligence,
4. a learned contextual security policy optimizer,
5. deployable cryptographic enforcement for explicit file transfers, and
6. a Hermes Agent integration boundary through Model Context Protocol (MCP).

## What AegisFlow does

### Live network intelligence

The local monitor observes aggregate host-network metadata/statistics such as packet
rate, throughput, active connections, destination changes, and derived behavioural
features.

It does **not** decrypt arbitrary HTTPS/TLS traffic or inspect unrelated application
payloads.

The deployment anomaly detector uses a network-specific learned baseline plus temporal
smoothing. The neural threat estimator remains experimental on real traffic; its held-out
metrics in the dashboard come from the synthetic evaluation pipeline.

### Automatic Content Intelligence v3

Files explicitly routed through AegisFlow Secure Transfer are analyzed automatically.
The normal-user Secure Transfer flow does not require a manual General/Personal/
Financial/Medical/IoT selection.

Evidence sources are bounded and local:

- filename and MIME metadata,
- safe plaintext sampling for text files,
- local PDF/DOCX/XLSX/PPTX text extraction,
- local OpenCLIP image semantics when the optional vision extra is installed,
- structured sensitive-data patterns,
- a conservative visual security gate that rejects weak sensitive OpenCLIP matches.

Binary image bytes are never decoded as plaintext.

The UI label **Context Confidence** is currently a heuristic score. It is not a measured
accuracy probability.

### Contextual policy optimizer

AegisFlow converts content/network/trust requirements into the frozen 14-feature policy
state and selects:

- LOW
- MEDIUM
- HIGH
- CRITICAL

The final selected policy model is the gamma=0 contextual neural policy optimizer. The
original environment is contextual-bandit-like, so this positioning is more accurate
than describing the current decision problem as a long-horizon MDP.

### Real cryptographic enforcement

Secure Transfer is the enforcement path. It uses mature primitives from the
`cryptography` package rather than custom cryptography.

Current file-envelope controls include:

- AES-128-GCM for LOW,
- AES-256-GCM for MEDIUM/HIGH/CRITICAL,
- ephemeral X25519 key agreement,
- HKDF-SHA256 key derivation,
- a fresh random content key and nonce per protected file,
- authenticated policy/content-context metadata.

AegisFlow can protect for the local installation or for another AegisFlow device by
using the recipient device's X25519 public key.

Scope: Secure Transfer protects files explicitly submitted to AegisFlow. It does not
transparently intercept and re-encrypt arbitrary Chrome, WhatsApp, banking-app, or other
third-party traffic.

## Hermes Agent integration

AegisFlow is prepared as a local MCP server for Hermes Agent. Hermes acts as the
orchestration/agent layer; AegisFlow remains authoritative for content intelligence,
network intelligence, security-policy selection, and encryption.

The adapter exposes tools for:

- health/status,
- starting/stopping monitoring,
- file analysis,
- adaptive file protection,
- file recovery,
- capability discovery.

The MCP file tools are restricted to `AEGISFLOW_HERMES_ALLOWED_ROOT` and refuse to
overwrite existing output files.

See:

```text
docs/HERMES_INTEGRATION.md
integrations/hermes/config.example.yaml
```

## Quick setup on Windows

From PowerShell in the repository root:

```powershell
.\scripts\setup_windows.ps1
.\scripts\start_windows.ps1
```

The setup script creates/uses `.venv`, installs AegisFlow as an editable package with
local vision + Hermes extras, and installs frontend dependencies.

Open:

```text
Dashboard       http://localhost:5173/
Secure Transfer http://localhost:5173/secure.html
API docs        http://127.0.0.1:8000/docs
```

## Manual setup

Core:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

With local OpenCLIP + Hermes MCP:

```powershell
pip install -e ".[vision,hermes]"
```

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

## Content-intelligence validation

Create local samples under:

```text
data/content_validation/
  public/
  general/
  personal/
  financial/
  medical/
  credentials/
  iot/
```

Then run:

```powershell
python -m aegisflow.evaluation.evaluate_content_intelligence
```

The evaluator reports accuracy, macro F1, per-class metrics, a confusion matrix,
`false_sensitive_escalation_rate`, and `sensitive_miss_rate`.

Do not commit real private documents to this public repository. Use redacted, synthetic,
or explicitly authorized samples for sensitive classes.

## Research/evaluation status

### Frozen synthetic policy/threat evaluation

The dashboard reports the already-frozen held-out synthetic test results. These numbers
must not be presented as real-world network accuracy.

### Real-network deployment

- aggregate live telemetry: implemented,
- network-specific baseline autoencoder: implemented,
- temporal anomaly smoothing: implemented,
- real-world attack ground-truth validation: still pending,
- threat-estimator real-world validation: still pending.

### Content Intelligence

The validation harness is implemented. A representative real/redacted validation corpus
still needs to be populated before claiming measured real-world content-classification
accuracy.

## Key project structure

```text
aegisflow/
  context/          automatic content intelligence
  deployment/       real-network anomaly baseline
  environment/      policy reward/cost abstractions
  evaluation/       synthetic + content-intelligence evaluation
  integrations/     Hermes MCP bridge
  intelligence/     anomaly/threat models
  security/         security profiles + authenticated file envelope
  service/          live and secure-transfer orchestration
  telemetry/        host-network collection and feature extraction

backend/             FastAPI + WebSocket API
frontend/            React/Vite/Tailwind dashboard + Secure Transfer
integrations/hermes/ Hermes config example
scripts/             Windows setup/launcher
```

## Important scientific wording

AegisFlow is a research/hackathon prototype, not a replacement for TLS, an IDS, or a
production endpoint security suite. Real-world claims should distinguish:

- synthetic held-out evaluation,
- network-specific deployment adaptation,
- experimental real-network threat estimation,
- locally enforced Secure Transfer cryptography.
