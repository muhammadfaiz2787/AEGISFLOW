# Hermes Agent integration

AegisFlow is prepared as an external **MCP tool server** for Hermes Agent.

This is intentionally an orchestration boundary: Hermes can decide *when* to ask
AegisFlow to monitor, analyze, protect, or recover a file, while AegisFlow remains
the authority for network intelligence, content context, policy selection, and
cryptographic enforcement. Hermes does not replace the AegisFlow policy model.

## Why MCP

Hermes Agent supports local stdio MCP servers and discovers their tools at startup.
AegisFlow therefore exposes a small, typed tool surface rather than coupling the
application to Hermes internals.

## Install

From the AegisFlow virtual environment:

```powershell
pip install -r requirements-hermes.txt
```

Run AegisFlow first:

```powershell
uvicorn backend.main:app --reload
```

Copy the `mcp_servers.aegisflow` block from:

```text
integrations/hermes/config.example.yaml
```

into `~/.hermes/config.yaml`, adjusting paths.

Then test it from Hermes:

```text
hermes mcp test aegisflow
```

or restart/reload Hermes MCP discovery.

## Exposed tools

- `aegisflow_health` — read-only backend health.
- `aegisflow_status` — read-only live-monitoring state/latest intelligence.
- `aegisflow_capabilities` — read-only integration/secure-transfer capabilities.
- `aegisflow_start_monitoring` — starts aggregate host-network monitoring.
- `aegisflow_stop_monitoring` — stops monitoring.
- `aegisflow_analyze_file` — read-only automatic content/policy analysis.
- `aegisflow_protect_file` — writes an adaptively encrypted `.aegis` envelope.
- `aegisflow_unprotect_file` — recovers a local recipient envelope.

Hermes will expose these with its MCP server prefix, typically
`mcp_aegisflow_<tool_name>`.

## File-system safety boundary

The MCP adapter refuses to read or write files outside
`AEGISFLOW_HERMES_ALLOWED_ROOT`. Keep that root narrow. Do not set it to an entire
system drive.

Protect and recover operations also refuse to overwrite existing output files.

## Scope

AegisFlow Secure Transfer protects files explicitly routed through AegisFlow. It
does not transparently intercept or re-encrypt arbitrary Chrome, WhatsApp, banking
app, or other third-party traffic.

The live monitor observes aggregate host-network metadata/statistics. The real
network threat estimator remains experimental until externally validated.

## Native HTTP contract

The backend also exposes:

```text
GET /api/integrations/hermes/v1/manifest
GET /api/integrations/hermes/v1/readiness
```

These stable endpoints make it straightforward to build a future native Hermes
plugin or remote HTTP MCP deployment without changing the AegisFlow core.
