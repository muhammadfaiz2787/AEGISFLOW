from __future__ import annotations

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from aegisflow.integrations.hermes_client import HermesAegisFlowClient


mcp = MCPServer("AegisFlow")
client = HermesAegisFlowClient()

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
WRITE_IDEMPOTENT = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
WRITE_FILE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)


@mcp.tool(
    title="AegisFlow health",
    annotations=READ_ONLY,
)
def aegisflow_health() -> dict:
    """Check whether the local AegisFlow backend is reachable and healthy."""
    return client.health()


@mcp.tool(
    title="AegisFlow live status",
    annotations=READ_ONLY,
)
def aegisflow_status() -> dict:
    """Read current monitoring state and the latest network-intelligence result."""
    return client.status()


@mcp.tool(
    title="AegisFlow capabilities",
    annotations=READ_ONLY,
)
def aegisflow_capabilities() -> dict:
    """Read Secure Transfer capabilities and the Hermes integration manifest."""
    return {
        "secure_transfer": client.secure_capabilities(),
        "hermes": client.hermes_manifest(),
    }


@mcp.tool(
    title="Start AegisFlow monitoring",
    annotations=WRITE_IDEMPOTENT,
)
def aegisflow_start_monitoring(
    device_trust: float = 0.75,
    destination_trust: float = 0.75,
    latency_sensitivity: float = 0.50,
) -> dict:
    """Start aggregate local network monitoring.

    This observes host-network metadata/statistics. It does not decrypt HTTPS or
    intercept arbitrary third-party application payloads.
    """
    return client.start_monitoring(
        device_trust=device_trust,
        destination_trust=destination_trust,
        latency_sensitivity=latency_sensitivity,
    )


@mcp.tool(
    title="Stop AegisFlow monitoring",
    annotations=WRITE_IDEMPOTENT,
)
def aegisflow_stop_monitoring() -> dict:
    """Stop AegisFlow live network monitoring."""
    return client.stop_monitoring()


@mcp.tool(
    title="Analyze a file with AegisFlow",
    annotations=READ_ONLY,
)
def aegisflow_analyze_file(
    file_path: str,
    device_trust: float = 0.75,
    destination_trust: float = 0.75,
    latency_sensitivity: float = 0.50,
) -> dict:
    """Analyze one allowed local file and return content context, policy, and crypto profile.

    The path must be inside AEGISFLOW_HERMES_ALLOWED_ROOT. The file is sent only
    to the configured AegisFlow API, which defaults to localhost.
    """
    return client.analyze_file(
        file_path,
        device_trust=device_trust,
        destination_trust=destination_trust,
        latency_sensitivity=latency_sensitivity,
    )


@mcp.tool(
    title="Protect a file with AegisFlow",
    annotations=WRITE_FILE,
)
def aegisflow_protect_file(
    file_path: str,
    output_path: str | None = None,
    device_trust: float = 0.75,
    destination_trust: float = 0.75,
    latency_sensitivity: float = 0.50,
    recipient_public_key: str | None = None,
) -> dict:
    """Adaptively encrypt an allowed local file and save an .aegis envelope.

    Existing output files are never overwritten.
    """
    return client.protect_file(
        file_path,
        output_path=output_path,
        device_trust=device_trust,
        destination_trust=destination_trust,
        latency_sensitivity=latency_sensitivity,
        recipient_public_key=recipient_public_key,
    )


@mcp.tool(
    title="Recover an AegisFlow file",
    annotations=WRITE_FILE,
)
def aegisflow_unprotect_file(
    file_path: str,
    output_path: str | None = None,
) -> dict:
    """Decrypt an allowed .aegis envelope for this local recipient installation.

    Existing output files are never overwritten.
    """
    return client.unprotect_file(
        file_path,
        output_path=output_path,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
