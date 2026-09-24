from __future__ import annotations

from fastapi import APIRouter


def build_hermes_router(get_runtime) -> APIRouter:
    router = APIRouter(
        prefix="/api/integrations/hermes/v1",
        tags=["hermes-integration"],
    )

    @router.get("/manifest")
    async def manifest():
        return {
            "integration": "hermes-agent",
            "version": "1.0",
            "preferred_transport": "mcp_stdio",
            "mcp_module": "aegisflow.integrations.hermes_mcp",
            "core_role": (
                "Hermes orchestrates tool usage; AegisFlow remains authoritative "
                "for network intelligence, content context, policy selection, and "
                "cryptographic enforcement."
            ),
            "tools": [
                {"name": "aegisflow_health", "read_only": True},
                {"name": "aegisflow_status", "read_only": True},
                {"name": "aegisflow_capabilities", "read_only": True},
                {"name": "aegisflow_start_monitoring", "read_only": False},
                {"name": "aegisflow_stop_monitoring", "read_only": False},
                {"name": "aegisflow_analyze_file", "read_only": True},
                {"name": "aegisflow_protect_file", "read_only": False},
                {"name": "aegisflow_unprotect_file", "read_only": False},
            ],
            "security": {
                "recommended_hermes_trust": "untrusted",
                "file_root_env": "AEGISFLOW_HERMES_ALLOWED_ROOT",
                "api_base_env": "AEGISFLOW_API_BASE",
                "overwrite_existing_files": False,
            },
            "scope": {
                "live_monitoring": "aggregate host-network metadata/statistics",
                "secure_transfer": "explicit files routed through AegisFlow",
                "arbitrary_tls_interception": False,
            },
        }

    @router.get("/readiness")
    async def readiness():
        runtime = get_runtime()
        return {
            "ready": True,
            "monitoring": bool(runtime.monitoring),
            "models_loaded": runtime.service is not None,
            "secure_transfer_initialized": runtime.secure_transfer is not None,
            "latest_sequence": int(runtime.sequence),
            "error": runtime.error,
        }

    return router
