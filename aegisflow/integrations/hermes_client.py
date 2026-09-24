from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx


class HermesAegisFlowClient:
    """Safe local bridge used by the Hermes MCP server.

    The bridge talks to the already-running AegisFlow FastAPI process so Hermes,
    the web dashboard, and Secure Transfer all share one runtime state.

    File tools are restricted to AEGISFLOW_HERMES_ALLOWED_ROOT. This prevents an
    agent from reading or writing arbitrary paths outside the explicitly allowed
    workspace.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        allowed_root: str | Path | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("AEGISFLOW_API_BASE", "http://127.0.0.1:8000")
        ).rstrip("/")
        root_value = (
            allowed_root
            or os.getenv("AEGISFLOW_HERMES_ALLOWED_ROOT")
            or os.getcwd()
        )
        self.allowed_root = Path(root_value).expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )

    def _resolve_allowed_path(
        self,
        path_value: str | Path,
        *,
        must_exist: bool,
    ) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = self.allowed_root / path
        path = path.resolve()

        try:
            path.relative_to(self.allowed_root)
        except ValueError as exc:
            raise ValueError(
                f"Path is outside AEGISFLOW_HERMES_ALLOWED_ROOT: {path}"
            ) from exc

        if must_exist and not path.is_file():
            raise FileNotFoundError(path)

        return path

    @staticmethod
    def _raise_for_response(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = response.text
        try:
            payload = response.json()
            detail = str(payload.get("detail") or payload)
        except Exception:
            pass
        raise RuntimeError(
            f"AegisFlow API {response.status_code}: {detail[:500]}"
        )

    def health(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/api/health")
            self._raise_for_response(response)
            return response.json()

    def status(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/api/status")
            self._raise_for_response(response)
            return response.json()

    def secure_capabilities(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/api/secure/capabilities")
            self._raise_for_response(response)
            return response.json()

    def hermes_manifest(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/api/integrations/hermes/v1/manifest")
            self._raise_for_response(response)
            return response.json()

    def start_monitoring(
        self,
        *,
        device_trust: float = 0.75,
        destination_trust: float = 0.75,
        latency_sensitivity: float = 0.50,
    ) -> dict[str, Any]:
        payload = {
            "profile_name": "general",
            "device_trust": float(device_trust),
            "destination_trust": float(destination_trust),
            "latency_sensitivity": float(latency_sensitivity),
        }
        with self._client() as client:
            response = client.post("/api/monitor/start", json=payload)
            self._raise_for_response(response)
            return response.json()

    def stop_monitoring(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.post("/api/monitor/stop")
            self._raise_for_response(response)
            return response.json()

    def analyze_file(
        self,
        file_path: str,
        *,
        device_trust: float = 0.75,
        destination_trust: float = 0.75,
        latency_sensitivity: float = 0.50,
    ) -> dict[str, Any]:
        path = self._resolve_allowed_path(file_path, must_exist=True)
        data = {
            "device_trust": str(float(device_trust)),
            "destination_trust": str(float(destination_trust)),
            "latency_sensitivity": str(float(latency_sensitivity)),
        }
        with path.open("rb") as handle, self._client() as client:
            response = client.post(
                "/api/secure/analyze",
                data=data,
                files={"file": (path.name, handle)},
            )
            self._raise_for_response(response)
            return response.json()

    def protect_file(
        self,
        file_path: str,
        *,
        output_path: str | None = None,
        device_trust: float = 0.75,
        destination_trust: float = 0.75,
        latency_sensitivity: float = 0.50,
    ) -> dict[str, Any]:
        source = self._resolve_allowed_path(file_path, must_exist=True)
        target = self._resolve_allowed_path(
            output_path or f"{source}.aegis",
            must_exist=False,
        )
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing protected file: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "device_trust": str(float(device_trust)),
            "destination_trust": str(float(destination_trust)),
            "latency_sensitivity": str(float(latency_sensitivity)),
        }
        with source.open("rb") as handle, self._client() as client:
            response = client.post(
                "/api/secure/protect",
                data=data,
                files={"file": (source.name, handle)},
            )
            self._raise_for_response(response)
            target.write_bytes(response.content)

        manifest = None
        raw_manifest = response.headers.get("X-AegisFlow-Manifest")
        if raw_manifest:
            try:
                manifest = json.loads(raw_manifest)
            except json.JSONDecodeError:
                manifest = None

        return {
            "success": True,
            "source_path": str(source),
            "protected_path": str(target),
            "size_bytes": target.stat().st_size,
            "policy": response.headers.get("X-AegisFlow-Policy"),
            "context": response.headers.get("X-AegisFlow-Context"),
            "manifest": manifest,
        }

    def unprotect_file(
        self,
        file_path: str,
        *,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        source = self._resolve_allowed_path(file_path, must_exist=True)
        with source.open("rb") as handle, self._client() as client:
            response = client.post(
                "/api/secure/unprotect",
                files={"file": (source.name, handle)},
            )
            self._raise_for_response(response)

        suggested = "recovered.bin"
        disposition = response.headers.get("Content-Disposition", "")
        match = re.search(r'filename="([^"]+)"', disposition)
        if match:
            suggested = Path(match.group(1)).name

        target = self._resolve_allowed_path(
            output_path or str(source.with_name(suggested)),
            must_exist=False,
        )
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing recovered file: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)

        return {
            "success": True,
            "protected_path": str(source),
            "recovered_path": str(target),
            "size_bytes": target.stat().st_size,
            "policy": response.headers.get("X-AegisFlow-Policy"),
        }
