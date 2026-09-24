from __future__ import annotations

import asyncio

import json
from io import BytesIO
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from aegisflow.service.secure_transfer_orchestrator import AdaptiveSecureTransfer


MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def build_secure_transfer_router(get_service: Callable, get_runtime: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/secure", tags=["secure-transfer"])

    def orchestrator() -> AdaptiveSecureTransfer:
        runtime = get_runtime()
        if getattr(runtime, "secure_transfer", None) is None:
            runtime.secure_transfer = AdaptiveSecureTransfer(get_service())
        return runtime.secure_transfer

    async def read_upload(file: UploadFile) -> bytes:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB prototype limit.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return content

    @router.get("/capabilities")
    async def capabilities():
        service = orchestrator()
        return {
            "mode": "adaptive_secure_transfer_v3",
            "automatic_content_detection": True,
            "content_intelligence": {
                "text": True,
                "images": "local OpenCLIP when optional vision dependencies are installed",
                "documents": ["PDF", "DOCX", "XLSX", "PPTX"],
                "binary_plaintext_scanning": False,
            },
            "network_context_source": "live monitor when available; safe defaults otherwise",
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "recipient_public_key": service.crypto.key_store.public_key_b64(),
            "remote_recipient_public_key_supported": True,
            "hermes_mcp_ready": True,
            "important_scope": (
                "This protects files explicitly sent through AegisFlow Secure Transfer. "
                "It does not intercept or re-encrypt arbitrary browser/application traffic."
            ),
        }

    @router.post("/analyze")
    async def analyze_file(
        file: UploadFile = File(...),
        device_trust: float = Form(0.75),
        destination_trust: float = Form(0.75),
        latency_sensitivity: float = Form(0.50),
    ):
        runtime = get_runtime()
        content = await read_upload(file)
        try:
            return await asyncio.to_thread(
                orchestrator().analyze,
                filename=file.filename or "unnamed.bin",
                content=content,
                mime_type=file.content_type,
                latest_result=runtime.latest_result,
                device_trust=device_trust,
                destination_trust=destination_trust,
                latency_sensitivity=latency_sensitivity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/protect")
    async def protect_file(
        file: UploadFile = File(...),
        device_trust: float = Form(0.75),
        destination_trust: float = Form(0.75),
        latency_sensitivity: float = Form(0.50),
        recipient_public_key: str | None = Form(None),
    ):
        runtime = get_runtime()
        content = await read_upload(file)
        try:
            envelope, manifest = await asyncio.to_thread(
                orchestrator().protect,
                filename=file.filename or "unnamed.bin",
                content=content,
                mime_type=file.content_type,
                latest_result=runtime.latest_result,
                device_trust=device_trust,
                destination_trust=destination_trust,
                latency_sensitivity=latency_sensitivity,
                recipient_public_key=recipient_public_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        headers = {
            "Content-Disposition": f'attachment; filename="{manifest["protected_filename"]}"',
            "X-AegisFlow-Policy": str(manifest["policy"]),
            "X-AegisFlow-Context": str(manifest["content_context"]["category"]),
            "X-AegisFlow-Manifest": json.dumps(manifest, separators=(",", ":")),
        }
        return StreamingResponse(
            BytesIO(envelope),
            media_type="application/vnd.aegisflow.secure-transfer",
            headers=headers,
        )

    @router.post("/unprotect")
    async def unprotect_file(file: UploadFile = File(...)):
        envelope = await read_upload(file)
        try:
            plaintext, metadata = await asyncio.to_thread(
                orchestrator().unprotect,
                envelope,
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Unable to decrypt envelope: {exc}") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Envelope authentication failed or the recipient key does not match.",
            ) from exc

        original_filename = metadata.get("original_filename", "recovered.bin")
        headers = {
            "Content-Disposition": f'attachment; filename="{original_filename}"',
            "X-AegisFlow-Policy": str(metadata.get("policy", "UNKNOWN")),
        }
        return StreamingResponse(
            BytesIO(plaintext),
            media_type="application/octet-stream",
            headers=headers,
        )

    return router
