import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aegisflow.service.live_intelligence import AegisFlowLiveService
from backend.hermes_routes import build_hermes_router
from backend.secure_transfer_routes import build_secure_transfer_router


class MonitoringConfig(BaseModel):
    # profile_name is retained for the live dashboard/debug path. Secure Transfer does
    # automatic content detection and does not require the user to choose a profile.
    profile_name: str = "general"
    device_trust: float = Field(default=0.75, ge=0.0, le=1.0)
    destination_trust: float = Field(default=0.75, ge=0.0, le=1.0)
    latency_sensitivity: float = Field(default=0.50, ge=0.0, le=1.0)


class RuntimeState:
    def __init__(self):
        self.service: Optional[AegisFlowLiveService] = None
        self.secure_transfer = None
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.latest_result = None
        self.sequence = 0
        self.error = None
        self.config = MonitoringConfig()


runtime = RuntimeState()
VALID_PROFILES = {"general", "personal", "financial", "medical", "iot"}


def get_service():
    if runtime.service is None:
        print("Loading AegisFlow models...")
        runtime.service = AegisFlowLiveService(interval_seconds=2.0)
        print("AegisFlow models loaded.")
    return runtime.service


async def monitoring_loop():
    runtime.error = None
    service = get_service()
    try:
        while runtime.monitoring:
            config = runtime.config
            result = await asyncio.to_thread(
                service.run_once,
                profile_name=config.profile_name,
                device_trust=config.device_trust,
                destination_trust=config.destination_trust,
                latency_sensitivity=config.latency_sensitivity,
            )
            runtime.latest_result = result
            runtime.sequence += 1
    except asyncio.CancelledError:
        raise
    except Exception as error:
        runtime.error = str(error)
        runtime.monitoring = False
        print("Monitoring error:", error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AegisFlow API starting...")
    yield
    runtime.monitoring = False
    if runtime.monitor_task:
        runtime.monitor_task.cancel()
        try:
            await runtime.monitor_task
        except asyncio.CancelledError:
            pass
    print("AegisFlow API stopped.")


app = FastAPI(
    title="AegisFlow API",
    description="Real-time adaptive security intelligence and secure transfer backend.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-AegisFlow-Policy",
        "X-AegisFlow-Context",
        "X-AegisFlow-Manifest",
    ],
)


@app.get("/")
async def root():
    return {
        "name": "AegisFlow",
        "api_version": "0.3.0",
        "status": "online",
        "secure_transfer": True,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "monitoring": runtime.monitoring,
        "models_loaded": runtime.service is not None,
        "error": runtime.error,
    }


@app.get("/api/status")
async def status():
    return {
        "monitoring": runtime.monitoring,
        "sequence": runtime.sequence,
        "config": runtime.config.model_dump(),
        "error": runtime.error,
        "latest": runtime.latest_result,
    }


@app.post("/api/monitor/start")
async def start_monitoring(config: MonitoringConfig):
    if runtime.monitoring:
        return {
            "success": True,
            "message": "Monitoring already active.",
            "monitoring": True,
        }

    if config.profile_name not in VALID_PROFILES:
        return {
            "success": False,
            "message": f"Invalid profile_name. Available: {sorted(VALID_PROFILES)}",
        }

    runtime.config = config
    service = get_service()
    service.temporal_anomaly.reset()
    runtime.monitoring = True
    runtime.error = None
    runtime.monitor_task = asyncio.create_task(monitoring_loop())
    return {
        "success": True,
        "message": "AegisFlow monitoring started.",
        "monitoring": True,
        "config": runtime.config.model_dump(),
    }


@app.post("/api/monitor/stop")
async def stop_monitoring():
    if not runtime.monitoring:
        return {
            "success": True,
            "message": "Monitoring already stopped.",
            "monitoring": False,
        }

    runtime.monitoring = False
    task = runtime.monitor_task
    if task:
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    runtime.monitor_task = None
    return {
        "success": True,
        "message": "AegisFlow monitoring stopped.",
        "monitoring": False,
    }


@app.put("/api/config")
async def update_config(config: MonitoringConfig):
    if config.profile_name not in VALID_PROFILES:
        return {"success": False, "message": "Invalid data profile."}
    runtime.config = config
    return {"success": True, "config": runtime.config.model_dump()}


@app.get("/api/profiles")
async def profiles():
    return {
        "profiles": [
            {"id": "general", "name": "General", "description": "Debug/general preset."},
            {"id": "personal", "name": "Personal", "description": "Debug personal-data preset."},
            {"id": "financial", "name": "Financial", "description": "Debug financial-data preset."},
            {"id": "medical", "name": "Medical", "description": "Debug medical-data preset."},
            {"id": "iot", "name": "IoT", "description": "Debug IoT-data preset."},
        ],
        "note": "Secure Transfer classifies content automatically; these presets are retained for monitoring/debug overrides.",
    }


@app.get("/api/model-performance")
async def model_performance():
    return {
        "evaluation_scope": "held_out_synthetic_test",
        "policy": {
            "oracle_agreement_pct": 97.594570,
            "mean_reward": 2.037938,
            "mean_regret": 0.000939,
            "policy_consistency_pct": 99.182,
        },
        "threat_estimator": {
            "mae": 0.052783,
            "rmse": 0.066105,
            "r2": 0.891337,
            "correlation": 0.945085,
        },
        "synthetic_anomaly_detector": {
            "roc_auc": 0.832570,
            "pr_auc": 0.693644,
            "precision": 0.658602,
            "recall": 0.566081,
            "f1": 0.608847,
        },
        "deployment_anomaly_detector": {
            "mode": "network_specific_baseline",
            "threshold": 0.00483514,
            "validation_status": "prototype_external_validation_pending",
        },
    }


@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket):
    await websocket.accept()
    last_sequence = -1
    try:
        while True:
            if runtime.sequence != last_sequence:
                last_sequence = runtime.sequence
                await websocket.send_json(
                    {
                        "type": "telemetry",
                        "monitoring": runtime.monitoring,
                        "sequence": runtime.sequence,
                        "data": runtime.latest_result,
                        "error": runtime.error,
                    }
                )
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass
    except Exception as error:
        print("WebSocket error:", error)


app.include_router(
    build_secure_transfer_router(
        get_service=get_service,
        get_runtime=lambda: runtime,
    )
)


app.include_router(
    build_hermes_router(
        get_runtime=lambda: runtime,
    )
)
