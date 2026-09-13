from __future__ import annotations

from typing import Dict, Tuple

from aegisflow.context.content_classifier import ContentClassifier
from aegisflow.security.profiles import resolve_security_profile
from aegisflow.security.secure_transfer import SecureTransferService
from aegisflow.service.context_builder import build_policy_context_from_requirements


class AdaptiveSecureTransfer:
    """Connect automatic content context, live network intelligence, policy AI, and crypto."""

    def __init__(self, live_service):
        self.live_service = live_service
        self.classifier = ContentClassifier()
        self.crypto = SecureTransferService()

    @staticmethod
    def _live_values(latest_result: Dict | None) -> Tuple[float, float]:
        if not latest_result:
            return 0.20, 0.30

        intelligence = latest_result.get("intelligence") or {}
        telemetry = latest_result.get("telemetry") or {}
        threat = float(intelligence.get("threat_score", 0.20))
        frequency = float(telemetry.get("request_frequency", 0.30))
        return threat, frequency

    def analyze(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str | None,
        latest_result: Dict | None,
        device_trust: float,
        destination_trust: float,
        latency_sensitivity: float,
    ) -> Dict[str, object]:
        content_context = self.classifier.classify(
            filename=filename,
            content=content,
            mime_type=mime_type,
        )
        network_threat, transmission_frequency = self._live_values(latest_result)
        data_volume = self.classifier.estimate_data_volume(len(content))

        policy_context = build_policy_context_from_requirements(
            requirements=content_context.policy_requirements(),
            network_threat=network_threat,
            data_volume=data_volume,
            transmission_frequency=transmission_frequency,
            device_trust=device_trust,
            destination_trust=destination_trust,
            latency_sensitivity=latency_sensitivity,
        )
        decision = self.live_service.infer_policy(policy_context)
        profile = resolve_security_profile(decision["policy"])

        return {
            "classifier_mode": content_context.analysis_mode,
            "content_context": content_context.to_dict(),
            "policy_context": policy_context,
            "decision": decision,
            "security_profile": profile.to_dict(),
            "network_context": {
                "threat_score": network_threat,
                "transmission_frequency": transmission_frequency,
                "source": "live_monitoring" if latest_result else "safe_default_without_live_monitoring",
            },
        }

    def protect(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str | None,
        latest_result: Dict | None,
        device_trust: float,
        destination_trust: float,
        latency_sensitivity: float,
    ):
        analysis = self.analyze(
            filename=filename,
            content=content,
            mime_type=mime_type,
            latest_result=latest_result,
            device_trust=device_trust,
            destination_trust=destination_trust,
            latency_sensitivity=latency_sensitivity,
        )
        profile = resolve_security_profile(analysis["decision"]["policy"])
        envelope, manifest = self.crypto.protect(
            filename=filename,
            content=content,
            policy_name=analysis["decision"]["policy"],
            profile=profile,
            context=analysis["content_context"],
        )
        manifest["analysis"] = analysis
        return envelope, manifest

    def unprotect(self, envelope: bytes):
        return self.crypto.unprotect(envelope)
