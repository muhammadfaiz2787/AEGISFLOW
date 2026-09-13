from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, List


DEFAULT_VISION_MODEL = "openai/clip-vit-base-patch32"

# Human-readable zero-shot labels are intentionally broader than the policy
# categories. The model judges visual semantics; AegisFlow then maps the result
# into the normalized security-context categories used by the policy network.
VISION_LABEL_TO_CATEGORY: Dict[str, str] = {
    "a public event poster, registration flyer, announcement, or advertisement": "public",
    "a telecommunications tower, antenna, building, landscape, or public infrastructure photo": "general",
    "an identity card, passport, driver's license, or document containing personal identity data": "personal",
    "a bank statement, payment receipt, invoice, credit card, or financial document": "financial",
    "a medical record, prescription, laboratory result, or patient document": "medical",
    "a screenshot or document containing passwords, API keys, private keys, tokens, or credentials": "credentials",
    "an IoT device, sensor dashboard, telemetry screen, or industrial control interface": "iot",
    "a private personal portrait, family photo, or image containing private personal information": "personal",
}


@dataclass(frozen=True)
class VisionEvidence:
    used: bool
    status: str
    category_scores: Dict[str, float]
    top_labels: List[Dict[str, object]]


class ImageSemanticAnalyzer:
    """Optional local zero-shot image semantic analyzer.

    The dependency is deliberately optional so the core AegisFlow pipeline can
    still run without downloading a vision model. When ``transformers`` and
    ``Pillow`` are installed, the model is loaded lazily on the first image.

    No image bytes are sent to a cloud API by this class.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_name: str | None = None,
    ):
        env_enabled = os.getenv("AEGISFLOW_ENABLE_LOCAL_VISION", "1").strip().lower()
        self.enabled = bool(enabled) and env_enabled not in {"0", "false", "no", "off"}
        self.model_name = model_name or os.getenv(
            "AEGISFLOW_VISION_MODEL",
            DEFAULT_VISION_MODEL,
        )
        self._pipeline = None
        self._load_error: str | None = None

    def _get_pipeline(self):
        if not self.enabled:
            return None
        if self._pipeline is not None:
            return self._pipeline
        if self._load_error is not None:
            return None

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="zero-shot-image-classification",
                model=self.model_name,
                device=-1,
            )
        except Exception as exc:  # optional deployment dependency / first-run download
            self._load_error = exc.__class__.__name__
            return None

        return self._pipeline

    def analyze(self, content: bytes) -> VisionEvidence:
        if not self.enabled:
            return VisionEvidence(
                used=False,
                status="disabled",
                category_scores={},
                top_labels=[],
            )

        try:
            from PIL import Image
        except Exception:
            return VisionEvidence(
                used=False,
                status="pillow_not_installed",
                category_scores={},
                top_labels=[],
            )

        classifier = self._get_pipeline()
        if classifier is None:
            status = (
                f"model_unavailable:{self._load_error}"
                if self._load_error
                else "transformers_not_installed"
            )
            return VisionEvidence(
                used=False,
                status=status,
                category_scores={},
                top_labels=[],
            )

        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            outputs = classifier(
                image,
                candidate_labels=list(VISION_LABEL_TO_CATEGORY.keys()),
            )
        except Exception as exc:
            return VisionEvidence(
                used=False,
                status=f"image_analysis_failed:{exc.__class__.__name__}",
                category_scores={},
                top_labels=[],
            )

        category_scores: Dict[str, float] = {}
        top_labels: List[Dict[str, object]] = []

        for item in outputs[:5]:
            label = str(item.get("label", ""))
            score = float(item.get("score", 0.0))
            category = VISION_LABEL_TO_CATEGORY.get(label)
            if not category:
                continue

            # Keep the strongest visual evidence for each category. Using max
            # avoids double-counting multiple semantically similar prompt labels.
            category_scores[category] = max(
                category_scores.get(category, 0.0),
                score,
            )
            top_labels.append(
                {
                    "label": label,
                    "category": category,
                    "score": score,
                }
            )

        return VisionEvidence(
            used=True,
            status=f"active:{self.model_name}",
            category_scores=category_scores,
            top_labels=top_labels[:3],
        )
