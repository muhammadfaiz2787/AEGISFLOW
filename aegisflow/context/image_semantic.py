from __future__ import annotations

import hashlib
import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple


DEFAULT_VISION_MODEL = "RN50"
DEFAULT_VISION_PRETRAINED = "openai"

# Prompt ensembles make the classifier less sensitive to one oddly-worded label.
# The generic/icon prompts are important: without a benign alternative, a closed-set
# CLIP classifier is forced to call even a lightning icon "financial" or "credentials"
# simply because one of those prompts is the least-bad match.
VISION_PROMPTS: Dict[str, Tuple[str, ...]] = {
    "public": (
        "a public event poster, flyer, banner, announcement, or advertisement",
        "a public registration poster or promotional event graphic",
    ),
    "general": (
        "a simple vector icon, logo, symbol, lightning bolt, or decorative graphic",
        "a generic illustration, diagram, chart, screenshot, or non-sensitive graphic",
        "a telecommunications tower, antenna, building, landscape, or public infrastructure photo",
        "an ordinary non-sensitive photograph or image",
    ),
    "personal": (
        "an identity card, passport, driver's license, or document containing personal identity data",
        "a private personal portrait or image containing private personal information",
    ),
    "financial": (
        "a bank statement, payment receipt, invoice, credit card, or financial document",
        "a document showing financial transactions, banking details, or payment information",
    ),
    "medical": (
        "a medical record, prescription, laboratory result, or patient document",
        "a document or screenshot containing private health information",
    ),
    "credentials": (
        "a screenshot or document containing passwords, API keys, private keys, tokens, or credentials",
        "a secret authentication credential, password, access token, or cryptographic private key",
    ),
    "iot": (
        "an IoT device, sensor dashboard, telemetry screen, or industrial control interface",
        "a sensor, embedded device, telemetry dashboard, or machine control panel",
    ),
}

SENSITIVE_CATEGORIES = {"credentials", "financial", "medical", "personal"}

# Conservative deployment gate. Zero-shot softmax scores are relative similarities,
# not calibrated security probabilities. A weak sensitive match must not be allowed
# to drive HIGH/CRITICAL policy by itself.
SENSITIVE_MIN_SCORE = 0.60
SENSITIVE_MIN_MARGIN_OVER_BENIGN = 0.15
IOT_MIN_SCORE = 0.55
IOT_MIN_MARGIN_OVER_BENIGN = 0.12


@dataclass(frozen=True)
class VisionEvidence:
    used: bool
    status: str
    category_scores: Dict[str, float]
    top_labels: List[Dict[str, object]]


def gate_category_scores(
    category_scores: Dict[str, float],
) -> tuple[Dict[str, float], str]:
    """Reject weak security-sensitive zero-shot matches.

    CLIP/OpenCLIP always ranks the supplied prompts, even when none is a good semantic
    fit. This gate makes the image path open-set-ish: a weak or ambiguous sensitive
    result falls back to GENERAL instead of escalating the security policy.
    """

    if not category_scores:
        return {}, "no_scores"

    normalized = {
        str(category): max(0.0, min(1.0, float(score)))
        for category, score in category_scores.items()
    }
    top_category, top_score = max(normalized.items(), key=lambda item: item[1])
    benign_best = max(
        normalized.get("general", 0.0),
        normalized.get("public", 0.0),
    )
    margin = top_score - benign_best

    if top_category in SENSITIVE_CATEGORIES:
        if (
            top_score < SENSITIVE_MIN_SCORE
            or margin < SENSITIVE_MIN_MARGIN_OVER_BENIGN
        ):
            adjusted = dict(normalized)
            # Preserve diagnostic scores, but make the policy-facing winner benign.
            adjusted[top_category] = min(adjusted[top_category], benign_best)
            adjusted["general"] = max(
                adjusted.get("general", 0.0),
                min(0.55, max(0.30, top_score)),
            )
            return adjusted, f"rejected_weak_sensitive:{top_category}"

    if top_category == "iot":
        if top_score < IOT_MIN_SCORE or margin < IOT_MIN_MARGIN_OVER_BENIGN:
            adjusted = dict(normalized)
            adjusted["iot"] = min(adjusted["iot"], benign_best)
            adjusted["general"] = max(
                adjusted.get("general", 0.0),
                min(0.55, max(0.30, top_score)),
            )
            return adjusted, "rejected_weak_iot"

    return normalized, f"accepted:{top_category}"


class ImageSemanticAnalyzer:
    """Local OpenCLIP image semantic analyzer for AegisFlow.

    Default model: OpenCLIP RN50 with the OpenAI pretrained checkpoint. It is much
    smaller than the previous ViT-B/32 checkpoint and remains fully local after the
    checkpoint is downloaded. No image bytes are sent to a cloud inference API.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_name: str | None = None,
        pretrained: str | None = None,
        result_cache_size: int = 32,
    ):
        env_enabled = os.getenv("AEGISFLOW_ENABLE_LOCAL_VISION", "1").strip().lower()
        self.enabled = bool(enabled) and env_enabled not in {"0", "false", "no", "off"}
        self.model_name = model_name or os.getenv(
            "AEGISFLOW_VISION_MODEL",
            DEFAULT_VISION_MODEL,
        )
        self.pretrained = pretrained or os.getenv(
            "AEGISFLOW_VISION_PRETRAINED",
            DEFAULT_VISION_PRETRAINED,
        )
        self.result_cache_size = max(1, int(result_cache_size))
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._category_text_features = None
        self._load_error: str | None = None
        self._result_cache: OrderedDict[str, VisionEvidence] = OrderedDict()

    def _load(self):
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device="cpu",
            )
            tokenizer = open_clip.get_tokenizer(self.model_name)
            model.eval()

            # Encode prompt ensembles once. Later images only need one image forward
            # pass plus a tiny matrix multiplication.
            category_text_features = {}
            with torch.inference_mode():
                for category, prompts in VISION_PROMPTS.items():
                    tokens = tokenizer(list(prompts))
                    text_features = model.encode_text(tokens)
                    text_features = text_features / text_features.norm(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)
                    category_vector = text_features.mean(dim=0, keepdim=True)
                    category_vector = category_vector / category_vector.norm(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)
                    category_text_features[category] = category_vector.cpu()

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._category_text_features = category_text_features
        except Exception as exc:  # optional dependency / first-run checkpoint download
            self._load_error = f"{exc.__class__.__name__}:{str(exc)[:120]}"
            return False

        return True

    def _cache_get(self, content_hash: str) -> VisionEvidence | None:
        cached = self._result_cache.get(content_hash)
        if cached is not None:
            self._result_cache.move_to_end(content_hash)
        return cached

    def _cache_put(self, content_hash: str, evidence: VisionEvidence) -> None:
        self._result_cache[content_hash] = evidence
        self._result_cache.move_to_end(content_hash)
        while len(self._result_cache) > self.result_cache_size:
            self._result_cache.popitem(last=False)

    def analyze(self, content: bytes) -> VisionEvidence:
        if not self.enabled:
            return VisionEvidence(
                used=False,
                status="disabled",
                category_scores={},
                top_labels=[],
            )

        content_hash = hashlib.sha256(content).hexdigest()
        cached = self._cache_get(content_hash)
        if cached is not None:
            return VisionEvidence(
                used=cached.used,
                status=f"{cached.status};cache=hit",
                category_scores=dict(cached.category_scores),
                top_labels=list(cached.top_labels),
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

        if not self._load():
            status = (
                f"model_unavailable:{self._load_error}"
                if self._load_error
                else "openclip_not_installed"
            )
            return VisionEvidence(
                used=False,
                status=status,
                category_scores={},
                top_labels=[],
            )

        try:
            import torch

            image = Image.open(io.BytesIO(content)).convert("RGB")
            image_tensor = self._preprocess(image).unsqueeze(0)

            categories = list(self._category_text_features.keys())
            text_matrix = torch.cat(
                [self._category_text_features[category] for category in categories],
                dim=0,
            )

            with torch.inference_mode():
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)

                logits = 100.0 * image_features.cpu() @ text_matrix.T
                probabilities = logits.softmax(dim=-1)[0]

            raw_scores = {
                category: float(probabilities[index].item())
                for index, category in enumerate(categories)
            }
            gated_scores, gate_status = gate_category_scores(raw_scores)

            ordered = sorted(
                raw_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top_labels = [
                {
                    "label": category,
                    "category": category,
                    "score": score,
                }
                for category, score in ordered[:4]
            ]
        except Exception as exc:
            return VisionEvidence(
                used=False,
                status=f"image_analysis_failed:{exc.__class__.__name__}",
                category_scores={},
                top_labels=[],
            )

        evidence = VisionEvidence(
            used=True,
            status=(
                f"active:openclip:{self.model_name}:{self.pretrained};"
                f"gate={gate_status};cache=miss"
            ),
            category_scores=gated_scores,
            top_labels=top_labels,
        )
        self._cache_put(content_hash, evidence)
        return evidence
