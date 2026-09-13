from __future__ import annotations

import math
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from aegisflow.context.image_semantic import ImageSemanticAnalyzer


@dataclass(frozen=True)
class ContentContext:
    category: str
    confidence: float
    sensitivity: float
    confidentiality: float
    integrity: float
    authenticity: float
    privacy: float
    regulatory_requirement: float
    replay_freshness: float
    availability_criticality: float
    detected_mime: str
    size_bytes: int
    signals: List[str]
    analysis_mode: str
    vision_status: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def policy_requirements(self) -> Dict[str, float]:
        return {
            "sensitivity": self.sensitivity,
            "confidentiality": self.confidentiality,
            "integrity": self.integrity,
            "authenticity": self.authenticity,
            "privacy": self.privacy,
            "regulatory_requirement": self.regulatory_requirement,
            "replay_freshness": self.replay_freshness,
            "availability_criticality": self.availability_criticality,
        }


CATEGORY_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    "public": {
        "sensitivity": 0.22,
        "confidentiality": 0.30,
        "integrity": 0.60,
        "authenticity": 0.55,
        "privacy": 0.20,
        "regulatory_requirement": 0.10,
        "replay_freshness": 0.30,
        "availability_criticality": 0.42,
    },
    "general": {
        "sensitivity": 0.40,
        "confidentiality": 0.45,
        "integrity": 0.55,
        "authenticity": 0.50,
        "privacy": 0.40,
        "regulatory_requirement": 0.25,
        "replay_freshness": 0.40,
        "availability_criticality": 0.50,
    },
    "personal": {
        "sensitivity": 0.68,
        "confidentiality": 0.74,
        "integrity": 0.66,
        "authenticity": 0.66,
        "privacy": 0.84,
        "regulatory_requirement": 0.62,
        "replay_freshness": 0.55,
        "availability_criticality": 0.55,
    },
    "financial": {
        "sensitivity": 0.92,
        "confidentiality": 0.92,
        "integrity": 0.97,
        "authenticity": 0.96,
        "privacy": 0.87,
        "regulatory_requirement": 0.92,
        "replay_freshness": 0.92,
        "availability_criticality": 0.84,
    },
    "medical": {
        "sensitivity": 0.94,
        "confidentiality": 0.96,
        "integrity": 0.92,
        "authenticity": 0.88,
        "privacy": 0.97,
        "regulatory_requirement": 0.95,
        "replay_freshness": 0.72,
        "availability_criticality": 0.90,
    },
    "credentials": {
        "sensitivity": 0.98,
        "confidentiality": 0.99,
        "integrity": 0.95,
        "authenticity": 0.98,
        "privacy": 0.95,
        "regulatory_requirement": 0.90,
        "replay_freshness": 0.96,
        "availability_criticality": 0.72,
    },
    "iot": {
        "sensitivity": 0.56,
        "confidentiality": 0.52,
        "integrity": 0.84,
        "authenticity": 0.84,
        "privacy": 0.46,
        "regulatory_requirement": 0.42,
        "replay_freshness": 0.88,
        "availability_criticality": 0.82,
    },
}


CATEGORY_KEYWORDS: Dict[str, Iterable[str]] = {
    "public": (
        "poster", "flyer", "banner", "brochure", "brosur", "advertisement",
        "announcement", "pengumuman", "registration", "pendaftaran", "open registration",
        "event", "acara", "competition", "lomba", "seminar", "webinar", "workshop",
    ),
    "general": (
        "tower", "bts", "antenna", "telecom", "telecommunication", "telekomunikasi",
        "infrastructure", "infrastruktur", "building", "gedung", "landscape",
    ),
    "financial": (
        "invoice", "rekening", "bank", "financial", "finance", "payment",
        "transaction", "transaksi", "salary", "gaji", "credit", "debit",
        "account number", "nomor rekening", "swift", "iban",
    ),
    "medical": (
        "medical", "medis", "patient", "pasien", "diagnosis", "diagnosa",
        "health", "kesehatan", "prescription", "resep", "hospital", "lab result",
    ),
    "credentials": (
        "password", "passwd", "credential", "credentials", "api key", "apikey",
        "secret key", "private key", "access token", "refresh token", "otp",
    ),
    "personal": (
        "nik", "passport", "paspor", "birth date", "tanggal lahir", "address",
        "alamat", "phone number", "nomor telepon", "email", "personal data",
        "data pribadi",
    ),
    "iot": (
        "telemetry", "sensor", "mqtt", "device id", "iot", "temperature",
        "humidity", "actuator",
    ),
}


EXTENSION_HINTS: Dict[str, str] = {
    ".pem": "credentials",
    ".key": "credentials",
    ".p12": "credentials",
    ".pfx": "credentials",
    ".env": "credentials",
    ".kdbx": "credentials",
    ".ofx": "financial",
    ".qif": "financial",
    ".hl7": "medical",
    ".fhir": "medical",
}


PATTERN_HINTS = {
    "financial": (
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    ),
    "personal": (
        re.compile(r"\b\d{16}\b"),
        re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    ),
    "credentials": (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
        re.compile(r"(?i)\b(?:password|passwd|api[_ -]?key|access[_ -]?token)\b\s*[:=]"),
    ),
}


TEXT_EXTENSIONS = {
    ".txt", ".csv", ".json", ".xml", ".yaml", ".yml", ".md", ".log",
    ".ini", ".cfg", ".conf", ".env", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".sql",
}


FILENAME_WEIGHTS = {
    "public": 1.15,
    "general": 0.70,
    "financial": 0.90,
    "medical": 0.90,
    "credentials": 0.95,
    "personal": 0.80,
    "iot": 0.75,
}


class ContentClassifier:
    """Automatic application-context detector for AegisFlow Secure Transfer v2.

    v2 fixes two important deployment problems from v1:
    1. binary images are no longer decoded as if they were plaintext, and
    2. images can optionally be inspected by a local zero-shot vision model.

    The local vision path is optional. If its dependencies/model are unavailable,
    classification gracefully falls back to filename, MIME, and safe text parsing.
    """

    def __init__(
        self,
        max_text_sample_bytes: int = 256_000,
        *,
        enable_vision: bool = True,
        image_analyzer: ImageSemanticAnalyzer | None = None,
    ):
        self.max_text_sample_bytes = int(max_text_sample_bytes)
        self.image_analyzer = image_analyzer or ImageSemanticAnalyzer(enabled=enable_vision)

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def estimate_data_volume(size_bytes: int) -> float:
        # Smoothly maps roughly 1 KiB -> 0.17, 1 MiB -> 0.52, 1 GiB -> 0.86.
        return ContentClassifier._clip01(math.log10(max(size_bytes, 1) + 1) / 9.0)

    @staticmethod
    def _is_text_candidate(mime_type: str, suffix: str) -> bool:
        if mime_type.startswith("text/"):
            return True
        if suffix in TEXT_EXTENSIONS:
            return True
        return mime_type in {
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-yaml",
        }

    def classify(self, filename: str, content: bytes, mime_type: str | None = None) -> ContentContext:
        safe_name = Path(filename or "unnamed.bin").name
        suffix = Path(safe_name).suffix.lower()
        detected_mime = mime_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

        scores: Dict[str, float] = {name: 0.0 for name in CATEGORY_REQUIREMENTS}
        scores["general"] = 0.15
        signals: List[str] = []

        hinted = EXTENSION_HINTS.get(suffix)
        if hinted:
            scores[hinted] += 2.0
            signals.append(f"extension:{suffix}->{hinted}")

        lower_name = safe_name.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lower_name:
                    scores[category] += FILENAME_WEIGHTS[category]
                    signals.append(f"filename_keyword:{category}:{keyword}")

        text_candidate = self._is_text_candidate(detected_mime, suffix)
        if text_candidate:
            sample = content[: self.max_text_sample_bytes]
            text = sample.decode("utf-8", errors="ignore")
            lowered = text.lower()

            if text:
                for category, keywords in CATEGORY_KEYWORDS.items():
                    hits = sum(1 for keyword in keywords if keyword in lowered)
                    if hits:
                        scores[category] += min(1.5, 0.30 * hits)
                        signals.append(f"text_keywords:{category}:{hits}")

                for category, patterns in PATTERN_HINTS.items():
                    hits = sum(1 for pattern in patterns if pattern.search(text))
                    if hits:
                        scores[category] += 1.40 * hits
                        signals.append(f"structured_pattern:{category}:{hits}")
        else:
            signals.append("binary_content:not_text_scanned")

        vision_used = False
        vision_status = "not_applicable"
        if detected_mime.startswith("image/"):
            # Images are not private merely because they are images. v1 added a
            # personal-data bias here; v2 intentionally removes that assumption.
            scores["general"] += 0.20
            signals.append("mime:image")

            evidence = self.image_analyzer.analyze(content)
            vision_used = evidence.used
            vision_status = evidence.status

            if evidence.used:
                for category, score in evidence.category_scores.items():
                    # Visual semantics are useful evidence, but filename / strong
                    # structured signals can still dominate when appropriate.
                    if score >= 0.10:
                        scores[category] += 2.25 * score

                for item in evidence.top_labels:
                    signals.append(
                        "vision:%s:%.3f" % (
                            item["category"],
                            float(item["score"]),
                        )
                    )
            else:
                signals.append(f"vision:{vision_status}")

        if detected_mime.startswith("text/"):
            scores["general"] += 0.10

        category = max(scores, key=scores.get)
        best_score = scores[category]
        ordered = sorted(scores.values(), reverse=True)
        second_score = ordered[1] if len(ordered) > 1 else 0.0
        margin = max(0.0, best_score - second_score)

        if category in {"public", "general"} and best_score <= 0.35:
            confidence = 0.55
        else:
            confidence = self._clip01(0.54 + 0.14 * best_score + 0.09 * margin)

        requirements = CATEGORY_REQUIREMENTS[category]
        if not signals:
            signals.append("fallback:general_metadata")

        analysis_mode = (
            "hybrid_local_vision_metadata_text_v2"
            if vision_used
            else "metadata_text_fallback_v2"
        )

        return ContentContext(
            category=category,
            confidence=confidence,
            detected_mime=detected_mime,
            size_bytes=len(content),
            signals=signals[:16],
            analysis_mode=analysis_mode,
            vision_status=vision_status,
            **requirements,
        )
