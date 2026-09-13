from __future__ import annotations

import math
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


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


class ContentClassifier:
    """Automatic application-context detector for AegisFlow Secure Transfer v1.

    The classifier intentionally uses metadata and a bounded plaintext sample. It does
    not intercept or decrypt unrelated network traffic. This first deployment version
    is deterministic and explainable; it can later be replaced by a learned classifier
    while keeping the same normalized policy interface.
    """

    def __init__(self, max_text_sample_bytes: int = 256_000):
        self.max_text_sample_bytes = int(max_text_sample_bytes)

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def estimate_data_volume(size_bytes: int) -> float:
        # Smoothly maps roughly 1 KiB -> 0.17, 1 MiB -> 0.52, 1 GiB -> 0.86.
        return ContentClassifier._clip01(math.log10(max(size_bytes, 1) + 1) / 9.0)

    def classify(self, filename: str, content: bytes, mime_type: str | None = None) -> ContentContext:
        safe_name = Path(filename or "unnamed.bin").name
        suffix = Path(safe_name).suffix.lower()
        detected_mime = mime_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

        scores: Dict[str, float] = {name: 0.0 for name in CATEGORY_REQUIREMENTS}
        scores["general"] = 0.15
        signals: List[str] = []

        hinted = EXTENSION_HINTS.get(suffix)
        if hinted:
            scores[hinted] += 1.4
            signals.append(f"extension:{suffix}->{hinted}")

        lower_name = safe_name.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lower_name:
                    scores[category] += 0.9
                    signals.append(f"filename_keyword:{keyword}")

        sample = content[: self.max_text_sample_bytes]
        text = sample.decode("utf-8", errors="ignore")
        lowered = text.lower()

        if text:
            for category, keywords in CATEGORY_KEYWORDS.items():
                hits = sum(1 for keyword in keywords if keyword in lowered)
                if hits:
                    scores[category] += min(1.8, 0.35 * hits)
                    signals.append(f"text_keywords:{category}:{hits}")

            for category, patterns in PATTERN_HINTS.items():
                hits = sum(1 for pattern in patterns if pattern.search(text))
                if hits:
                    scores[category] += 1.2 * hits
                    signals.append(f"structured_pattern:{category}:{hits}")

        if detected_mime.startswith("text/"):
            scores["general"] += 0.1
        elif detected_mime.startswith("image/"):
            scores["personal"] += 0.15
            signals.append("mime:image")

        category = max(scores, key=scores.get)
        best_score = scores[category]
        ordered = sorted(scores.values(), reverse=True)
        second_score = ordered[1] if len(ordered) > 1 else 0.0
        margin = max(0.0, best_score - second_score)

        if category == "general" and best_score <= 0.25:
            confidence = 0.55
        else:
            confidence = self._clip01(0.58 + 0.16 * best_score + 0.08 * margin)

        requirements = CATEGORY_REQUIREMENTS[category]
        if not signals:
            signals.append("fallback:general_metadata")

        return ContentContext(
            category=category,
            confidence=confidence,
            detected_mime=detected_mime,
            size_bytes=len(content),
            signals=signals[:12],
            **requirements,
        )
