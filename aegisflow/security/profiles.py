from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class SecurityProfile:
    level: str
    data_cipher: str
    session_key_bits: int
    key_agreement: str
    kdf: str
    authentication: str
    enforced_controls: tuple[str, ...]
    recommended_controls: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["enforced_controls"] = list(self.enforced_controls)
        payload["recommended_controls"] = list(self.recommended_controls)
        return payload


# These are deployable cryptographic suites for AegisFlow Secure Transfer v1.
# They intentionally use mature primitives from the `cryptography` package.
# AegisFlow never implements a custom cipher.
SECURITY_PROFILES: Dict[str, SecurityProfile] = {
    "LOW": SecurityProfile(
        level="LOW",
        data_cipher="AES-128-GCM",
        session_key_bits=128,
        key_agreement="X25519",
        kdf="HKDF-SHA256",
        authentication="AEAD integrity tag",
        enforced_controls=(
            "authenticated encryption",
            "ephemeral X25519 sender key",
            "fresh random nonce per protected file",
        ),
        recommended_controls=(
            "standard transport authentication",
            "standard session lifetime",
        ),
    ),
    "MEDIUM": SecurityProfile(
        level="MEDIUM",
        data_cipher="AES-256-GCM",
        session_key_bits=256,
        key_agreement="X25519",
        kdf="HKDF-SHA256",
        authentication="AEAD integrity tag",
        enforced_controls=(
            "authenticated encryption",
            "ephemeral X25519 sender key",
            "fresh random nonce per protected file",
            "256-bit content key",
        ),
        recommended_controls=(
            "authenticated recipient identity",
            "moderate key/session lifetime",
        ),
    ),
    "HIGH": SecurityProfile(
        level="HIGH",
        data_cipher="AES-256-GCM",
        session_key_bits=256,
        key_agreement="X25519",
        kdf="HKDF-SHA256",
        authentication="AEAD integrity tag",
        enforced_controls=(
            "authenticated encryption",
            "ephemeral X25519 sender key",
            "fresh random nonce per protected file",
            "256-bit content key",
            "policy/context metadata authenticated as AAD",
        ),
        recommended_controls=(
            "strong recipient identity verification",
            "short-lived recipient/session keys",
            "strict replay tracking at transport layer",
        ),
    ),
    "CRITICAL": SecurityProfile(
        level="CRITICAL",
        data_cipher="AES-256-GCM",
        session_key_bits=256,
        key_agreement="X25519",
        kdf="HKDF-SHA256",
        authentication="AEAD integrity tag",
        enforced_controls=(
            "authenticated encryption",
            "ephemeral X25519 sender key",
            "fresh random nonce per protected file",
            "256-bit content key",
            "policy/context metadata authenticated as AAD",
        ),
        recommended_controls=(
            "mutual endpoint authentication",
            "very short-lived recipient/session keys",
            "strict anti-replay state",
            "hybrid post-quantum key establishment when deployed end-to-end",
        ),
    ),
}


def resolve_security_profile(level: str) -> SecurityProfile:
    normalized = str(level).upper()
    if normalized not in SECURITY_PROFILES:
        raise ValueError(f"Unknown AegisFlow security level: {level}")
    return SECURITY_PROFILES[normalized]
