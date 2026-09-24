from __future__ import annotations

import base64
import json
import os
import struct
from pathlib import Path
from typing import Dict, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from aegisflow.security.profiles import SecurityProfile


MAGIC = b"AEGISFLOW1\n"
KEY_WRAP_AAD = b"aegisflow-key-wrap-v1"


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _canonical_json(payload: Dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class LocalRecipientKeyStore:
    """Local receiver key for the single-host Secure Transfer prototype.

    The private key is generated at runtime and stored under data/keys. It must never
    be committed to source control. In a multi-device deployment, each receiver would
    own its own private key and expose only its public key.
    """

    def __init__(self, private_key_path: str | Path = "data/keys/aegisflow_x25519_private.pem"):
        self.private_key_path = Path(private_key_path)
        self.private_key = self._load_or_create()

    def _load_or_create(self) -> x25519.X25519PrivateKey:
        if self.private_key_path.exists():
            return serialization.load_pem_private_key(
                self.private_key_path.read_bytes(),
                password=None,
            )

        self.private_key_path.parent.mkdir(parents=True, exist_ok=True)
        private_key = x25519.X25519PrivateKey.generate()
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        self.private_key_path.write_bytes(pem)
        return private_key

    def public_key_bytes(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_key_b64(self) -> str:
        return _b64(self.public_key_bytes())


class SecureTransferService:
    def __init__(self, key_store: LocalRecipientKeyStore | None = None):
        self.key_store = key_store or LocalRecipientKeyStore()

    @staticmethod
    def _derive_wrap_key(shared_secret: bytes, salt: bytes) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"aegisflow-secure-transfer-v1",
        ).derive(shared_secret)

    def protect(
        self,
        *,
        filename: str,
        content: bytes,
        policy_name: str,
        profile: SecurityProfile,
        context: Dict[str, object],
        recipient_public_key_b64: str | None = None,
    ) -> Tuple[bytes, Dict[str, object]]:
        original_filename = Path(filename or "unnamed.bin").name

        if recipient_public_key_b64:
            try:
                recipient_public_bytes = _unb64(recipient_public_key_b64)
                if len(recipient_public_bytes) != 32:
                    raise ValueError("X25519 public key must decode to 32 bytes.")
                recipient_public = x25519.X25519PublicKey.from_public_bytes(
                    recipient_public_bytes
                )
            except Exception as exc:
                raise ValueError("Invalid recipient X25519 public key.") from exc
        else:
            recipient_public = self.key_store.private_key.public_key()
            recipient_public_bytes = self.key_store.public_key_bytes()
        ephemeral_private = x25519.X25519PrivateKey.generate()
        ephemeral_public = ephemeral_private.public_key()

        shared_secret = ephemeral_private.exchange(recipient_public)
        salt = os.urandom(16)
        wrapping_key = self._derive_wrap_key(shared_secret, salt)

        content_key = os.urandom(profile.session_key_bits // 8)
        wrap_nonce = os.urandom(12)
        wrapped_key = AESGCM(wrapping_key).encrypt(
            wrap_nonce,
            content_key,
            KEY_WRAP_AAD,
        )

        metadata = {
            "version": 1,
            "original_filename": original_filename,
            "policy": policy_name,
            "security_profile": profile.to_dict(),
            "content_context": context,
            "recipient_public_key": _b64(recipient_public_bytes),
        }
        data_aad = _canonical_json(metadata)
        data_nonce = os.urandom(12)
        ciphertext = AESGCM(content_key).encrypt(
            data_nonce,
            content,
            data_aad,
        )

        header = {
            "metadata": metadata,
            "ephemeral_public_key": _b64(
                ephemeral_public.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            ),
            "salt": _b64(salt),
            "wrap_nonce": _b64(wrap_nonce),
            "wrapped_key": _b64(wrapped_key),
            "data_nonce": _b64(data_nonce),
        }
        header_bytes = _canonical_json(header)
        envelope = MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + ciphertext

        manifest = {
            "original_filename": original_filename,
            "protected_filename": f"{original_filename}.aegis",
            "policy": policy_name,
            "security_profile": profile.to_dict(),
            "content_context": context,
            "recipient_public_key": _b64(recipient_public_bytes),
            "envelope_version": 1,
        }
        return envelope, manifest

    def unprotect(self, envelope: bytes) -> Tuple[bytes, Dict[str, object]]:
        if not envelope.startswith(MAGIC):
            raise ValueError("Not an AegisFlow Secure Transfer envelope.")

        cursor = len(MAGIC)
        if len(envelope) < cursor + 4:
            raise ValueError("Truncated AegisFlow envelope.")

        header_length = struct.unpack(">I", envelope[cursor:cursor + 4])[0]
        cursor += 4
        header_end = cursor + header_length
        if len(envelope) <= header_end:
            raise ValueError("Invalid AegisFlow envelope header length.")

        header = json.loads(envelope[cursor:header_end].decode("utf-8"))
        ciphertext = envelope[header_end:]

        ephemeral_public = x25519.X25519PublicKey.from_public_bytes(
            _unb64(header["ephemeral_public_key"])
        )
        shared_secret = self.key_store.private_key.exchange(ephemeral_public)
        wrapping_key = self._derive_wrap_key(
            shared_secret,
            _unb64(header["salt"]),
        )
        content_key = AESGCM(wrapping_key).decrypt(
            _unb64(header["wrap_nonce"]),
            _unb64(header["wrapped_key"]),
            KEY_WRAP_AAD,
        )

        metadata = header["metadata"]
        plaintext = AESGCM(content_key).decrypt(
            _unb64(header["data_nonce"]),
            ciphertext,
            _canonical_json(metadata),
        )
        return plaintext, metadata
