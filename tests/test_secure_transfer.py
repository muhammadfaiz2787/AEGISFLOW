import tempfile
import unittest
from pathlib import Path

from aegisflow.context.content_classifier import ContentClassifier
from aegisflow.security.profiles import resolve_security_profile
from aegisflow.security.secure_transfer import LocalRecipientKeyStore, SecureTransferService


class ContentClassifierTests(unittest.TestCase):
    def test_financial_filename_is_detected(self):
        classifier = ContentClassifier()
        result = classifier.classify(
            filename="laporan_transaksi_bank.txt",
            content=b"monthly transaction report",
            mime_type="text/plain",
        )
        self.assertEqual(result.category, "financial")
        self.assertGreater(result.integrity, 0.9)

    def test_credential_content_is_detected(self):
        classifier = ContentClassifier()
        result = classifier.classify(
            filename="config.txt",
            content=b"api_key = super-secret-value\npassword = example",
            mime_type="text/plain",
        )
        self.assertEqual(result.category, "credentials")
        self.assertGreater(result.confidentiality, 0.95)


class SecureTransferTests(unittest.TestCase):
    def test_round_trip_authenticated_encryption(self):
        with tempfile.TemporaryDirectory() as directory:
            key_store = LocalRecipientKeyStore(Path(directory) / "recipient.pem")
            service = SecureTransferService(key_store=key_store)
            plaintext = b"AegisFlow secure transfer test payload"
            profile = resolve_security_profile("HIGH")
            envelope, manifest = service.protect(
                filename="private.txt",
                content=plaintext,
                policy_name="HIGH",
                profile=profile,
                context={"category": "personal", "confidence": 0.9},
            )

            recovered, metadata = service.unprotect(envelope)
            self.assertEqual(recovered, plaintext)
            self.assertEqual(metadata["policy"], "HIGH")
            self.assertEqual(manifest["security_profile"]["data_cipher"], "AES-256-GCM")

    def test_modified_ciphertext_fails_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            key_store = LocalRecipientKeyStore(Path(directory) / "recipient.pem")
            service = SecureTransferService(key_store=key_store)
            envelope, _ = service.protect(
                filename="payload.bin",
                content=b"original",
                policy_name="MEDIUM",
                profile=resolve_security_profile("MEDIUM"),
                context={"category": "general", "confidence": 0.6},
            )
            tampered = bytearray(envelope)
            tampered[-1] ^= 0x01
            with self.assertRaises(Exception):
                service.unprotect(bytes(tampered))


if __name__ == "__main__":
    unittest.main()
