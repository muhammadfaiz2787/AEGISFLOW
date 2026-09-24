import tempfile
import unittest
from pathlib import Path

from aegisflow.context.content_classifier import ContentClassifier
from aegisflow.context.image_semantic import gate_category_scores
from aegisflow.security.profiles import resolve_security_profile
from aegisflow.security.secure_transfer import LocalRecipientKeyStore, SecureTransferService


class ContentClassifierTests(unittest.TestCase):
    def test_financial_filename_is_detected(self):
        classifier = ContentClassifier(enable_vision=False)
        result = classifier.classify(
            filename="laporan_transaksi_bank.txt",
            content=b"monthly transaction report",
            mime_type="text/plain",
        )
        self.assertEqual(result.category, "financial")
        self.assertGreater(result.integrity, 0.9)

    def test_credential_content_is_detected(self):
        classifier = ContentClassifier(enable_vision=False)
        result = classifier.classify(
            filename="config.txt",
            content=b"api_key = super-secret-value\npassword = example",
            mime_type="text/plain",
        )
        self.assertEqual(result.category, "credentials")
        self.assertGreater(result.confidentiality, 0.95)

    def test_public_registration_poster_is_not_treated_as_sensitive_binary_text(self):
        classifier = ContentClassifier(enable_vision=False)
        fake_png = b"\x89PNG\r\n\x1a\n\x00\xffpassword = not-actual-plaintext\x00"
        result = classifier.classify(
            filename="poster_pendaftaran_acara.png",
            content=fake_png,
            mime_type="image/png",
        )
        self.assertEqual(result.category, "public")
        self.assertLess(result.sensitivity, 0.4)
        self.assertIn("binary_content:not_text_scanned", result.signals)

    def test_bts_tower_photo_defaults_to_general_context_without_vision(self):
        classifier = ContentClassifier(enable_vision=False)
        result = classifier.classify(
            filename="foto_bts_tower.jpg",
            content=b"\xff\xd8\xff\xe0binary-image-data",
            mime_type="image/jpeg",
        )
        self.assertEqual(result.category, "general")
        self.assertLess(result.confidentiality, 0.6)

    def test_weak_credential_visual_match_is_rejected_to_general(self):
        # Reproduces the real failure case: a generic lightning/vector icon was
        # weakly ranked closest to the credential prompt. Relative CLIP scores
        # are not sufficient evidence for a sensitive security classification.
        raw = {
            "credentials": 0.482,
            "financial": 0.171,
            "iot": 0.080,
            "general": 0.140,
            "public": 0.065,
            "personal": 0.035,
            "medical": 0.027,
        }
        gated, status = gate_category_scores(raw)
        self.assertEqual(max(gated, key=gated.get), "general")
        self.assertTrue(status.startswith("rejected_weak_sensitive:credentials"))

    def test_strong_unambiguous_credential_visual_match_is_accepted(self):
        raw = {
            "credentials": 0.79,
            "general": 0.10,
            "public": 0.03,
            "financial": 0.04,
            "personal": 0.02,
            "medical": 0.01,
            "iot": 0.01,
        }
        gated, status = gate_category_scores(raw)
        self.assertEqual(max(gated, key=gated.get), "credentials")
        self.assertEqual(status, "accepted:credentials")


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

    def test_cross_device_recipient_public_key_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sender_keys = LocalRecipientKeyStore(root / "sender.pem")
            receiver_keys = LocalRecipientKeyStore(root / "receiver.pem")
            sender = SecureTransferService(key_store=sender_keys)
            receiver = SecureTransferService(key_store=receiver_keys)

            plaintext = b"cross-device AegisFlow payload"
            profile = resolve_security_profile("HIGH")
            envelope, manifest = sender.protect(
                filename="shared.bin",
                content=plaintext,
                policy_name="HIGH",
                profile=profile,
                context={"category": "general", "confidence": 0.7},
                recipient_public_key_b64=receiver_keys.public_key_b64(),
            )

            recovered, metadata = receiver.unprotect(envelope)
            self.assertEqual(recovered, plaintext)
            self.assertEqual(
                manifest["recipient_public_key"],
                receiver_keys.public_key_b64(),
            )
            self.assertEqual(
                metadata["recipient_public_key"],
                receiver_keys.public_key_b64(),
            )

            with self.assertRaises(Exception):
                sender.unprotect(envelope)


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
