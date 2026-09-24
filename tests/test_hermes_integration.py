import tempfile
import unittest
from pathlib import Path

from aegisflow.integrations.hermes_client import HermesAegisFlowClient


class HermesBridgeSafetyTests(unittest.TestCase):
    def test_allowed_relative_path_is_resolved_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "sample.txt"
            file_path.write_text("hello", encoding="utf-8")

            client = HermesAegisFlowClient(allowed_root=root)
            resolved = client._resolve_allowed_path("sample.txt", must_exist=True)

            self.assertEqual(resolved, file_path.resolve())

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "allowed"
            root.mkdir()
            outside = Path(directory) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")

            client = HermesAegisFlowClient(allowed_root=root)

            with self.assertRaises(ValueError):
                client._resolve_allowed_path(outside, must_exist=True)

    def test_existing_output_is_not_overwritten_by_default_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.aegis"
            output.write_bytes(b"existing")

            client = HermesAegisFlowClient(allowed_root=root)
            resolved = client._resolve_allowed_path(output, must_exist=False)

            self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
