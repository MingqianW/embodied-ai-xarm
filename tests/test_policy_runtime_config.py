from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from policy_runtime.config import deep_merge, resolve_config


class ConfigTests(unittest.TestCase):
    def test_deep_merge_preserves_unrelated_defaults(self) -> None:
        value = deep_merge({"policy": {"host": "a", "port": 1}}, {"policy": {"port": 2}})
        self.assertEqual(value, {"policy": {"host": "a", "port": 2}})

    def test_precedence_cli_environment_local_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("policy:\n  host: local\n  port: 100\n", encoding="utf-8")
            value = resolve_config(
                {"policy": {"host": "default", "port": 1}},
                local_config=path,
                environ={"OPENPI_POLICY_HOST": "environment", "OPENPI_POLICY_PORT": "200"},
                cli_overrides={"policy": {"host": "cli"}},
            )
        self.assertEqual(value["policy"], {"host": "cli", "port": 200})


if __name__ == "__main__":
    unittest.main()
