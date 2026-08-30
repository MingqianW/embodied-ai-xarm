from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy_runtime.episode_logging import EpisodeLogger


class EpisodeLoggingTests(unittest.TestCase):
    def test_saved_array_has_schema_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = EpisodeLogger(Path(temporary), simulator="test")
            path = logger.save_array(
                "step/actions.npy",
                np.zeros((10, 7), dtype=np.float32),
            )
            metadata_path = path.with_name(f"{path.name}.meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(metadata["shape"], [10, 7])
            self.assertEqual(metadata["dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
