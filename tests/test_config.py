from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from niche_scout.config import Settings


class ConfigTests(unittest.TestCase):
    def test_tuple_environment_override_with_postponed_annotations(self) -> None:
        with patch.dict(
            os.environ,
            {"NICHE_SCOUT_GROUNDED_MODELS": "gemini-2.5-flash,gemini-2.5-flash-lite"},
            clear=False,
        ):
            settings = Settings.load()
        self.assertEqual(
            settings.grounded_models,
            ("gemini-2.5-flash", "gemini-2.5-flash-lite"),
        )

    def test_finalists_must_divide_evenly_into_sites(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            Settings(finalist_target=501).validate()


if __name__ == "__main__":
    unittest.main()
