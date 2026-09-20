"""Tests for newsradar.filename_dates."""

import unittest
from datetime import date

from newsradar.filename_dates import parse_date_from_stem


class TestParseDateFromStem(unittest.TestCase):
    def test_parses_valid_stem(self):
        result = parse_date_from_stem("ai-radar-2026-05-21", "ai-radar")
        self.assertEqual(result, date(2026, 5, 21))

    def test_parses_robotics_prefix(self):
        result = parse_date_from_stem("robotics-radar-2026-01-03", "robotics-radar")
        self.assertEqual(result, date(2026, 1, 3))

    def test_returns_none_for_unparseable_suffix(self):
        result = parse_date_from_stem("ai-radar-latest", "ai-radar")
        self.assertIsNone(result)

    def test_returns_none_for_too_few_parts(self):
        result = parse_date_from_stem("ai-radar-2026-05", "ai-radar")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
