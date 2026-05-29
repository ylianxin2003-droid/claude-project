import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import map_cache


class CacheMetadataTests(unittest.TestCase):
    def test_list_cached_maps_preserves_underscore_variable_and_iso_timestamp(self):
        original_root = map_cache._CACHE_ROOT

        with TemporaryDirectory() as tmpdir:
            map_cache._CACHE_ROOT = Path(tmpdir)
            try:
                path = map_cache.get_cache_path(
                    model="AIDA",
                    variable="MUF3000_depression",
                    timestamp="2024-05-10T00:00:00",
                    resolution=10.0,
                    region="uk",
                )
                path.write_text("placeholder", encoding="utf-8")

                listed = map_cache.list_cached_maps()
            finally:
                map_cache._CACHE_ROOT = original_root

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["model"], "AIDA")
        self.assertEqual(listed[0]["variable"], "MUF3000_depression")
        self.assertEqual(listed[0]["timestamp"], "2024-05-10T00:00:00")
        self.assertEqual(listed[0]["resolution"], 10.0)
        self.assertEqual(listed[0]["region"], "uk")


if __name__ == "__main__":
    unittest.main()
