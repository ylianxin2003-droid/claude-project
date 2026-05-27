import unittest

import pandas as pd

import historical_runner as runner


class HistoricalCacheOnlyTests(unittest.TestCase):
    def test_cache_only_cache_miss_skips_without_calling_builder(self):
        original_build_fixed_map = runner.build_fixed_map
        original_cache_exists = getattr(runner, "cache_exists", None)

        called = {"build_fixed_map": False}

        def fake_build_fixed_map(*args, **kwargs):
            called["build_fixed_map"] = True
            return pd.DataFrame(), "builder called"

        runner.build_fixed_map = fake_build_fixed_map
        runner.cache_exists = lambda *args, **kwargs: False

        try:
            maps_meta, hazards_df, alerts_df, summary = runner.run_historical_analysis(
                model="AIDA",
                variable="TEC",
                start_time="2024-05-10T00:00:00",
                end_time="2024-05-10T01:00:00",
                time_step_hours=12,
                region="uk",
                resolution=10.0,
                use_cache=True,
                force_refresh=False,
                allow_api=False,
            )
        finally:
            runner.build_fixed_map = original_build_fixed_map
            if original_cache_exists is None:
                delattr(runner, "cache_exists")
            else:
                runner.cache_exists = original_cache_exists

        self.assertFalse(called["build_fixed_map"])
        self.assertEqual(maps_meta, [])
        self.assertTrue(hazards_df.empty)
        self.assertTrue(alerts_df.empty)
        self.assertEqual(summary.map_count, 0)
        self.assertEqual(summary.failures, 1)
        self.assertIn("skipped: no cache in cache-only mode", summary.messages[0])


if __name__ == "__main__":
    unittest.main()
