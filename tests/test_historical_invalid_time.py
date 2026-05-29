import unittest

import historical_runner as runner


class HistoricalInvalidTimeTests(unittest.TestCase):
    def test_invalid_timestamp_returns_error_without_using_current_time(self):
        original_cache_exists = runner.cache_exists
        called = {"cache_exists": False}

        def fake_cache_exists(*args, **kwargs):
            called["cache_exists"] = True
            return False

        runner.cache_exists = fake_cache_exists
        try:
            maps_meta, hazards_df, alerts_df, summary = runner.run_historical_analysis(
                model="AIDA",
                variable="TEC",
                start_time="not-a-timestamp",
                end_time="2024-05-10T00:00:00",
                time_step_hours=12,
                region="uk",
                resolution=10.0,
                use_cache=True,
                force_refresh=False,
                allow_api=False,
            )
        finally:
            runner.cache_exists = original_cache_exists

        self.assertFalse(called["cache_exists"])
        self.assertEqual(maps_meta, [])
        self.assertTrue(hazards_df.empty)
        self.assertTrue(alerts_df.empty)
        self.assertEqual(summary.map_count, 0)
        self.assertTrue(summary.messages)
        self.assertIn("Invalid historical time window", summary.messages[0])


if __name__ == "__main__":
    unittest.main()
