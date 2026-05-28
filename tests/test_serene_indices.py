import unittest

import pandas as pd

from serene_indices import (
    add_kp_risk_columns,
    build_indices_alerts,
    classify_kp_risk,
    daily_peak_risk,
    filter_indices_by_time,
)


class SereneIndicesTests(unittest.TestCase):
    def test_classify_kp_risk_maps_g_scale_to_prototype_risk(self):
        cases = [
            (4.9, ("G0 Below storm", "Normal")),
            (5.0, ("G1 Minor", "Watch")),
            (6.7, ("G2 Moderate", "Watch")),
            (7.0, ("G3 Strong", "Warning")),
            (8.7, ("G4 Severe", "Severe")),
            (9.0, ("G5 Extreme", "Severe")),
        ]
        for kp, expected in cases:
            with self.subTest(kp=kp):
                self.assertEqual(classify_kp_risk(kp), expected)

    def test_filter_indices_by_time_is_inclusive(self):
        raw = pd.DataFrame({
            "time": [
                "2024-05-10T18:00:00Z",
                "2024-05-10T21:00:00Z",
                "2024-05-11T00:00:00Z",
            ],
            "Kp": [8.7, 8.7, 9.0],
            "ap": [300, 300, 400],
            "rAp": [0, 0, 0],
        })
        df = add_kp_risk_columns(raw)
        filtered = filter_indices_by_time(
            df,
            "2024-05-10T21:00:00Z",
            "2024-05-11T00:00:00Z",
        )
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered["Kp"].tolist(), [8.7, 9.0])

    def test_daily_peak_and_alert_rows(self):
        raw = pd.DataFrame({
            "time": [
                "2024-05-10T18:00:00Z",
                "2024-05-10T21:00:00Z",
                "2024-05-11T00:00:00Z",
            ],
            "Kp": [4.0, 7.0, 9.0],
            "ap": [27, 132, 400],
            "rAp": [0, 0, 0],
        })
        df = add_kp_risk_columns(raw)
        daily = daily_peak_risk(df)
        self.assertEqual(daily["peak_g_scale"].tolist(), ["G3 Strong", "G5 Extreme"])

        alerts = build_indices_alerts(df, minimum_kp=7.0)
        self.assertEqual(len(alerts), 2)
        self.assertIn("g_scale", alerts.columns)
        self.assertEqual(alerts.iloc[-1]["risk_level"], "Severe")


if __name__ == "__main__":
    unittest.main()
