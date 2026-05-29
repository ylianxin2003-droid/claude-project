import unittest

import config
import map_builder
from serene_client import SereneClient


class DynamicConfigTests(unittest.TestCase):
    def test_serene_client_reads_current_config_values(self):
        old_base = config.SERENE_API_BASE_URL
        old_token = config.SERENE_API_TOKEN
        old_timeout = config.SERENE_API_TIMEOUT
        old_scheme = config.SERENE_AUTH_SCHEME

        try:
            config.SERENE_API_BASE_URL = "https://example.invalid"
            config.SERENE_API_TOKEN = "dynamic-token"
            config.SERENE_API_TIMEOUT = 7
            config.SERENE_AUTH_SCHEME = "Token"

            client = SereneClient()
            self.assertEqual(client.base_url, "https://example.invalid")
            self.assertEqual(client.token, "dynamic-token")
            self.assertEqual(client.timeout, 7)
            self.assertEqual(client.auth_scheme, "Token")
        finally:
            config.SERENE_API_BASE_URL = old_base
            config.SERENE_API_TOKEN = old_token
            config.SERENE_API_TIMEOUT = old_timeout
            config.SERENE_AUTH_SCHEME = old_scheme

    def test_map_builder_uses_current_config_token_not_stale_import(self):
        old_token = config.SERENE_API_TOKEN
        old_stale = getattr(map_builder, "SERENE_API_TOKEN", None)

        try:
            if hasattr(map_builder, "SERENE_API_TOKEN"):
                map_builder.SERENE_API_TOKEN = ""
            config.SERENE_API_TOKEN = "dynamic-token"

            _df, message = map_builder.build_fixed_map(
                model="AIDA",
                timestamp="2024-05-10T00:00:00",
                variable="TEC",
                region="uk",
                resolution=10.0,
                use_cache=False,
                force_refresh=True,
                max_points=0,
            )
        finally:
            config.SERENE_API_TOKEN = old_token
            if old_stale is not None:
                map_builder.SERENE_API_TOKEN = old_stale

        self.assertNotIn("token is not configured", message)
        self.assertIn("Too many API calls", message)


if __name__ == "__main__":
    unittest.main()
