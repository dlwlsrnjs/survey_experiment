import json
import unittest
from unittest.mock import patch

from survey_exp.client import OpenAICompatibleClient
from survey_exp.config import ModelConfig


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class ClientTests(unittest.TestCase):
    def test_null_compatible_api_content_is_a_recorded_error(self):
        client = OpenAICompatibleClient(ModelConfig(name="test", base_url="http://local/v1"))
        payload = {
            "id": "response-1",
            "choices": [{"message": {"content": None}}],
            "usage": {"completion_tokens": 1},
        }
        with patch("urllib.request.urlopen", return_value=_Response(payload)):
            result = client.complete_messages([{"role": "user", "content": "hello"}])
        self.assertEqual(result.text, "")
        self.assertEqual(result.error["type"], "InvalidResponseContent")
        self.assertEqual(result.response_id, "response-1")
        self.assertEqual(result.raw, payload)


if __name__ == "__main__":
    unittest.main()
