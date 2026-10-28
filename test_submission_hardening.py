"""Submission-hardening regression tests for AegisOmni.

These tests cover deterministic behavior and mocked provider responses. They do not
replace a live Meta Model API integration test or a browser-level security test.
"""
import json
import os
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from app import app
from src.inspector_client import inspect_equipment_with_meta_api, parse_sam_segmentation_boxes

VALID_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAElFTkSuQmCC"


class TestSubmissionHardening(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_sam_documented_xyxy_array_normalizes_to_xywh(self):
        boxes = parse_sam_segmentation_boxes(
            [{"label": "Gauge", "score": 0.9, "box": [20, 10, 60, 50]}],
            image_width=100,
            image_height=100,
        )
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["x"], 0.2)
        self.assertAlmostEqual(boxes[0]["y"], 0.1)
        self.assertAlmostEqual(boxes[0]["w"], 0.4)
        self.assertAlmostEqual(boxes[0]["h"], 0.4)

    def test_sam_output_text_accepts_json_array(self):
        boxes = parse_sam_segmentation_boxes(
            {"output_text": json.dumps([{"label": "Bolt", "box": [10, 10, 30, 40]}])},
            image_width=100,
            image_height=100,
        )
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["w"], 0.2)
        self.assertAlmostEqual(boxes[0]["h"], 0.3)

    def test_sam_documented_special_tokens_parse_and_do_not_invent_confidence(self):
        output_text = "<0f>0<|box;x1=20;y1=10;x2=60;y2=50;w=100;h=100|><|mask;x=0;y=0;data=1|>"
        boxes = parse_sam_segmentation_boxes({"output_text": output_text})
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["x"], 0.2)
        self.assertAlmostEqual(boxes[0]["y"], 0.1)
        self.assertAlmostEqual(boxes[0]["w"], 0.4)
        self.assertAlmostEqual(boxes[0]["h"], 0.4)
        self.assertIsNone(boxes[0]["confidence"])

    def test_sam_live_endpoint_uses_documented_stream_request_and_parses_sse(self):
        captured = {}
        token = "<0f>0<|box;x1=20;y1=10;x2=60;y2=50;w=100;h=100|><|mask;x=0;y=0;data=1|>"

        class MockResponse:
            def __init__(self, url):
                self.url = url
                self.headers = {"content-type": "text/event-stream"} if "/responses" in url else {"content-type": "application/json"}

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": json.dumps({
                    "visible_observations": [{"component": "Gauge", "observation": "Gauge face visible", "confidence": 0.88}],
                    "uncertainty": "A photo cannot verify calibration.",
                    "next_steps": ["Ask a qualified technician to verify the reading."],
                })}}]}

            def iter_lines(self, decode_unicode=True):
                events = [
                    {"type": "response.output_text.delta", "delta": token[:32]},
                    {"type": "response.output_text.delta", "delta": token[32:]},
                    {"type": "response.completed", "response": {"output_text": token}},
                ]
                return iter(["data: " + json.dumps(event) for event in events] + ["data: [DONE]"])

        class MockSession:
            def post(self, url, **kwargs):
                captured[url] = kwargs
                return MockResponse(url)

        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "mocked-test-key"
        try:
            result = inspect_equipment_with_meta_api(
                user_question="Inspect the visible pressure gauge.",
                image_base64=VALID_PNG_B64,
                component_prompt="pressure gauge",
                enable_segmentation=True,
                mode="live",
                session=MockSession(),
                image_width=100,
                image_height=100,
            )
            self.assertEqual(result["status"], "LIVE_API_SUCCESS")
            self.assertEqual(result["segmentation"]["status"], "SUCCESS")
            box = result["segmentation"]["boxes"][0]
            self.assertAlmostEqual(box["x"], 0.2)
            self.assertAlmostEqual(box["y"], 0.1)
            sam_request = captured["https://api.meta.ai/v1/responses"]
            self.assertTrue(sam_request["stream"])
            self.assertTrue(sam_request["json"]["stream"])
            self.assertEqual(sam_request["json"]["input"][0]["type"], "message")
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_api_rejects_unknown_mode(self):
        response = self.client.post(
            "/api/inspect",
            json={"query": "Inspect pump", "mode": "pretend-live"},
        )
        self.assertEqual(response.status_code, 422)

    def test_frontend_escapes_untrusted_live_fields(self):
        html_path = Path(__file__).resolve().parents[1] / "aegis_omni_meta_hackathon.html"
        html = html_path.read_text(encoding="utf-8")
        self.assertIn("function escapeHtml(value)", html)
        self.assertIn("escapeHtml(state.liveError.message)", html)
        self.assertIn("escapeHtml(obs.component)", html)
        self.assertIn("escapeHtml(obs.observation)", html)
        self.assertIn("escapeHtml(item)", html)
        self.assertIn("escapeHtml(state.uploadedFileName)", html)
        self.assertIn("LIVE_REQUIRES_UPLOADED_IMAGE", html)
        self.assertIn("custom photo was not live-analyzed", html)
        self.assertIn("live speech recognition is not connected", html)
        self.assertNotIn("${obs.component}", html)
        self.assertNotIn("${obs.observation}", html)

    def test_sam_response_without_usable_boxes_is_not_reported_as_success(self):
        class DummyNoBoxesSession:
            def post(self, url, *args, **kwargs):
                class Response:
                    def raise_for_status(self):
                        return None

                    def json(self):
                        if "/chat/completions" in url:
                            return {
                                "choices": [{"message": {"content": json.dumps({
                                    "visible_observations": [{
                                        "component": "Gauge",
                                        "observation": "Glass appears intact in this view",
                                        "confidence": 0.8,
                                    }],
                                    "uncertainty": "A photograph cannot confirm calibration or pressure.",
                                    "next_steps": ["Have a qualified technician verify the reading under site procedures."],
                                })}}]
                            }
                        return {"output_text": "No coordinates were returned."}
                return Response()

        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "mocked-test-key"
        try:
            result = inspect_equipment_with_meta_api(
                user_question="Inspect the visible pressure gauge.",
                image_base64=VALID_PNG_B64,
                component_prompt="pressure gauge",
                enable_segmentation=True,
                mode="live",
                session=DummyNoBoxesSession(),
                image_width=16,
                image_height=16,
            )
            self.assertEqual(result["status"], "LIVE_API_SUCCESS")
            self.assertEqual(result["segmentation"]["status"], "NO_VALID_BOXES_RETURNED")
            self.assertEqual(result["segmentation"]["boxes"], [])
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main(verbosity=2)
