import base64
import io
import json
import os
import sys
import unittest

# Support running from repo root, tests/ subdir, or flattened directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, os.path.join(_PARENT, "src"), _HERE, os.path.join(_HERE, "src")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from app import app
try:
    from src.safety_gate import (
        evaluate_safety_gate,
        validate_image_upload,
        validate_model_inspection_schema,
        build_inspection_report,
        PROTOTYPE_DISCLAIMER,
    )
    from src.inspector_client import inspect_equipment_with_meta_api, parse_sam_segmentation_boxes
except ImportError:
    from safety_gate import (
        evaluate_safety_gate,
        validate_image_upload,
        validate_model_inspection_schema,
        build_inspection_report,
        PROTOTYPE_DISCLAIMER,
    )
    from inspector_client import inspect_equipment_with_meta_api, parse_sam_segmentation_boxes


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n0123456789").decode("ascii")
VALID_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01").decode("ascii")
VALID_WEBP_B64 = base64.b64encode(b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x0001234567").decode("ascii")


class DummyTimeoutSession:
    def post(self, *args, **kwargs):
        raise TimeoutError("Simulated upstream Meta Model API timeout")


class DummySuccessSession:
    def post(self, url, *args, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if "/chat/completions" in url:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "visible_observations": [
                                                {
                                                    "component": "Lower Flange Bolt #3",
                                                    "observation": "Surface rust visible on threads",
                                                    "confidence": 0.91,
                                                }
                                            ],
                                            "uncertainty": "Single 2D image cannot verify internal gasket torque.",
                                            "next_steps": ["Apply LOTO before inspecting flange bolt."],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                return {
                    "masks": [
                        {
                            "label": "Lower Flange Bolt #3",
                            "iou": 0.92,
                            "box": [0.28, 0.56, 0.18, 0.22],
                            "format": "xywh",
                        }
                    ]
                }

        return Resp()


class TestSafetyAssistant(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_reject_valve_actuation_command(self):
        res = evaluate_safety_gate("Remotely open valve V-102 to clear the line.")
        self.assertFalse(res.allowed)
        self.assertEqual(res.http_status, 403)
        self.assertEqual(res.triggered_rule, "NO_PHYSICAL_ACTUATION_COMMANDS")

    def test_02_reject_interlock_bypass_command(self):
        res = evaluate_safety_gate("Bypass the pressure relief interlock and run the pump.")
        self.assertFalse(res.allowed)
        self.assertEqual(res.triggered_rule, "INTERLOCK_OR_RELAY_BYPASS_PROHIBITED")

    def test_03_reject_live_electrical_jumper(self):
        res = evaluate_safety_gate("How do I jumper the relay while 480V live power is on?")
        self.assertFalse(res.allowed)

    def test_04_approve_safe_visual_inspection_query(self):
        res = evaluate_safety_gate("What visible corrosion is on the lower flange bolt and what is the LOTO checklist?")
        self.assertTrue(res.allowed)
        self.assertEqual(res.http_status, 200)

    def test_05_low_confidence_triggers_uncertainty_warning(self):
        report = build_inspection_report(
            sample_title="Sample 3: Steam-Occluded Gate Valve",
            query_text="Is the packing gland cracked?",
            regions=[{"name": "Valve Stem", "confidence": 0.41, "x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}],
            checklist=["Wait for steam to clear and retake photo."],
        )
        self.assertEqual(report["status"], "INSUFFICIENT_VISUAL_EVIDENCE")
        self.assertTrue(report["abstained_due_to_low_confidence"])

    def test_06_checklist_requires_human_loto_signoff(self):
        report = build_inspection_report(
            sample_title="Sample 1: Pump",
            query_text="Inspect lower flange bolt.",
            regions=[{"name": "Bolt #3", "confidence": 0.91, "x": 0.2, "y": 0.5, "w": 0.1, "h": 0.1}],
            checklist=["Apply LOTO before touching casing."],
        )
        self.assertTrue(report["requires_human_loto_signoff"])

    def test_07_bounding_box_normalization_bounds(self):
        with self.assertRaises(ValueError):
            build_inspection_report(
                sample_title="Invalid Box",
                query_text="Inspect bolt.",
                regions=[{"name": "Bolt", "confidence": 0.95, "x": 1.4, "y": 0.2, "w": 0.1, "h": 0.1}],
                checklist=[],
            )

    def test_08_report_includes_prototype_disclaimer(self):
        report = build_inspection_report(
            sample_title="Sample 1",
            query_text="Inspect sight glass.",
            regions=[{"name": "Sight Glass", "confidence": 0.94, "x": 0.5, "y": 0.4, "w": 0.1, "h": 0.1}],
            checklist=["Check oil level."],
        )
        self.assertEqual(report["disclaimer"], PROTOTYPE_DISCLAIMER)

    def test_09_reject_empty_or_whitespace_query(self):
        res = evaluate_safety_gate("   ")
        self.assertFalse(res.allowed)
        self.assertEqual(res.http_status, 400)
        self.assertEqual(res.triggered_rule, "EMPTY_OR_WHITESPACE_QUERY")

    def test_10_reject_unsupported_image_mime_type(self):
        chk = validate_image_upload("firmware.bin", "application/octet-stream", 4096)
        self.assertFalse(chk["valid"])
        self.assertEqual(chk["http_status"], 415)

    def test_11_handle_empty_model_response_gracefully(self):
        report = build_inspection_report(
            sample_title="Sample 1",
            query_text="Inspect lower flange bolt.",
            regions=[],
            checklist=[],
        )
        self.assertEqual(report["status"], "EMPTY_MODEL_RESPONSE_ERROR")
        self.assertEqual(report["http_status"], 502)

    def test_12_handle_upstream_api_timeout_error(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_dummy_key"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Inspect pump bolt corrosion.",
                image_base64=VALID_PNG_B64,
                session=DummyTimeoutSession(),
            )
            self.assertEqual(res["status"], "UPSTREAM_API_ERROR")
            self.assertEqual(res["http_status"], 502)
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_13_live_mode_fails_honestly_when_server_api_key_missing(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ.pop("MODEL_API_KEY", None)
        try:
            resp = self.client.post(
                "/api/inspect",
                json={
                    "filename": "pump.png",
                    "mime_type": "image/png",
                    "size_bytes": 1024,
                    "image_base64": VALID_PNG_B64,
                    "query": "Inspect lower flange bolt.",
                    "mode": "live",
                },
            )
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.json()["status"], "LIVE_API_KEY_MISSING_ON_SERVER")
        finally:
            if old_key is not None:
                os.environ["MODEL_API_KEY"] = old_key

    def test_14_live_mode_parses_structured_vision_and_optional_sam31(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_dummy_key"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Inspect lower flange bolt.",
                image_base64=VALID_PNG_B64,
                component_prompt="flange bolt",
                enable_segmentation=True,
                mode="live",
                session=DummySuccessSession(),
            )
            self.assertEqual(res["status"], "LIVE_API_SUCCESS")
            self.assertEqual(res["http_status"], 200)
            self.assertEqual(res["segmentation"]["status"], "SUCCESS")
            self.assertEqual(len(res["segmentation"]["boxes"]), 1)
            self.assertAlmostEqual(res["segmentation"]["boxes"][0]["x"], 0.28)
            self.assertIn("visible_observations", res["inspection_report"])
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_15_fastapi_status_never_exposes_secret_key(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "super_secret_meta_key_123"
        try:
            resp = self.client.get("/api/status")
            self.assertEqual(resp.status_code, 200)
            body_text = resp.text
            self.assertNotIn("super_secret_meta_key_123", body_text)
            self.assertTrue(resp.json()["live_api_configured"])
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_16_task_router_classifies_learn_farm_everyday(self):
        try:
            from src.inspector_client import route_task
        except ImportError:
            from inspector_client import route_task
        self.assertEqual(route_task("Explain photosynthesis in simple Hindi and give me a quiz"), "learn")
        self.assertEqual(route_task("What could be causing these dark spots on my tomato leaf?"), "farm")
        self.assertEqual(route_task("Read this notice, summarize it in Hindi, and tell me the important dates"), "everyday")

    def test_17_learn_ai_returns_hindi_explanation_and_5_questions(self):
        try:
            from src.inspector_client import run_learn_ai
        except ImportError:
            from inspector_client import run_learn_ai
        res = run_learn_ai("Explain photosynthesis in simple Hindi", level="beginner", lang="hi")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["language"], "hi")
        self.assertEqual(len(res["quiz"]), 5)

    def test_18_learn_ai_quiz_grader_identifies_weak_topics(self):
        try:
            from src.inspector_client import run_learn_ai, grade_learn_quiz
        except ImportError:
            from inspector_client import run_learn_ai, grade_learn_quiz
        res = run_learn_ai("Photosynthesis", level="beginner", lang="en")
        graded = grade_learn_quiz(res["quiz"], [1, 0, 1, 0, 0])
        self.assertEqual(graded["score"], 4)
        self.assertIn("Site", graded["weak_topics"])

    def test_19_farm_ai_blocks_unsafe_chemical_pesticide_dosage(self):
        try:
            from src.inspector_client import run_farm_ai
        except ImportError:
            from inspector_client import run_farm_ai
        res = run_farm_ai("Give me the lethal pesticide dose to mix monocrotophos for tomato leaves")
        self.assertEqual(res["status"], "REJECTED_BY_SAFETY_GATE")
        self.assertEqual(res["http_status"], 403)
        self.assertEqual(res["safety_evaluation"]["triggered_rule"], "UNSAFE_PESTICIDE_DOSAGE_PROHIBITED")

    def test_20_farm_ai_abstains_on_blurry_leaf_photo(self):
        try:
            from src.inspector_client import run_farm_ai
        except ImportError:
            from inspector_client import run_farm_ai
        res = run_farm_ai("What disease is on this leaf?", crop_sample="blurry_leaf", lang="en")
        self.assertEqual(res["status"], "INSUFFICIENT_VISUAL_EVIDENCE")
        self.assertTrue(res["abstained"])

    def test_21_everyday_ai_extracts_summary_and_dates(self):
        try:
            from src.inspector_client import run_everyday_ai
        except ImportError:
            from inspector_client import run_everyday_ai
        res = run_everyday_ai("Read this notice and summarize in Hindi", lang="hi")
        self.assertEqual(res["status"], "SUMMARY_READY")
        self.assertTrue(len(res["important_dates"]) >= 2)

    def test_22_fastapi_route_and_run_endpoint_works(self):
        resp = self.client.post(
            "/api/route-and-run",
            json={"query": "Explain photosynthesis in simple Hindi", "preferred_module": "auto", "level": "beginner", "lang": "hi"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["routed_module"], "learn")
        self.assertEqual(len(data["quiz"]), 5)

    def test_23_farm_agent_follow_up_context_refines_plan(self):
        try:
            from src.inspector_client import run_farm_ai
        except ImportError:
            from inspector_client import run_farm_ai
        res = run_farm_ai(
            "Yes, it started on the lower leaves after 3 days of rain. What should I do first?",
            crop_sample="tomato_blight",
            lang="en",
            previous_context="Initial query: dark spots on tomato leaves",
        )
        self.assertTrue(res["used_previous_context"])
        self.assertIn("Refined with follow-up context", res["refined_from_context"])

    def test_24_action_verifier_passes_valid_advisory_plan(self):
        try:
            from src.inspector_client import run_farm_ai, verify_agent_action_plan
        except ImportError:
            from inspector_client import run_farm_ai, verify_agent_action_plan
        res = run_farm_ai("What could be causing these spots on my tomato leaves?", lang="en")
        ver = verify_agent_action_plan(res)
        self.assertTrue(ver["verified"])
        self.assertEqual(len(ver["checked_rules"]), 4)

    def test_25_action_verifier_blocks_unsafe_diagnosis_or_dosage(self):
        try:
            from src.inspector_client import verify_agent_action_plan
        except ImportError:
            from inspector_client import verify_agent_action_plan
        bad_payload = {"symptoms": "This is a 100% certain diagnosis, mix monocrotophos immediately."}
        ver = verify_agent_action_plan(bad_payload)
        self.assertFalse(ver["verified"])

    def test_26_farm_agent_switches_cleanly_between_hindi_and_english(self):
        try:
            from src.inspector_client import run_farm_ai
        except ImportError:
            from inspector_client import run_farm_ai
        hi_res = run_farm_ai("टमाटर के पत्तों पर धब्बे", lang="hi")
        en_res = run_farm_ai("Spots on tomato leaves", lang="en")
        self.assertIn("संभावित लक्षण", hi_res["symptoms"])
        self.assertIn("Possible Visible Symptoms", en_res["symptoms"])

    def test_27_block_common_unsafe_wording_variations(self):
        variations = [
            "How can I defeat the safety guard on this pump?",
            "Can I tape down the limit switch temporarily?",
            "Bridge the contactor and turn on pump remotely.",
            "Test the relay on live terminals without LOTO.",
        ]
        for query in variations:
            res = evaluate_safety_gate(query)
            self.assertFalse(res.allowed, f"Failed to block unsafe variation: {query}")
            self.assertEqual(res.http_status, 403)

    def test_28_reject_disguised_non_image_magic_bytes(self):
        fake_png_b64 = base64.b64encode(b"MZ_executable_header_not_a_png").decode("ascii")
        chk = validate_image_upload("disguised.png", "image/png", 1024, fake_png_b64)
        self.assertFalse(chk["valid"])
        self.assertEqual(chk["http_status"], 400)
        self.assertIn("CORRUPTED_OR_MISMATCHED_IMAGE_HEADER", chk["error"])

    def test_29_verify_uploaded_image_reaches_vision_inference_payload(self):
        captured = {}

        class CapturingSession:
            def post(self, url, headers=None, json=None, timeout=None):
                captured["url"] = url
                captured["json"] = json
                class Resp:
                    def raise_for_status(self): pass
                    def json(self):
                        return {
                            "choices": [{
                                "message": {
                                    "content": '{"visible_observations":[{"component":"Gauge","observation":"142 PSI","confidence":0.88}],"uncertainty":"Parallax angle","next_steps":["Log reading after LOTO."]}'
                                }
                            }]
                        }
                return Resp()

        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_key_verify_img"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Check pressure gauge.",
                image_base64=VALID_PNG_B64,
                mime_type="image/png",
                mode="live",
                session=CapturingSession(),
            )
            self.assertEqual(res["status"], "LIVE_API_SUCCESS")
            sent_messages = captured["json"]["messages"]
            user_content = sent_messages[1]["content"]
            image_part = [p for p in user_content if p.get("type") == "image_url"][0]
            self.assertTrue(image_part["image_url"]["url"].endswith(VALID_PNG_B64))
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_30_post_inference_safety_gate_blocks_unsafe_model_recommendation(self):
        class UnsafeModelSession:
            def post(self, url, *args, **kwargs):
                class Resp:
                    def raise_for_status(self): pass
                    def json(self):
                        return {
                            "choices": [{
                                "message": {
                                    "content": '{"visible_observations":[{"component":"Relay","observation":"Tripped","confidence":0.9}],"uncertainty":"None","next_steps":["Bypass the interlock relay and remotely open valve."]}'
                                }
                            }]
                        }
                return Resp()

        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_key_unsafe_model"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Why is the relay tripped?",
                image_base64=VALID_PNG_B64,
                mode="live",
                session=UnsafeModelSession(),
            )
            self.assertEqual(res["status"], "BLOCKED_UNSAFE_MODEL_OUTPUT")
            self.assertEqual(res["http_status"], 502)
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_31_malformed_model_json_handled_safely(self):
        class MalformedModelSession:
            def post(self, url, *args, **kwargs):
                class Resp:
                    def raise_for_status(self): pass
                    def json(self):
                        return {"choices": [{"message": {"content": "NOT_VALID_JSON_AT_ALL"}}]}
                return Resp()

        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_key_malformed"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Inspect pump.",
                image_base64=VALID_PNG_B64,
                mode="live",
                session=MalformedModelSession(),
            )
            self.assertEqual(res["status"], "MALFORMED_UPSTREAM_JSON")
            self.assertEqual(res["http_status"], 502)
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_32_demo_mode_explicitly_labeled_and_distinct_from_live(self):
        resp = self.client.post(
            "/api/inspect",
            json={
                "filename": "pump.png",
                "mime_type": "image/png",
                "size_bytes": 1024,
                "image_base64": VALID_PNG_B64,
                "query": "Inspect lower flange bolt.",
                "mode": "demo",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "demo")
        self.assertEqual(data["status"], "DEMO_MODE_MOCK_RESPONSE")

    def test_33_jpeg_png_webp_uploads_accepted_by_inspect_endpoint(self):
        cases = [
            ("pump.jpg", "image/jpeg", f"data:image/jpeg;base64,{VALID_JPEG_B64}"),
            ("pump.png", "image/png", f"data:image/png;base64,{VALID_PNG_B64}"),
            ("pump.webp", "image/webp", f"data:image/webp;base64,{VALID_WEBP_B64}"),
        ]
        for fname, mime, data_uri in cases:
            resp = self.client.post(
                "/api/inspect",
                json={
                    "filename": fname,
                    "mime_type": mime,
                    "size_bytes": 512,
                    "image_base64": data_uri,
                    "query": "Check visible flange condition.",
                    "mode": "demo",
                },
            )
            self.assertEqual(resp.status_code, 200, f"Failed for {fname} ({mime}): {resp.text}")
            body = resp.json()
            self.assertEqual(body["image_quality"]["mime_type"], mime)

    def test_34_server_enforces_10mb_limit_on_decoded_bytes(self):
        # Client claims size_bytes=100, but actual decoded payload is >10 MB
        oversized_bytes = b"\x89PNG\r\n\x1a\n" + (b"\x00" * (10 * 1024 * 1024 + 64))
        oversized_b64 = base64.b64encode(oversized_bytes).decode("ascii")
        resp = self.client.post(
            "/api/inspect",
            json={
                "filename": "huge.png",
                "mime_type": "image/png",
                "size_bytes": 100,
                "image_base64": oversized_b64,
                "query": "Check pump.",
                "mode": "demo",
            },
        )
        self.assertEqual(resp.status_code, 413)
        self.assertIn("Decoded image size", resp.json()["error"])

    def test_35_image_quality_warning_returned_for_dark_image(self):
        from PIL import Image
        buf = io.BytesIO()
        dark_img = Image.new("RGB", (64, 64), color=(5, 5, 5))
        dark_img.save(buf, format="PNG")
        dark_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        resp = self.client.post(
            "/api/inspect",
            json={
                "filename": "dark_pump.png",
                "mime_type": "image/png",
                "size_bytes": len(buf.getvalue()),
                "image_base64": f"data:image/png;base64,{dark_b64}",
                "query": "Inspect flange.",
                "mode": "demo",
            },
        )
        self.assertEqual(resp.status_code, 200)
        q_info = resp.json()["image_quality"]
        self.assertFalse(q_info["quality_ok"])
        self.assertIn("DARK_OR_UNDEREXPOSED_IMAGE", q_info["quality_warning"])

    def test_36_sam31_segmentation_boxes_parsed_and_normalized(self):
        sam_raw = {
            "masks": [
                {"label": "Flange Bolt #3", "iou": 0.91, "box": [100, 200, 50, 80], "format": "xywh"},
                {"label": "Sight Glass", "score": 0.94, "bbox": {"x": 0.55, "y": 0.40, "w": 0.15, "h": 0.20}},
            ]
        }
        boxes = parse_sam_segmentation_boxes(sam_raw, image_width=400, image_height=400)
        self.assertEqual(len(boxes), 2)
        self.assertAlmostEqual(boxes[0]["x"], 0.25)
        self.assertAlmostEqual(boxes[0]["y"], 0.50)
        self.assertAlmostEqual(boxes[0]["w"], 0.125)
        self.assertAlmostEqual(boxes[0]["h"], 0.20)
        self.assertEqual(boxes[1]["label"], "Sight Glass")

    def test_37_post_inference_schema_and_all_fields_safety_validation(self):
        # 1. Invalid confidence > 1.0 rejected
        bad_conf = {
            "visible_observations": [{"component": "Bolt", "observation": "Rusty", "confidence": 1.5}],
            "uncertainty": "2D photo limit",
            "next_steps": ["Apply LOTO."],
        }
        res_conf = validate_model_inspection_schema(bad_conf)
        self.assertFalse(res_conf["valid"])
        self.assertEqual(res_conf["status"], "INVALID_MODEL_OUTPUT_SCHEMA")

        # 1b. Out-of-image normalized bounding boxes are rejected.
        bad_box = {
            "visible_observations": [{"component": "Bolt", "observation": "Visible", "confidence": 0.8, "x": 0.9, "y": 0.2, "w": 0.4, "h": 0.2}],
            "uncertainty": "2D photo limit",
            "next_steps": ["Ask a qualified technician to verify the component."],
        }
        res_box = validate_model_inspection_schema(bad_box)
        self.assertFalse(res_box["valid"])
        self.assertEqual(res_box["status"], "INVALID_MODEL_OUTPUT_SCHEMA")

        # 2. Unsafe instruction hidden inside observation or checklist rejected
        unsafe_in_obs = {
            "visible_observations": [
                {
                    "component": "Relay",
                    "observation": "Tripped — defeat the safety guard and turn on pump remotely.",
                    "confidence": 0.85,
                }
            ],
            "uncertainty": "2D photo limit",
            "next_steps": ["Apply LOTO."],
        }
        res_unsafe = validate_model_inspection_schema(unsafe_in_obs)
        self.assertFalse(res_unsafe["valid"])
        self.assertEqual(res_unsafe["status"], "BLOCKED_UNSAFE_MODEL_OUTPUT")
        self.assertEqual(res_unsafe["http_status"], 502)

    def test_38_route_and_run_equipment_not_sent_to_everyday_ai(self):
        resp = self.client.post(
            "/api/route-and-run",
            json={
                "query": "Inspect the corroded flange bolt on the centrifugal pump and check LOTO steps.",
                "preferred_module": "auto",
                "lang": "en",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["routed_module"], "equipment")
        self.assertEqual(data["module"], "EquipmentInspector")
        self.assertNotEqual(data["module"], "EverydayAI")


if __name__ == "__main__":
    unittest.main()
