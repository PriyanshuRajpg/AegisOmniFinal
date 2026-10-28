"""Deterministic Safety & Input Validation Gate for AegisOmni Equipment Safety Assistant.
Enforces:
  1. Server-side image validation (MIME allowlist for JPEG/PNG/WebP, 10 MB limit enforced on actual decoded bytes, magic-byte header verification, and dark/low-contrast detection via Pillow).
  2. Pre-execution user input safety gate (blocking interlock bypass, defeating guards, physical actuation, live electrical work, and unsafe pesticide dosages).
  3. Strict post-inference schema and recommendation safety validation (requiring visible_observations, confidence in [0, 1], uncertainty, and next_steps, and checking all action-bearing fields against deterministic safety rules).
"""
from dataclasses import dataclass, asdict
import base64
import io
import re
from typing import Dict, Any, List, Optional

try:
    from PIL import Image, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


PROTOTYPE_DISCLAIMER = (
    "PROTOTYPE DECISION-SUPPORT TOOL: Not a certified industrial safety system. "
    "Cannot issue physical control commands. Requires human Lockout/Tagout (LOTO) verification."
)

ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class SafetyEvaluation:
    allowed: bool
    action: str
    triggered_rule: str
    reason: str
    required_human_action: str
    http_status: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


UNSAFE_PATTERNS = [
    (
        re.compile(
            r"\b(bypass|override|jumper|disable|short-circuit|defeat|tape\s+down|bridge|remove)\b.*"
            r"\b(interlock|relay|relief|breaker|fuse|loto|lockout|guard|limit\s+switch|ground|earthing|sensor|contactor|protective)\b",
            re.I,
        ),
        "INTERLOCK_OR_RELAY_BYPASS_PROHIBITED",
        "Bypassing or defeating safety interlocks, guards, limit switches, or protective relays creates severe physical hazard.",
    ),
    (
        re.compile(
            r"\b(remotely\s+open|force\s+open|actuate|energize|start\s+motor|close\s+breaker|open\s+valve|turn\s+on\s+pump|switch\s+on\s+breaker|trigger\s+actuator)\b",
            re.I,
        ),
        "NO_PHYSICAL_ACTUATION_COMMANDS",
        "AegisOmni is a read-only advisory assistant and is prohibited from issuing physical actuation or equipment-control commands.",
    ),
    (
        re.compile(
            r"\b(while\s+energized|while\s+live|480v\s+live|without\s+loto|skip\s+lockout|live\s+terminals|hot\s+work\s+without\s+isolation)\b",
            re.I,
        ),
        "LIVE_ELECTRICAL_WORK_PROHIBITED",
        "Instructions for live electrical work without Lockout/Tagout (LOTO) violate electrical safety standards.",
    ),
    (
        re.compile(
            r"\b(exact\s+chemical\s+dosage|mix\s+monocrotophos|paraquat\s+dosage|lethal\s+pesticide\s+dose|unlabeled\s+pesticide\s+mix)\b",
            re.I,
        ),
        "UNSAFE_PESTICIDE_DOSAGE_PROHIBITED",
        "FarmAI provides cautious symptom guidance and does not prescribe hazardous chemical pesticide dosages. Consult a certified agricultural extension officer (KVK).",
    ),
]


def _has_valid_magic_bytes(raw_bytes: bytes, mime_type: str) -> bool:
    m = (mime_type or "").lower()
    if m == "image/png":
        return raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if m == "image/jpeg":
        return raw_bytes.startswith(b"\xff\xd8\xff")
    if m == "image/webp":
        return len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP"
    return False


def assess_image_quality(raw_bytes: bytes) -> Dict[str, Any]:
    """Inspects decoded image dimensions, brightness, and contrast when Pillow is available."""
    if not PIL_AVAILABLE:
        return {"quality_ok": True, "warning": None, "width": None, "height": None}
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size
            if width < 16 or height < 16:
                return {
                    "quality_ok": False,
                    "warning": f"LOW_RESOLUTION_IMAGE ({width}x{height}px): Image is too small for reliable component inspection.",
                    "width": width,
                    "height": height,
                }
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            mean_luma = stat.mean[0]
            std_luma = stat.stddev[0]
            if mean_luma < 22.0:
                return {
                    "quality_ok": False,
                    "warning": f"DARK_OR_UNDEREXPOSED_IMAGE (luma={mean_luma:.1f}): Lighting is too dark; capture a well-lit photo.",
                    "width": width,
                    "height": height,
                }
            if std_luma < 8.0:
                return {
                    "quality_ok": False,
                    "warning": f"LOW_CONTRAST_OR_OBSTRUCTED_IMAGE (stddev={std_luma:.1f}): Image appears blurry, washed out, or obstructed.",
                    "width": width,
                    "height": height,
                }
            return {"quality_ok": True, "warning": None, "width": width, "height": height}
    except Exception:
        return {"quality_ok": True, "warning": None, "width": None, "height": None}


def validate_image_upload(
    filename: str,
    mime_type: str,
    size_bytes: int,
    image_base64: Optional[str] = None,
) -> Dict[str, Any]:
    if not filename:
        return {"valid": False, "http_status": 400, "error": "EMPTY_IMAGE_FILE: Filename is missing."}

    # Extract MIME from data URI header if present and reconcile with declared mime_type
    effective_mime = (mime_type or "").strip().lower()
    if image_base64 and image_base64.strip().startswith("data:"):
        header_part = image_base64.strip().split(",", 1)[0]
        if ";" in header_part and ":" in header_part:
            uri_mime = header_part.split(":", 1)[1].split(";", 1)[0].strip().lower()
            if not effective_mime:
                effective_mime = uri_mime
            elif uri_mime and uri_mime != effective_mime:
                return {
                    "valid": False,
                    "http_status": 400,
                    "error": f"MISMATCHED_DATA_URI_MIME: Declared mime_type '{effective_mime}' does not match data URI '{uri_mime}'.",
                }

    if effective_mime not in ALLOWED_IMAGE_MIMES:
        return {
            "valid": False,
            "http_status": 415,
            "error": f"UNSUPPORTED_MEDIA_TYPE: '{mime_type}'. Allowed formats: JPEG, PNG, WebP.",
        }

    if image_base64 is not None:
        raw_b64 = image_base64.split(",", 1)[-1].strip()
        if not raw_b64:
            return {"valid": False, "http_status": 400, "error": "EMPTY_IMAGE_DATA: Base64 payload is empty."}
        try:
            decoded = base64.b64decode(raw_b64, validate=True)
        except Exception:
            return {"valid": False, "http_status": 400, "error": "INVALID_BASE64_IMAGE: Could not decode image bytes."}

        actual_bytes = len(decoded)
        if actual_bytes <= 0:
            return {"valid": False, "http_status": 400, "error": "EMPTY_IMAGE_FILE: Decoded image has 0 bytes."}
        if actual_bytes > MAX_IMAGE_BYTES:
            return {
                "valid": False,
                "http_status": 413,
                "error": f"IMAGE_TOO_LARGE: Decoded image size ({actual_bytes} bytes) exceeds 10 MB limit ({MAX_IMAGE_BYTES} bytes).",
            }
        if size_bytes is not None and size_bytes > MAX_IMAGE_BYTES:
            return {
                "valid": False,
                "http_status": 413,
                "error": f"IMAGE_TOO_LARGE: Declared size ({size_bytes} bytes) exceeds 10 MB limit.",
            }
        if actual_bytes < 8 or not _has_valid_magic_bytes(decoded, effective_mime):
            return {
                "valid": False,
                "http_status": 400,
                "error": f"CORRUPTED_OR_MISMATCHED_IMAGE_HEADER: File bytes do not match declared MIME type {effective_mime}.",
            }
        quality = assess_image_quality(decoded)
        return {
            "valid": True,
            "http_status": 200,
            "error": None,
            "mime_type": effective_mime,
            "decoded_size_bytes": actual_bytes,
            "quality_ok": quality.get("quality_ok", True),
            "quality_warning": quality.get("warning"),
            "width": quality.get("width"),
            "height": quality.get("height"),
        }

    # Fallback when only metadata is validated without base64 payload
    if size_bytes is None or size_bytes <= 0:
        return {"valid": False, "http_status": 400, "error": "EMPTY_IMAGE_FILE: Image file is empty or missing."}
    if size_bytes > MAX_IMAGE_BYTES:
        return {
            "valid": False,
            "http_status": 413,
            "error": f"IMAGE_TOO_LARGE: {size_bytes} bytes exceeds 10 MB limit.",
        }
    return {
        "valid": True,
        "http_status": 200,
        "error": None,
        "mime_type": effective_mime,
        "decoded_size_bytes": size_bytes,
        "quality_ok": True,
        "quality_warning": None,
    }


def evaluate_safety_gate(query_text: Optional[str]) -> SafetyEvaluation:
    cleaned = (query_text or "").strip()
    if not cleaned:
        return SafetyEvaluation(
            allowed=False,
            action="REJECT_EMPTY_QUERY",
            triggered_rule="EMPTY_OR_WHITESPACE_QUERY",
            reason="Question input cannot be empty.",
            required_human_action="Provide a specific equipment inspection question.",
            http_status=400,
        )

    for pattern, rule_name, reason in UNSAFE_PATTERNS:
        if pattern.search(cleaned):
            return SafetyEvaluation(
                allowed=False,
                action="BLOCK_EXECUTION",
                triggered_rule=rule_name,
                reason=reason,
                required_human_action="Physical Lockout/Tagout (LOTO) verification and licensed supervisor sign-off required.",
                http_status=403,
            )
    return SafetyEvaluation(
        allowed=True,
        action="ALLOW_ADVISORY_INSPECTION",
        triggered_rule="NONE",
        reason="Query requests read-only visual inspection or checklist generation with no physical actuation or interlock bypass.",
        required_human_action="Technician must still verify physical LOTO before performing maintenance.",
        http_status=200,
    )


def validate_generated_recommendations(recommendations: List[str]) -> SafetyEvaluation:
    """Runs deterministic safety rules on model-generated recommendations so the model can never override safety policy."""
    for step in recommendations or []:
        if not isinstance(step, str) or not step.strip():
            continue
        check = evaluate_safety_gate(step)
        if not check.allowed and check.http_status == 403:
            return SafetyEvaluation(
                allowed=False,
                action="BLOCK_UNSAFE_MODEL_RECOMMENDATION",
                triggered_rule=f"MODEL_OUTPUT_{check.triggered_rule}",
                reason=f"Blocked unsafe model-generated recommendation: '{step}'. {check.reason}",
                required_human_action="Reject model advice; consult a licensed supervisor and follow physical LOTO procedures.",
                http_status=502,
            )
    return SafetyEvaluation(
        allowed=True,
        action="RECOMMENDATIONS_VERIFIED_SAFE",
        triggered_rule="NONE",
        reason="All generated recommendations passed deterministic safety validation.",
        required_human_action="Technician must verify LOTO before physical inspection.",
        http_status=200,
    )


def validate_model_inspection_schema(parsed_report: Any) -> Dict[str, Any]:
    """Validates the full structure, types, confidence bounds, and safety of a model-generated inspection report."""
    if not isinstance(parsed_report, dict):
        return {
            "valid": False,
            "status": "MALFORMED_UPSTREAM_JSON",
            "http_status": 502,
            "error": "Model output is not a JSON object.",
        }

    observations = parsed_report.get("visible_observations")
    if not isinstance(observations, list) or len(observations) == 0:
        return {
            "valid": False,
            "status": "INVALID_MODEL_OUTPUT_SCHEMA",
            "http_status": 502,
            "error": "Model output must include a non-empty 'visible_observations' list.",
        }

    uncertainty = parsed_report.get("uncertainty")
    if not isinstance(uncertainty, str) or not uncertainty.strip():
        return {
            "valid": False,
            "status": "INVALID_MODEL_OUTPUT_SCHEMA",
            "http_status": 502,
            "error": "Model output must include a non-empty 'uncertainty' string.",
        }

    next_steps = parsed_report.get("next_steps")
    if not isinstance(next_steps, list) or len(next_steps) == 0 or not all(isinstance(s, str) and s.strip() for s in next_steps):
        return {
            "valid": False,
            "status": "INVALID_MODEL_OUTPUT_SCHEMA",
            "http_status": 502,
            "error": "Model output must include a non-empty 'next_steps' list of strings.",
        }

    actionable_texts: List[str] = [uncertainty] + list(next_steps)

    # Also collect any optional action-bearing fields if present
    for extra_key in ("checklist", "recommendations", "actions", "summary"):
        val = parsed_report.get(extra_key)
        if isinstance(val, list):
            for item in val:
                if not isinstance(item, str):
                    return {
                        "valid": False,
                        "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                        "http_status": 502,
                        "error": f"Field '{extra_key}' must contain only strings.",
                    }
                actionable_texts.append(item)
        elif isinstance(val, str):
            actionable_texts.append(val)

    normalized_obs = []
    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            return {
                "valid": False,
                "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                "http_status": 502,
                "error": f"visible_observations[{idx}] must be an object.",
            }
        comp = obs.get("component") or obs.get("name")
        finding = obs.get("observation") or obs.get("finding")
        conf = obs.get("confidence")

        if not isinstance(comp, str) or not comp.strip():
            return {
                "valid": False,
                "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                "http_status": 502,
                "error": f"visible_observations[{idx}] is missing a valid 'component' string.",
            }
        if not isinstance(finding, str) or not finding.strip():
            return {
                "valid": False,
                "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                "http_status": 502,
                "error": f"visible_observations[{idx}] is missing a valid 'observation' string.",
            }
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            return {
                "valid": False,
                "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                "http_status": 502,
                "error": f"visible_observations[{idx}].confidence must be a float in [0.0, 1.0].",
            }
        conf_f = float(conf)
        if not (0.0 <= conf_f <= 1.0):
            return {
                "valid": False,
                "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                "http_status": 502,
                "error": f"visible_observations[{idx}].confidence ({conf_f}) is outside [0.0, 1.0].",
            }

        entry: Dict[str, Any] = {
            "component": comp.strip(),
            "observation": finding.strip(),
            "confidence": round(conf_f, 4),
        }
        # Validate optional bounding box coordinates if included in observation
        if all(k in obs for k in ("x", "y", "w", "h")):
            try:
                bx, by, bw, bh = float(obs["x"]), float(obs["y"]), float(obs["w"]), float(obs["h"])
            except (TypeError, ValueError):
                return {
                    "valid": False,
                    "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                    "http_status": 502,
                    "error": f"visible_observations[{idx}] has non-numeric box coordinates.",
                }
            if (
                not all(0.0 <= c <= 1.0 for c in (bx, by, bw, bh))
                or bw <= 0.0
                or bh <= 0.0
                or bx + bw > 1.000001
                or by + bh > 1.000001
            ):
                return {
                    "valid": False,
                    "status": "INVALID_MODEL_OUTPUT_SCHEMA",
                    "http_status": 502,
                    "error": (
                        f"visible_observations[{idx}] box must have normalized coordinates and positive size "
                        "fully inside the image bounds [0.0, 1.0]."
                    ),
                }
            entry.update({"x": bx, "y": by, "w": bw, "h": bh})

        normalized_obs.append(entry)
        actionable_texts.append(comp)
        actionable_texts.append(finding)

    # Run deterministic safety check across EVERY action-bearing and text field in the model output
    rec_gate = validate_generated_recommendations(actionable_texts)
    if not rec_gate.allowed:
        return {
            "valid": False,
            "status": "BLOCKED_UNSAFE_MODEL_OUTPUT",
            "http_status": rec_gate.http_status,
            "error": rec_gate.reason,
            "safety_evaluation": rec_gate.to_dict(),
        }

    return {
        "valid": True,
        "status": "VALID",
        "http_status": 200,
        "normalized_report": {
            "visible_observations": normalized_obs,
            "uncertainty": uncertainty.strip(),
            "next_steps": [s.strip() for s in next_steps],
        },
        "safety_evaluation": rec_gate.to_dict(),
    }


def build_inspection_report(
    sample_title: str,
    query_text: str,
    regions: Optional[List[Dict[str, Any]]],
    checklist: Optional[List[str]],
    min_confidence_threshold: float = 0.70,
    mode: str = "demo",
) -> Dict[str, Any]:
    gate = evaluate_safety_gate(query_text)
    if not gate.allowed:
        return {
            "status": "REJECTED_BY_SAFETY_GATE" if gate.http_status == 403 else "VALIDATION_ERROR",
            "mode": mode,
            "http_status": gate.http_status,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
            "findings": [],
            "checklist": [],
        }

    if regions is None or len(regions) == 0:
        return {
            "status": "EMPTY_MODEL_RESPONSE_ERROR",
            "mode": mode,
            "http_status": 502,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
            "abstained_due_to_low_confidence": True,
            "requires_human_loto_signoff": True,
            "findings": [],
            "checklist": ["Model returned an empty response. Perform manual LOTO visual inspection."],
        }

    all_texts: List[str] = list(checklist or [])
    for r in regions:
        if isinstance(r, dict):
            for k in ("name", "component", "finding", "observation", "status"):
                if isinstance(r.get(k), str):
                    all_texts.append(r[k])

    rec_check = validate_generated_recommendations(all_texts)
    if not rec_check.allowed:
        return {
            "status": "BLOCKED_UNSAFE_MODEL_OUTPUT",
            "mode": mode,
            "http_status": rec_check.http_status,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": rec_check.to_dict(),
            "findings": [],
            "checklist": [],
        }

    low_conf_regions = []
    for r in regions:
        conf_val = r.get("confidence", 0.0)
        if isinstance(conf_val, bool) or not isinstance(conf_val, (int, float)):
            raise ValueError(f"Invalid confidence type: {conf_val}")
        conf_f = float(conf_val)
        if not (0.0 <= conf_f <= 1.0):
            raise ValueError(f"Confidence {conf_f} out of [0.0, 1.0] range")
        if conf_f < min_confidence_threshold:
            low_conf_regions.append(r)

        for coord in ("x", "y", "w", "h"):
            val = float(r.get(coord, 0.0))
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Bounding box coordinate {coord}={val} out of normalized [0, 1] range")

    abstained = len(low_conf_regions) > 0

    return {
        "status": "INSUFFICIENT_VISUAL_EVIDENCE" if abstained else "ADVISORY_REPORT_READY",
        "mode": mode,
        "http_status": 200,
        "disclaimer": PROTOTYPE_DISCLAIMER,
        "safety_evaluation": gate.to_dict(),
        "recommendation_safety": rec_check.to_dict(),
        "abstained_due_to_low_confidence": abstained,
        "requires_human_loto_signoff": True,
        "sample_title": sample_title,
        "findings": regions,
        "checklist": checklist or [],
    }
