"""FastAPI Backend Server for AegisOmni — Multimodal Equipment Safety Assistant.
Keeps MODEL_API_KEY strictly on the server and exposes:
  - GET  /health
  - GET  /api/status
  - POST /api/safety-check
  - POST /api/inspect
  - POST /api/route-and-run
"""
import os
import sys
from typing import Literal, Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Support both package (`src.safety_gate`) and flat directory imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from src.safety_gate import evaluate_safety_gate, validate_image_upload
    from src.inspector_client import (
        inspect_equipment_with_meta_api,
        get_server_api_status,
        route_task,
        run_equipment_route_summary,
        run_learn_ai,
        grade_learn_quiz,
        run_farm_ai,
        run_everyday_ai,
    )
except ImportError:
    from safety_gate import evaluate_safety_gate, validate_image_upload
    from inspector_client import (
        inspect_equipment_with_meta_api,
        get_server_api_status,
        route_task,
        run_equipment_route_summary,
        run_learn_ai,
        grade_learn_quiz,
        run_farm_ai,
        run_everyday_ai,
    )

app = FastAPI(title="AegisOmni Multimodal Equipment Safety Assistant")


class SafetyRequest(BaseModel):
    query: str


class InspectRequest(BaseModel):
    filename: str = "sample_pump.png"
    mime_type: Optional[str] = None
    image_mime_type: Optional[str] = None
    size_bytes: int = 10240
    image_base64: Optional[str] = None
    query: str
    component_prompt: Optional[str] = None
    enable_segmentation: bool = False
    mode: Literal["demo", "live"] = "live"
    sample_id: Optional[str] = None


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(_HERE, "aegis_omni_meta_hackathon.html"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "AegisOmni Safety Assistant"}


@app.get("/api/status")
def api_status():
    return get_server_api_status()


@app.post("/api/safety-check")
def safety_check(req: SafetyRequest):
    res = evaluate_safety_gate(req.query)
    return JSONResponse(status_code=res.http_status, content=res.to_dict())


@app.post("/api/inspect")
def inspect_endpoint(req: InspectRequest):
    effective_mime = (req.mime_type or req.image_mime_type or "image/png").strip().lower()
    img_check = validate_image_upload(req.filename, effective_mime, req.size_bytes, req.image_base64)
    if not img_check["valid"]:
        return JSONResponse(status_code=img_check["http_status"], content=img_check)

    validated_mime = img_check.get("mime_type", effective_mime)
    result = inspect_equipment_with_meta_api(
        user_question=req.query,
        image_base64=req.image_base64,
        mime_type=validated_mime,
        component_prompt=req.component_prompt,
        enable_segmentation=req.enable_segmentation,
        mode=req.mode,
        image_width=img_check.get("width"),
        image_height=img_check.get("height"),
    )
    result["image_quality"] = {
        "quality_ok": img_check.get("quality_ok", True),
        "quality_warning": img_check.get("quality_warning"),
        "decoded_size_bytes": img_check.get("decoded_size_bytes", req.size_bytes),
        "mime_type": validated_mime,
        "filename": req.filename,
    }
    if img_check.get("quality_warning") and isinstance(result.get("inspection_report"), dict):
        prev_unc = result["inspection_report"].get("uncertainty", "")
        warn_txt = f"Image Quality Warning: {img_check['quality_warning']}"
        if warn_txt not in prev_unc:
            result["inspection_report"]["uncertainty"] = f"{warn_txt} | {prev_unc}".strip(" |")
    return JSONResponse(status_code=result.get("http_status", 200), content=result)


class MultiDomainRequest(BaseModel):
    query: str
    preferred_module: str = "auto"
    level: str = "beginner"
    lang: str = "hi"
    crop_sample: str = "tomato_blight"
    mode: str = "demo"


@app.post("/api/route-and-run")
def route_and_run_endpoint(req: MultiDomainRequest):
    chosen = route_task(req.query, req.preferred_module)
    if chosen == "learn":
        res = run_learn_ai(req.query, level=req.level, lang=req.lang, mode=req.mode)
    elif chosen == "farm":
        res = run_farm_ai(req.query, crop_sample=req.crop_sample, lang=req.lang, mode=req.mode)
    elif chosen == "everyday":
        res = run_everyday_ai(req.query, lang=req.lang, mode=req.mode)
    else:
        res = run_equipment_route_summary(req.query, lang=req.lang, mode=req.mode)
    res["routed_module"] = chosen
    return JSONResponse(status_code=res.get("http_status", 200), content=res)
