# AegisOmni — Multimodal Visual Inspection & Safety Assistant

**Hackathon project prototype** for the Meta Global AI Developer Hackathon. AegisOmni combines an image-inspection workflow with deterministic safety checks, uncertainty handling, and clearly labelled demo modules.

> **Safety disclaimer:** This is a read-only advisory prototype, not a certified industrial-safety system. It cannot verify internal equipment condition or live electrical state from a photograph, does not control PLCs or physical actuators, and never replaces a qualified technician or required physical Lockout/Tagout (LOTO) verification.

## What works in this repository

- A responsive, single-page HTML interface intended to work on desktop and Android browsers.
- A FastAPI backend with health/status, safety-check, image-inspection and secondary-module routing endpoints.
- Server-side image payload validation (JPEG, PNG and WebP allowlist, declared MIME reconciliation, actual decoded-size limit, magic-byte checks and image-quality warnings when Pillow is available).
- A pre-inference deterministic safety gate and strict post-inference report schema and recommendation checks.
- A live vision path configured for the Meta Model API's `muse-spark-1.3` Chat Completions endpoint.
- Optional SAM 3.1 segmentation integration using the documented Responses API message envelope and streaming output. The parser handles SAM special-token box records and supported JSON/mock shapes, normalizing boxes to `[x, y, width, height]` in the `[0, 1]` range. Confidence is shown only when a confidence value is actually supplied. If a response has no usable coordinates, the interface must not pretend that boxes exist.
- Clearly separated DEMO and LIVE behavior. LIVE inspector mode requires an explicitly uploaded photo; the built-in canvas illustration is not accepted as live evidence. The secondary LearnAI, FarmAI and EverydayAI demonstrations use curated sample responses; an uploaded FarmAI sample photo is not sent to live model inference.
- Regression tests using mocked upstream responses.

**Test status:** Run the commands below to reproduce the automated test results in your environment. Mocked tests do not prove that a real Meta API credential works, that current account access includes the requested models, or that a public deployment is healthy.

## Project layout

```text
AegisOmni_SubmissionReady/
├── app.py
├── aegis_omni_meta_hackathon.html
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── SUBMISSION_DESCRIPTION.md
├── DEMO_SCRIPT.md
├── src/
│   ├── __init__.py
│   ├── safety_gate.py
│   ├── inspector_client.py
│   └── multi_domain_engine.py
└── tests/
    ├── test_safety_assistant.py
    └── test_submission_hardening.py
```

## Quick start

Use Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.venv\\Scripts\\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies and test:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`. On Android, a local URL works only in the same device/network context. For a judge-accessible demo, deploy to a reachable HTTPS host and test its public URL. Do not use `localhost` in the submission as a public demo link.

## Configure LIVE mode

Set `MODEL_API_KEY` in the server's environment or deployment secret manager. Do not place it in HTML, client JavaScript, GitHub, or `.env.example`.

Linux/macOS shell example (for local testing only):

```bash
export MODEL_API_KEY="YOUR_REAL_KEY_HERE"
export META_API_BASE="https://api.meta.ai/v1"
export META_VISION_MODEL="muse-spark-1.3"
export META_SAM_MODEL="sam-3.1"
uvicorn app:app --host 0.0.0.0 --port 8000
```

The app reads `MODEL_API_KEY` server-side. `GET /api/status` reports only whether a key is configured and the model names; it must not return the key itself. Live inspection uses a real image payload and returns an error if a key is not configured or upstream inference fails. SAM segmentation is optional and should only be enabled when the account has access to the configured SAM model.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `GET` | `/health` | Process health response |
| `GET` | `/api/status` | Reports model configuration without exposing secrets |
| `POST` | `/api/safety-check` | Runs the deterministic request safety gate |
| `POST` | `/api/inspect` | Validates an image request, then runs DEMO or LIVE inspection |
| `POST` | `/api/route-and-run` | Routes to the curated LearnAI, FarmAI, EverydayAI or equipment demo module |

`POST /api/inspect` accepts JSON fields such as `filename`, `mime_type`, `size_bytes`, `image_base64`, `query`, `component_prompt`, `enable_segmentation`, and `mode` (`demo` or `live`). The server measures decoded payload size; client-declared metadata is not treated as the sole source of truth.

## Test coverage

The suite includes deterministic safety rules, image validation, empty/malformed upstream responses, simulated upstream timeouts, missing-key behavior, schema/recommendation rejection, optional SAM parsing, and FastAPI endpoint tests. All external inference behavior in unit tests is mocked. You still need one real, successful LIVE inspection with your own allowed API credentials before claiming live end-to-end verification in the submitted demo.

## Recommended demo flow

1. Open the deployed app and state that AegisOmni is an advisory prototype.
2. Upload a real, non-sensitive equipment image and ask a focused visual-inspection question.
3. Show the returned observations, confidence and uncertainty information.
4. Show a safety-gate rejection test without attempting any physical action.
5. If SAM is enabled and works, show actual returned boxes; otherwise explain that segmentation is optional and do not fake it.
6. End with limitations and the fact that technicians must independently verify physical LOTO.

See [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) and [`SUBMISSION_DESCRIPTION.md`](SUBMISSION_DESCRIPTION.md).

## Limitations and responsible use

- A single 2D image cannot prove internal wear, torque, pressure calibration, electrical isolation, or whether a machine is energized.
- Model-generated observations may be wrong or incomplete. A deterministic text-based safety gate reduces some risks but cannot guarantee safety.
- The project is not certified, and its output must not be used as a substitute for standards, site procedures, or qualified personnel.
- The auxiliary LearnAI/FarmAI/EverydayAI UI is primarily curated demonstration material; the equipment inspector is the primary live model workflow.
- Do not upload sensitive, regulated, or proprietary imagery without authorization.

## Meta Model API references

- [Meta Model API overview](https://dev.meta.ai/docs/overview)
- [Chat Completions quickstart](https://dev.meta.ai/docs/cookbook/quickstart-chat-completions)
- [Image understanding](https://dev.meta.ai/docs/image-understanding)
- [SAM overview](https://dev.meta.ai/docs/sam/overview)

Check current documentation and your account's model access before recording the final demo.

## Submission status gate

The code package and text assets are prepared, and the automated suite is passing. Before submitting a live-demo claim, you must still: (1) configure a legitimate `MODEL_API_KEY` on the server, (2) run one real LIVE photo inspection, (3) deploy and verify an HTTPS URL from a fresh session, and (4) record the demo from that verified build. Those external account and hosting actions cannot be verified from this source-only package.
