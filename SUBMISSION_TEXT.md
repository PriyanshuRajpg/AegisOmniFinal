# AegisOmni — Submission Text

Use the wording below as a starting point for the portal. Adjust it to the field's word limit, and only describe actions you have personally reproduced in the deployed demo.

## Project title
AegisOmni — AI-Assisted Visual Equipment Inspection

## One-line summary
AegisOmni turns an uploaded equipment photo into a structured, uncertainty-aware inspection report and runs deterministic safety checks before accepting AI-generated next steps.

## Project description
Visual inspection from a single photo is uncertain, and a plausible AI answer should not automatically be treated as a safe instruction. AegisOmni is an advisory prototype that lets a user upload an equipment image and ask a focused inspection question. Its backend validates the image, calls the configured Meta vision model, validates the returned report schema, and checks action-bearing text against a deterministic safety gate before presenting observations, confidence and uncertainty. Optional SAM 3.1 output can provide normalized object boxes when the upstream response contains usable coordinates. The UI separates DEMO from LIVE behavior and avoids substituting sample findings after a live error. AegisOmni cannot control equipment and does not replace qualified inspection or physical Lockout/Tagout verification.

## Technical implementation
- Mobile-friendly HTML/CSS/JavaScript single-page interface.
- FastAPI service for upload validation, inspection, health/status, safety-check and module-routing endpoints.
- Meta Model API client configured for `muse-spark-1.3` image understanding and optional `sam-3.1` segmentation.
- Server-side MIME, magic-header and decoded-size checks, with image-quality warnings when possible.
- Strict structured-report validation and deterministic checks over visible observations, uncertainty, and next steps.
- SAM 3.1 streaming-output parser for documented box tokens, with normalized coordinates and no invented confidence values.
- Automated unit/integration tests covering the safety gate, malformed responses, upload validation and mocked API flows.

## Meta AI tooling
The primary live inspection path is configured to call Muse Spark through the Meta Model API. Optional SAM 3.1 segmentation is configured through the Responses API. Do not claim a live run or successful segmentation in the demo unless it actually occurred with your API account.

## Verification and deployment note
The packaged project passes its automated tests in the build environment, including mocked API calls and SAM stream parsing. No `MODEL_API_KEY` was available in that environment, so a real external inference call was not performed there. Verify one real LIVE photo inspection and the deployed HTTPS URL before claiming end-to-end live success. This note describes the available verification evidence; it is not a claim of live API success.
