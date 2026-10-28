# AegisOmni Demo Script (target: 1:45–2:00)

**Recording checklist**
- Use the real application and an image you are allowed to share.
- Confirm LIVE mode is selected and the server has a valid `MODEL_API_KEY` before recording a live-call claim.
- Hide credentials, browser tabs with private data, personal notifications, and logs containing sensitive information.
- Record in landscape if possible; keep the browser zoom high enough to read the report.
- If LIVE inference fails, record an honest DEMO-mode walkthrough and state the live limitation. Do not edit a mocked response to look live.

## Script

**0:00–0:15 — The problem**
“Reading visible equipment conditions from a single photo is difficult, and a model answer should not automatically be treated as a safe instruction. AegisOmni is a prototype for structured visual review with explicit uncertainty and safety checks.”

**0:15–0:35 — The workflow**
Show the app and image upload. “The user uploads an image and asks a focused question. The backend validates the image before sending it to the configured Meta vision model.”

**0:35–1:05 — The actual report**
Perform a real LIVE request if verified. Show observations, confidence, uncertainty and next steps. “AegisOmni validates the report schema and checks action-bearing text before accepting the output. If validation fails, it rejects the report instead of silently substituting sample results.”

**1:05–1:25 — Safety and failure handling**
Use the safety testbench with a blocked example, without touching or controlling equipment. “This is a read-only advisory prototype; no physical actuation is possible from this app. The gate blocks certain unsafe requests and generated recommendations.”

**1:25–1:45 — Segmentation / limitation**
If real SAM boxes were returned, briefly show them and say the box positions are normalized from the upstream response. Otherwise say segmentation is optional and not demonstrated. “A photo cannot confirm hidden wear or electrical isolation. A qualified technician must verify the actual equipment and physical LOTO.”

**1:45–2:00 — Wrap-up**
“AegisOmni's aim is to make visual inspection results more structured and reviewable—not to replace engineering judgement. The repository includes setup instructions and automated tests for safety checks, malformed responses and image validation.”

## Final honest-status checklist

- [ ] Real Meta API call succeeded during the recorded demo.
- [ ] SAM was shown only if its call returned usable boxes.
- [ ] Deployed HTTPS URL was tested from a fresh browser session.
- [ ] Every performance or test claim in the narration matches actual evidence.
