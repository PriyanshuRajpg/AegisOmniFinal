# AegisOmni — Final Submission Readiness

## Verified in the packaged source

- [x] Consistent `src/` package layout and dependency manifest are present.
- [x] Python source compiles successfully.
- [x] Automated test suite passes: 45 tests.
- [x] HTML inline JavaScript passes Node syntax checks.
- [x] Local FastAPI smoke checks pass for `/`, `/health`, and `/api/status`.
- [x] Status endpoint does not return a model API key.
- [x] Mocked tests cover response schema checks, unsafe recommendations, image validation, SAM streamed token parsing, no-box outcomes, and safe rendering of untrusted text.

## Must be verified with the submitter's accounts before claiming a complete live demo

- [ ] Set a valid Meta Model API credential in the server/deployment secret manager (never in the HTML or repository).
- [ ] Confirm `GET /api/status` reports `live_api_configured: true` without revealing the secret.
- [ ] Upload a real, permitted equipment photo and confirm `/api/inspect` returns `LIVE_API_SUCCESS` and a validated report.
- [ ] Enable SAM only if the account has access; confirm real boxes are returned, or show segmentation as unavailable.
- [ ] Deploy the backend and HTML UI to an HTTPS endpoint reachable by judges; verify from a fresh browser session and on Android.
- [ ] Record a short demo from that verified deployment. Use `DEMO_SCRIPT.md`; never present mocked responses as live inference.
- [ ] Complete the organiser's submission form and confirm it accepts the final submission.
- [ ] Check whether judges need repository access. The working package does not publish a repository or change its visibility.

## Important limitation

No `MODEL_API_KEY` was present in the build environment, and no hosting or hackathon account was connected. Therefore the real external model call, public deployment, recorded video, and portal submission have **not** been verified from here. The code is packaged and unit-tested, but those external steps remain necessary before advertising a live, deployed end-to-end system.
