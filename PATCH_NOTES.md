# Patch Notes — Submission Package

This package preserves the submitted AegisOmni layout and project modules while applying targeted reliability changes.

- Organized the repository into a runnable `src/` package and `tests/` directory and added dependency/configuration files.
- Validated the inspection `mode` against `demo` and `live`.
- Retained server-side image validation and strict post-inference report/safety validation.
- Redacted detailed upstream API exceptions from HTTP responses while retaining server-side exception logging.
- Corrected SAM 3.1 request shape to use a Responses API `message` input and streaming output.
- Added parsing for Meta SAM's documented `<|box;x1=...;y1=...;x2=...;y2=...;w=...;h=...|>` output tokens, including normalized box coordinates.
- Treats missing SAM boxes as a no-box result, not success. Does not invent a confidence score where SAM has not provided one.
- Escaped user/model-controlled values before interpolation into HTML templates and added clearer labels for demo-only material.
- Required a genuinely uploaded equipment photo for LIVE inspections; the built-in illustrated canvas cannot be sent as if it were a real image.
- Added regression tests for SAM streaming/parsing, malformed/no-box behavior, invalid mode, and safe rendering markers.

Automated tests are mocked for external model calls. A real live API call and deployed public URL still require the developer's API credentials and hosting account to verify.
