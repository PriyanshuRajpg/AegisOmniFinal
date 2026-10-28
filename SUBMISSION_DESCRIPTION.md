# AegisOmni — AI-Assisted Visual Equipment Inspection

AegisOmni is a multimodal prototype designed to help users review visible equipment conditions from a photo while keeping the result bounded by explicit safety checks. A user uploads an image and asks a focused inspection question. A Meta vision model generates structured observations, confidence values, uncertainty notes, and suggested next steps. Before the report is accepted, a deterministic validation layer checks the required output schema and scans action-bearing text for unsafe recommendations. Invalid or unsafe results are rejected rather than presented as a successful inspection.

The project combines a mobile-friendly HTML interface, a FastAPI backend, Meta Model API vision inference (`muse-spark-1.3`), optional SAM 3.1 segmentation, image-upload validation, and deterministic safety checks. Demo mode is visibly distinct from live inference. Secondary LearnAI, FarmAI, and EverydayAI examples illustrate possible extensions but are curated demonstrations rather than additional verified live agents.

AegisOmni is advisory only: a photograph cannot verify internal equipment condition, electrical isolation, or safe operating state. The prototype cannot control equipment and does not replace qualified inspection or physical Lockout/Tagout verification.

**Before final submission, replace this status line with verified facts:** [State whether a real LIVE API inspection and the public deployment were successfully tested; include only features actually demonstrated in the recording.]
