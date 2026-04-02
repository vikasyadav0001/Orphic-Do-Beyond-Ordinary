# modalities/
# Multimodal input/output handling — how agents perceive and produce different data types.
# Enables the system to move beyond text to a truly multimodal AI agent.
# Responsibilities:
#   - text/    → Text preprocessing, chunking, tokenization
#   - vision/  → Image understanding, captioning, OCR (via vision LLMs or APIs)
#   - audio/   → Speech-to-text (transcription), text-to-speech (TTS)
# Each sub-module provides a unified interface so agents don't care about the modality type.
