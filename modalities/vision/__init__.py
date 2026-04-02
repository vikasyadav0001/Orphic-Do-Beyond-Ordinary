# modalities/vision/
# Vision modality — image and video understanding.
# Responsibilities:
#   - image_reader.py   → Loads and encodes images for vision LLMs (GPT-4o, Gemini)
#   - captioner.py      → Generates text descriptions from images
#   - ocr.py            → Extracts text from images/PDFs (Optical Character Recognition)
# Feeds visual context into the agent's message state as structured text or base64.
