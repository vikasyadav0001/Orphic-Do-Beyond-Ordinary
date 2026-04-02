# modalities/text/
# Text modality — everything related to raw text processing.
# Responsibilities:
#   - chunker.py      → Splits long documents into overlapping chunks for embedding/retrieval
#   - tokenizer.py    → Counts tokens, truncates inputs to model context window limits
#   - formatter.py    → Formats agent outputs (markdown, plain text, structured JSON)
