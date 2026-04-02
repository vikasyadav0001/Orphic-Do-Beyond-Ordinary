# api/
# FastAPI application layer.
# Contains the HTTP and WebSocket endpoints, route handlers, and middleware.
# Responsibilities:
#   - /chat      → WebSocket streaming endpoint for real-time AI responses
#   - /auth      → Login, token generation (JWT)
#   - /threads   → Chat history and thread management
#   - middleware → Auth guards, rate limiting, CORS
# This is the entry point for all client-facing communication.
