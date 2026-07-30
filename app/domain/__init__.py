"""Domain layer: models and ports (protocols).

Deliberately dependency-free — no FastAPI, no LangChain, no HTTP client. Both
the service layer and the infrastructure adapters depend on this module; it
depends on nothing, which is what keeps the pluggable seams honest.
"""
