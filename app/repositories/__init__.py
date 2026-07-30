"""Repository layer — the ONLY place that talks to the database or storage.

Services call repositories. Graph nodes call services. Nothing else touches
persistence, and no SQL or HTTP-to-Supabase appears outside this package.
"""
