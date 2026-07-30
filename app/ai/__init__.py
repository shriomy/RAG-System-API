"""AI layer — the ONLY package allowed to import LangChain.

Contents map 1:1 to the LangChain components we actually want:

    loaders.py      document loaders (PDF / TXT / Markdown)
    splitters.py    text splitters
    embeddings.py   embedding models
    vectorstore.py  Qdrant vector-store + retriever adapters
    prompts.py      prompt templates
    llm.py          chat models (OpenRouter via the OpenAI-compatible client)
    parsers.py      output parsers
    rerankers.py    reranking models

Everything outside this package — routes, services, repositories, graph nodes —
talks to it through the ports in app/domain/ports.py, so LangChain never leaks
into application logic and can be upgraded or replaced in isolation.
"""
