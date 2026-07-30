"""Graph nodes.

Each module exposes a `make_*_node(services...)` factory returning the async
callable LangGraph invokes. Factories (rather than classes or global lookups)
keep dependencies explicit and each node trivially testable with fakes.
"""

from app.graph.nodes.build_prompt import make_build_prompt_node
from app.graph.nodes.llm import make_llm_node
from app.graph.nodes.load_assistant import make_load_assistant_node
from app.graph.nodes.load_memory import make_load_memory_node
from app.graph.nodes.retrieve_documents import make_retrieve_documents_node
from app.graph.nodes.save_conversation import make_save_conversation_node
from app.graph.nodes.update_memory import make_update_memory_node

__all__ = [
    "make_build_prompt_node",
    "make_llm_node",
    "make_load_assistant_node",
    "make_load_memory_node",
    "make_retrieve_documents_node",
    "make_save_conversation_node",
    "make_update_memory_node",
]
