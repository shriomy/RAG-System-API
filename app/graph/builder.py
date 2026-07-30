"""Graph assembly.

The initial graph is exactly:

    START
      -> load_assistant       AssistantService   (config + model)
      -> load_user_memory     MemoryService      (summaries + recent messages)
      -> retrieve_documents   RetrievalService   (sources -> merge -> rerank)
      -> build_prompt         app/ai/prompts     (LangChain templates)
      -> llm                  OpenRouterService  (streaming)
      -> save_conversation    ConversationService
      -> update_memory        MemoryService
      -> END

Where later features attach — see the numbered comments below:

  (1) input guardrails    a node before load_assistant, or ChatService pre-check
  (2) query rewriting     a node between load_user_memory and retrieve_documents
  (3) tool loop           conditional edge llm -> tools -> llm
  (4) checkpointer        pass one to `compile()` for durable runs / HITL
  (5) more sources        inside RetrievalService; the graph is unaffected

Only (3) changes the topology; the rest are node insertions or service-internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger
from app.graph.nodes import (
    make_build_prompt_node,
    make_llm_node,
    make_load_assistant_node,
    make_load_memory_node,
    make_retrieve_documents_node,
    make_save_conversation_node,
    make_update_memory_node,
)
from app.graph.state import AgentState
from app.infrastructure.guardrails.pipeline import GuardrailPipeline
from app.infrastructure.tools.registry import ToolResolver
from app.services.assistant_service import AssistantService
from app.services.conversation_service import ConversationService
from app.services.memory_service import MemoryService
from app.services.openrouter_service import OpenRouterService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)

# Node names. Referenced by ChatService when interpreting stream events, so they
# live as constants rather than string literals.
LOAD_ASSISTANT = "load_assistant"
LOAD_USER_MEMORY = "load_user_memory"
RETRIEVE_DOCUMENTS = "retrieve_documents"
BUILD_PROMPT = "build_prompt"
LLM = "llm"
SAVE_CONVERSATION = "save_conversation"
UPDATE_MEMORY = "update_memory"

NODE_SEQUENCE = (
    LOAD_ASSISTANT,
    LOAD_USER_MEMORY,
    RETRIEVE_DOCUMENTS,
    BUILD_PROMPT,
    LLM,
    SAVE_CONVERSATION,
    UPDATE_MEMORY,
)


@dataclass(slots=True)
class GraphDependencies:
    """Everything the nodes need. Assembled once by the container."""

    assistant_service: AssistantService
    memory_service: MemoryService
    retrieval_service: RetrievalService
    conversation_service: ConversationService
    llm_service: OpenRouterService
    tool_resolver: ToolResolver
    guardrails: GuardrailPipeline


def build_nodes(deps: GraphDependencies) -> dict[str, Callable[..., Any]]:
    """Instantiate every node callable from its factory."""
    return {
        LOAD_ASSISTANT: make_load_assistant_node(deps.assistant_service),
        LOAD_USER_MEMORY: make_load_memory_node(deps.memory_service),
        RETRIEVE_DOCUMENTS: make_retrieve_documents_node(deps.retrieval_service),
        BUILD_PROMPT: make_build_prompt_node(),
        LLM: make_llm_node(
            deps.llm_service,
            tool_resolver=deps.tool_resolver,
            guardrails=deps.guardrails,
        ),
        SAVE_CONVERSATION: make_save_conversation_node(deps.conversation_service),
        UPDATE_MEMORY: make_update_memory_node(deps.memory_service),
    }


def build_agent_graph(deps: GraphDependencies, *, checkpointer: Any | None = None):
    """Compile the agent graph.

    `checkpointer` is extension point (4): pass a LangGraph checkpointer (e.g.
    AsyncPostgresSaver) to get durable execution, resumable runs and
    human-in-the-loop interrupts. Nothing else needs to change — AgentState is
    already fully JSON-serialisable.
    """
    graph = StateGraph(AgentState)

    nodes = build_nodes(deps)
    for name, node in nodes.items():
        graph.add_node(name, node)

    # (1) Input guardrails would be added here, ahead of load_assistant:
    #     graph.add_node(GUARD_INPUT, make_guard_input_node(deps.guardrails))
    #     graph.add_edge(START, GUARD_INPUT); graph.add_edge(GUARD_INPUT, LOAD_ASSISTANT)
    graph.add_edge(START, LOAD_ASSISTANT)
    graph.add_edge(LOAD_ASSISTANT, LOAD_USER_MEMORY)

    # (2) Query rewriting / HyDE would slot in between these two nodes.
    graph.add_edge(LOAD_USER_MEMORY, RETRIEVE_DOCUMENTS)
    graph.add_edge(RETRIEVE_DOCUMENTS, BUILD_PROMPT)
    graph.add_edge(BUILD_PROMPT, LLM)

    # (3) The tool loop replaces this single edge with a conditional one:
    #     graph.add_conditional_edges(
    #         LLM, route_after_llm, {"tools": TOOLS, "continue": SAVE_CONVERSATION}
    #     )
    #     graph.add_edge(TOOLS, LLM)
    graph.add_edge(LLM, SAVE_CONVERSATION)

    graph.add_edge(SAVE_CONVERSATION, UPDATE_MEMORY)
    graph.add_edge(UPDATE_MEMORY, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Agent graph compiled: START -> %s -> END", " -> ".join(NODE_SEQUENCE))
    return compiled


def route_after_llm(state: AgentState) -> str:
    """Router for extension point (3), the tool loop.

    Present and tested-by-shape now so that enabling tools is a matter of
    registering the conditional edge above — not redesigning the graph.
    """
    if state.get("tool_calls"):
        return "tools"
    return "continue"
