"""GraphService — owns the compiled LangGraph and how it is executed.

The graph is compiled once at startup (compilation is not free, and the node
callables are stateless), then invoked per request.

Streaming design: LangGraph's own stream modes emit *state updates*, not LLM
token deltas, and the token-level event APIs differ across versions. So the LLM
node is handed an `on_token` callback through the run config, and this service
bridges that callback to an async generator with a queue. The result is a
transport that does not depend on LangGraph internals, works identically whether
tokens come from OpenRouter or anything else, and leaves ChatService free of
concurrency plumbing.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from app.core.errors import AppError
from app.core.logging import get_logger
from app.core.observability import run_metadata, tracing_callbacks
from app.graph.builder import GraphDependencies, build_agent_graph
from app.graph.state import AgentState, initial_state

logger = get_logger(__name__)

_SENTINEL = object()

#: Detached graph runs, kept referenced so the event loop does not GC them
#: mid-flight when a client disconnects.
_BACKGROUND_RUNS: set[asyncio.Task[Any]] = set()


class GraphService:
    def __init__(self, deps: GraphDependencies, *, checkpointer: Any | None = None) -> None:
        self._graph = build_agent_graph(deps, checkpointer=checkpointer)

    @property
    def graph(self) -> Any:
        return self._graph

    def new_state(
        self, *, user_id: str, assistant_id: str, conversation_id: str, question: str
    ) -> AgentState:
        return initial_state(
            user_id=user_id,
            assistant_id=assistant_id,
            conversation_id=conversation_id,
            question=question,
        )

    # ======================================================================
    # Non-streaming
    # ======================================================================

    async def invoke(self, state: AgentState) -> AgentState:
        """Run the graph to completion and return the final state."""
        result = await self._graph.ainvoke(state, config=self._config(state))
        return result  # type: ignore[return-value]

    # ======================================================================
    # Streaming
    # ======================================================================

    async def stream(self, state: AgentState) -> AsyncIterator[tuple[str, Any]]:
        """Run the graph, yielding ('token' | 'revision' | 'result', payload).

        Exactly one 'result' is yielded last, carrying the final AgentState —
        which means the nodes after the LLM (save_conversation, update_memory)
        have already run by the time the caller sees it.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def on_token(token: str) -> None:
            await queue.put(("token", token))

        async def on_revision(text: str) -> None:
            await queue.put(("revision", text))

        config = self._config(state, on_token=on_token, on_revision=on_revision)

        async def runner() -> None:
            try:
                final = await self._graph.ainvoke(state, config=config)
                await queue.put(("result", final))
            except AppError as exc:
                await queue.put(("error", exc))
            except Exception as exc:
                logger.exception("Graph run failed")
                await queue.put(("error", exc))
            finally:
                await queue.put(_SENTINEL)

        task = asyncio.create_task(runner(), name="agent-graph-run")
        _BACKGROUND_RUNS.add(task)
        task.add_done_callback(_BACKGROUND_RUNS.discard)

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                kind, payload = item
                if kind == "error":
                    raise payload
                yield kind, payload
        finally:
            # Deliberately NOT cancelled. If the client disconnects mid-stream the
            # answer may already be generated; letting the run finish means
            # save_conversation and update_memory still execute, so the turn shows
            # up in the user's history rather than vanishing. The task is held in
            # _BACKGROUND_RUNS until it completes.
            if not task.done():
                logger.debug("Client stopped consuming; graph run continues detached")

    # ======================================================================
    # Internals
    # ======================================================================

    @staticmethod
    def _config(
        state: AgentState,
        *,
        on_token: Any | None = None,
        on_revision: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "run_name": "agentic_rag",
            "callbacks": tracing_callbacks(),
            "metadata": run_metadata(
                user_id=state.get("user_id"),
                assistant_id=state.get("assistant_id"),
                conversation_id=state.get("conversation_id"),
            ),
            "configurable": {
                # Node-visible knobs. `thread_id` is what a checkpointer would
                # key on, so durable runs need no further wiring.
                "thread_id": state.get("conversation_id"),
                "on_token": on_token,
                "on_revision": on_revision,
            },
            # Guards against a pathological tool loop once extension point (3)
            # is enabled; the linear graph never approaches it.
            "recursion_limit": 25,
        }

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait for detached runs to finish. Called during shutdown."""
        if not _BACKGROUND_RUNS:
            return
        pending = list(_BACKGROUND_RUNS)
        logger.info("Waiting for %d in-flight graph run(s)", len(pending))
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        if still_pending:
            logger.warning("%d graph run(s) did not finish before shutdown", len(still_pending))
