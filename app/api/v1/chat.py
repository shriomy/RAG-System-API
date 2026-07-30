"""/chat — run the agent graph and stream the answer.

This route is the only place that knows about Server-Sent Events. ChatService
yields transport-neutral event dicts; formatting them for the wire happens here,
so swapping to WebSockets later would not touch the service or the graph.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import ChatServiceDep, CurrentUser
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Stops nginx from buffering the stream and defeating the point of it.
    "X-Accel-Buffering": "no",
}

STREAM_TERMINATOR = "[DONE]"


def _sse(payload: dict[str, Any]) -> str:
    """Format one event as an SSE frame.

    `separators` keeps frames compact, and `ensure_ascii=False` avoids mangling
    non-Latin output into escapes mid-stream.
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n"


@router.post(
    "",
    summary="Chat with an assistant (streamed)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "SSE stream. Each frame is `data: <json>`. Event types:\n\n"
                "- `start` — conversation_id, assistant_id, model, is_new_conversation\n"
                "- `token` — one content delta in `content`\n"
                "- `revision` — a guardrail rewrote the answer; replace rendered text\n"
                "- `done` — final answer, citations, usage, message ids, optional title\n"
                "- `error` — code and message\n\n"
                "The stream ends with `data: [DONE]`."
            ),
        }
    },
    description=(
        "Executes the LangGraph workflow: load assistant -> load user memory -> "
        "retrieve documents -> build prompt -> LLM -> save conversation -> update "
        "memory. Tokens are streamed as they are generated; the conversation and "
        "memory are persisted after the stream completes, before the `done` event."
    ),
)
async def chat(
    payload: ChatRequest, user: CurrentUser, service: ChatServiceDep
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.stream_chat(
                user_id=user.id,
                assistant_id=payload.assistant_id,
                conversation_id=payload.conversation_id,
                question=payload.question,
            ):
                yield _sse(event)
        except Exception as exc:
            # The response has already begun, so a normal error response is no
            # longer possible — the failure has to go out as a stream event.
            logger.exception("Chat stream failed")
            yield _sse(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": f"The stream failed: {exc}",
                }
            )
        finally:
            yield f"data: {STREAM_TERMINATOR}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post(
    "/sync",
    response_model=ChatResponse,
    summary="Chat with an assistant (single JSON response)",
    description=(
        "Same graph, no streaming. Convenient for testing, scripting and "
        "non-browser clients."
    ),
)
async def chat_sync(
    payload: ChatRequest, user: CurrentUser, service: ChatServiceDep
) -> ChatResponse:
    result = await service.chat(
        user_id=user.id,
        assistant_id=payload.assistant_id,
        conversation_id=payload.conversation_id,
        question=payload.question,
    )
    return ChatResponse.model_validate(result.model_dump())
