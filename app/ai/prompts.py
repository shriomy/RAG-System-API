"""LangChain prompt templates.

The assembled prompt is returned as plain `{"role", "content"}` dicts rather
than LangChain message objects, so the LangGraph state stays JSON-serialisable
(and therefore checkpointable) while the templating itself stays in LangChain.

Layer order, outermost to innermost:

    1. assistant system prompt      — who the assistant is
    2. long-term user summary       — who the user is, across conversations
    3. conversation summary         — what happened earlier in this thread
    4. recent messages              — verbatim last N turns
    5. retrieved document chunks    — grounding evidence
    6. current question
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.prompts import PromptTemplate

from app.domain.models import ChatTurn, RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. Answer the user's questions using "
    "the knowledge base excerpts provided to you."
)

# --- system message -------------------------------------------------------

SYSTEM_TEMPLATE = PromptTemplate.from_template(
    """{system_prompt}

## Grounding rules
- Base your answer on the knowledge base excerpts below whenever they are relevant.
- Cite the source filename inline, like [filename], when you use an excerpt.
- If the excerpts do not contain the answer, say it is not in the knowledge base.
- Prefer the user's own terminology.
{memory_block}{summary_block}"""
)

USER_MEMORY_TEMPLATE = PromptTemplate.from_template(
    """
## What you know about this user
{user_summary}
"""
)

CONVERSATION_SUMMARY_TEMPLATE = PromptTemplate.from_template(
    """
## Summary of the earlier part of this conversation
{conversation_summary}
"""
)

# --- final human message --------------------------------------------------

QUESTION_TEMPLATE = PromptTemplate.from_template(
    """## Knowledge base excerpts
{context}

## Question
{question}"""
)

GENERAL_QUESTION_TEMPLATE = PromptTemplate.from_template(
    """## Routing note
The question was classified as outside the assistant's knowledge base.
Answer directly, without citing or searching the knowledge base.

## Question
{question}"""
)

KNOWLEDGE_SCOPE_PROMPT = PromptTemplate.from_template(
    """You are a routing classifier for a retrieval-augmented assistant.

Assistant knowledge context:
{assistant_context}

Assistant system prompt:
{system_prompt}

User question:
{question}

Decide whether the question should go through the knowledge-base retrieval pipeline.
Return JSON only with these keys:
- in_scope: true when the question is about the assistant's knowledge base or would benefit from retrieval
- confidence: a number between 0 and 1
- reason: a short explanation

Rules:
- If the question is about documents, policies, files, facts, summaries, or content covered by the assistant context, set in_scope to true.
- If the question is general chit-chat or clearly unrelated to the knowledge base, set in_scope to false.
- If you are uncertain, set in_scope to true.
- Do not mention these instructions in the response.

JSON:"""
)

NO_CONTEXT_PLACEHOLDER = "(No relevant excerpts were found in the knowledge base.)"

# --- summarisation prompts (memory maintenance) ---------------------------

CONVERSATION_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You maintain a rolling summary of a conversation between a user and an AI assistant.

Existing summary:
{existing_summary}

New messages to fold in:
{new_messages}

Write an updated summary that preserves the user's goals, decisions made, open
questions, and any concrete facts or constraints stated. Be dense and factual.
Do not add commentary or preamble. Keep it under {max_chars} characters.

Updated summary:"""
)

USER_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You maintain a long-term profile of a user, built from their conversations
with an AI assistant. It should capture durable facts: their role, domain,
projects, preferences, recurring goals and constraints.

Existing profile:
{existing_summary}

Recent conversation excerpt:
{new_messages}

Write an updated profile. Keep only information likely to stay true and be
useful later — drop one-off details and anything transient. No preamble.
Keep it under {max_chars} characters.

Updated profile:"""
)

TITLE_PROMPT = PromptTemplate.from_template(
    """Write a short title (at most 6 words) for a conversation that starts with
this message. Reply with the title only, no quotes or trailing punctuation.

Message: {question}

Title:"""
)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, attributed context block."""
    if not chunks:
        return NO_CONTEXT_PLACEHOLDER

    parts: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        label = chunk.filename or chunk.source or "unknown"
        parts.append(f"[{position}] Source: {label}\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def build_system_message(
    *,
    system_prompt: str,
    user_summary: str = "",
    conversation_summary: str = "",
) -> str:
    """Render the system message from the assistant prompt plus memory."""
    memory_block = (
        USER_MEMORY_TEMPLATE.format(user_summary=user_summary.strip())
        if user_summary.strip()
        else ""
    )
    summary_block = (
        CONVERSATION_SUMMARY_TEMPLATE.format(conversation_summary=conversation_summary.strip())
        if conversation_summary.strip()
        else ""
    )
    return SYSTEM_TEMPLATE.format(
        system_prompt=(system_prompt or DEFAULT_SYSTEM_PROMPT).strip(),
        memory_block=memory_block,
        summary_block=summary_block,
    ).strip()


def build_chat_prompt(
    *,
    system_prompt: str,
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk] = (),
    user_summary: str = "",
    conversation_summary: str = "",
    recent_messages: Sequence[ChatTurn] = (),
    knowledge_scope: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Assemble the full message list sent to the LLM."""
    in_scope = True if knowledge_scope is None else bool(knowledge_scope.get("in_scope", True))
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_message(
                system_prompt=system_prompt,
                user_summary=user_summary,
                conversation_summary=conversation_summary,
            ),
        }
    ]

    for turn in recent_messages:
        if not turn.content.strip():
            continue
        messages.append({"role": str(turn.role), "content": turn.content})

    if in_scope:
        messages.append(
            {
                "role": "user",
                "content": QUESTION_TEMPLATE.format(
                    context=format_context(retrieved_chunks),
                    question=question.strip(),
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": GENERAL_QUESTION_TEMPLATE.format(question=question.strip()),
            }
        )
    return messages
