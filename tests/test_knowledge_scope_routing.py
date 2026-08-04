from app.domain.models import Assistant
from app.ai.prompts import build_chat_prompt
from app.graph.builder import route_after_knowledge_scope


def test_route_after_knowledge_scope_retrieves_for_in_scope_questions() -> None:
    assert route_after_knowledge_scope({"knowledge_scope": {"in_scope": True}}) == "retrieve"


def test_route_after_knowledge_scope_skips_for_out_of_scope_questions() -> None:
    assert route_after_knowledge_scope({"knowledge_scope": {"in_scope": False}}) == "skip"


def test_assistant_knowledge_scope_context_uses_config_then_system_prompt() -> None:
    assistant = Assistant.model_validate(
        {
            "id": "a1",
            "user_id": "u1",
            "name": "Demo",
            "system_prompt": "System fallback",
            "is_active": True,
            "model": "openai/gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
            "config": {"knowledge_scope": {"summary": "Known docs: policy manual"}},
        }
    )
    assert assistant.knowledge_scope_context() == "Known docs: policy manual"


def test_assistant_knowledge_scope_context_falls_back_to_system_prompt() -> None:
    assistant = Assistant.model_validate(
        {
            "id": "a1",
            "user_id": "u1",
            "name": "Demo",
            "system_prompt": "System fallback",
            "is_active": True,
            "model": "openai/gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
            "config": {},
        }
    )
    assert assistant.knowledge_scope_context() == "System fallback"


def test_out_of_scope_prompt_omits_knowledge_base_context_block() -> None:
    messages = build_chat_prompt(
        system_prompt="System fallback",
        question="What is the weather today?",
        retrieved_chunks=[],
        knowledge_scope={"in_scope": False},
    )

    assert messages[-1]["content"].startswith("## Routing note")
    assert "Knowledge base excerpts" not in messages[-1]["content"]