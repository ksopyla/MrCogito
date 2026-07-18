"""Narrative fidelity judge (LLM reject-only; never overrides gold)."""

from __future__ import annotations

from typing import Any

from data.deductive_stories.expand.writer import (
    AzureChatClient,
    judge_fidelity_llm,
    judge_fidelity_mock,
)
from data.deductive_stories.schema import DeductiveExample


def run_fidelity_judge(
    example: DeductiveExample,
    *,
    client: AzureChatClient | None = None,
    mock_llm: bool = False,
) -> tuple[bool, dict[str, Any]]:
    if mock_llm or client is None:
        result = judge_fidelity_mock(example.graph, example.story_text)
        example.judge_model = "mock_llm"
    else:
        result = judge_fidelity_llm(client, example.graph, example.story_text)
        example.judge_model = client.config.judge_deployment
    ok = bool(result.get("pass")) and not result.get("missing_event_ids")
    if not ok:
        example.accepted = False
        example.filter_notes.append(f"fidelity_fail:{result}")
    return ok, result
