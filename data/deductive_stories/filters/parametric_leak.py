"""Parametric-leak filter: reject if answerable without the story."""

from __future__ import annotations

import re
from typing import Any

from data.deductive_stories.expand.writer import AzureChatClient
from data.deductive_stories.schema import DeductiveExample, normalize_answer


def _extract_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"(?i)answer\s*:\s*(.+)", text)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return text.split("\n")[0].strip()


def check_parametric_leak_mock(example: DeductiveExample) -> dict[str, Any]:
    """Mock: assume no leak (graphs use invented names unlikely to be memorized)."""
    return {"pass": True, "leaked_qids": [], "raw": {}}


def check_parametric_leak_llm(
    example: DeductiveExample,
    client: AzureChatClient,
    *,
    passes: int = 1,
) -> dict[str, Any]:
    leaked: list[str] = []
    raw: dict[str, list[str]] = {}
    for query, gold in zip(example.graph.queries, example.graph.gold, strict=True):
        system = (
            "Answer with a short exact value only, prefixed by 'Answer:'. "
            "If you do not know, answer 'Answer: unknown'."
        )
        user = (
            "No document is provided. Using only parametric knowledge, answer:\n"
            f"{query.prompt}"
        )
        responses = []
        for _ in range(max(1, passes)):
            responses.append(
                client.chat(
                    deployment=client.config.judge_deployment,
                    system=system,
                    user=user,
                    temperature=0.0,
                    max_tokens=64,
                )
            )
        raw[query.qid] = responses
        for resp in responses:
            pred = normalize_answer(
                _extract_answer(resp), answer_type=query.answer_type
            )
            if pred and pred != "unknown" and pred == gold.normalized:
                leaked.append(query.qid)
                break
    return {"pass": len(leaked) == 0, "leaked_qids": leaked, "raw": raw}


def run_parametric_leak_filter(
    example: DeductiveExample,
    *,
    client: AzureChatClient | None = None,
    mock_llm: bool = False,
) -> tuple[bool, dict[str, Any]]:
    if mock_llm or client is None:
        result = check_parametric_leak_mock(example)
    else:
        result = check_parametric_leak_llm(example, client)
    if not result["pass"]:
        example.accepted = False
        example.filter_notes.append(f"parametric_leak:{result['leaked_qids']}")
    return bool(result["pass"]), result
