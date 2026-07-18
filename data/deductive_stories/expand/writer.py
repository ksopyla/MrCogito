"""Azure OpenAI (and mock) clients for narrative expansion / judging."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from data.deductive_stories.schema import ChapterOutline, EventGraph


@dataclass
class LLMConfig:
    api_key: str
    endpoint: str
    api_version: str
    writer_deployment: str
    judge_deployment: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        ).strip()
        writer = (
            os.environ.get("AZURE_OPENAI_WRITER_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or ""
        ).strip()
        judge = (
            os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT") or writer
        ).strip()
        missing = [
            name
            for name, val in (
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_WRITER_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT", writer),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                "Missing Azure OpenAI env vars: "
                + ", ".join(missing)
                + ". Copy from .env.example into .env."
            )
        return cls(
            api_key=api_key,
            endpoint=endpoint.rstrip("/"),
            api_version=api_version,
            writer_deployment=writer,
            judge_deployment=judge or writer,
        )


class AzureChatClient:
    """Thin wrapper around openai.AzureOpenAI chat completions."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for Azure expansion. "
                "Install with: uv pip install openai"
            ) from exc
        self._client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.endpoint,
        )

    def chat(
        self,
        *,
        deployment: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        response = self._client.chat.completions.create(
            model=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()


def mock_expand_chapter(
    graph: EventGraph,
    chapter: ChapterOutline,
    *,
    words_target: int = 220,
) -> str:
    """Deterministic prose for tests / --mock-llm (no API calls)."""
    facts = chapter.bullet_facts or ["The investigation continued without decisive news."]
    body = " ".join(facts)
    # Pad with neutral filler so length-view tests have material.
    filler = (
        f" Meanwhile, clerks filed routine paperwork and the weather stayed dull. "
        f"None of this paperwork named a final conclusion. "
    )
    text = f"## {chapter.title}\n\n{body}"
    while len(text.split()) < words_target:
        text += filler
    # Never append gold answers as a solution block.
    return text.strip()


def expand_chapter_with_llm(
    client: AzureChatClient,
    graph: EventGraph,
    chapter: ChapterOutline,
    *,
    words_target: int = 400,
) -> str:
    system = (
        "You write long-form narrative chapters for a deductive puzzle dataset. "
        "Include EVERY bullet fact naturally. Do NOT state the final answers "
        "to the puzzle questions. Do NOT add contradictory facts. "
        "Write in clear English prose."
    )
    user = (
        f"Domain: {graph.domain}\n"
        f"Chapter title: {chapter.title}\n"
        f"Target length: ~{words_target} words\n"
        f"Must include these facts:\n"
        + "\n".join(f"- {b}" for b in chapter.bullet_facts)
        + "\n\nWrite the chapter now."
    )
    return client.chat(
        deployment=client.config.writer_deployment,
        system=system,
        user=user,
        temperature=0.8,
        max_tokens=min(4096, max(512, words_target * 2)),
    )


def expand_story(
    graph: EventGraph,
    outline: list[ChapterOutline],
    *,
    client: AzureChatClient | None = None,
    mock_llm: bool = False,
    words_per_chapter: int = 400,
) -> tuple[str, str]:
    """Return (story_text, writer_model_id)."""
    chapters_out: list[str] = []
    if mock_llm or client is None:
        writer_id = "mock_llm"
        for chapter in outline:
            chapters_out.append(
                mock_expand_chapter(
                    graph, chapter, words_target=max(80, words_per_chapter // 2)
                )
            )
    else:
        writer_id = client.config.writer_deployment
        for chapter in outline:
            chapters_out.append(
                expand_chapter_with_llm(
                    client, graph, chapter, words_target=words_per_chapter
                )
            )
    return "\n\n".join(chapters_out).strip(), writer_id


def judge_fidelity_mock(graph: EventGraph, story: str) -> dict[str, Any]:
    missing = []
    story_l = story.lower()
    for event in graph.events:
        if event.attrs.get("distractor"):
            continue
        if not event.text_seed:
            continue
        # Require a few distinctive tokens from the seed to appear.
        tokens = [t for t in re.findall(r"[a-z0-9]+", event.text_seed.lower()) if len(t) > 3]
        hit = sum(1 for t in tokens[:6] if t in story_l)
        if tokens and hit < max(1, min(2, len(tokens) // 3)):
            missing.append(event.id)
    return {
        "pass": len(missing) == 0,
        "missing_event_ids": missing,
        "contradictions": [],
    }


def judge_fidelity_llm(
    client: AzureChatClient,
    graph: EventGraph,
    story: str,
) -> dict[str, Any]:
    facts = [
        {"event_id": e.id, "fact": e.text_seed}
        for e in graph.events
        if e.text_seed and not e.attrs.get("distractor")
    ]
    system = (
        "You are a strict fidelity checker. Given a story and required facts, "
        "return JSON only: "
        '{"pass": bool, "missing_event_ids": [str], "contradictions": [str]}. '
        "pass=false if any required fact is missing or contradicted. "
        "Do not rewrite the story."
    )
    user = json.dumps({"required_facts": facts, "story": story[:120000]}, ensure_ascii=True)
    raw = client.chat(
        deployment=client.config.judge_deployment,
        system=system,
        user=user,
        temperature=0.0,
        max_tokens=1024,
    )
    try:
        # Allow fenced JSON.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return {
            "pass": False,
            "missing_event_ids": ["<judge_parse_error>"],
            "contradictions": [raw[:500]],
        }
    return {
        "pass": bool(payload.get("pass")),
        "missing_event_ids": list(payload.get("missing_event_ids") or []),
        "contradictions": list(payload.get("contradictions") or []),
    }
