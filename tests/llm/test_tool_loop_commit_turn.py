"""The tool loop's commit turn (`commit_tool_names`).

An exhausted tool loop used to end with one tool-LESS synthesis call, so a
caller whose run must end in a write tool (the dream specialists' deduction /
induction create_observations*) could never reach it once the iteration budget
was spent on discovery — the model could only print the payload as prose.
Measured on the pinned stack: 7 of 8 deduction runs, ~13 LLM turns each, zero
writes.

`commit_tool_names` reserves the final permitted iteration for those tools.
These tests pin both halves: the old tool-less ending (left in place for every
caller that does not opt in) and the commit turn.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.config import ModelConfig
from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse, ProviderClient

SEARCH = {
    "name": "search_memory",
    "description": "Search memory",
    "input_schema": {"type": "object"},
}
COMMIT = {
    "name": "create_observations_deductive",
    "description": "Commit deductive observations",
    "input_schema": {"type": "object"},
}


def _make_plan() -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="deepseek-v4-pro",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(model="deepseek-v4-pro", transport="openai"),
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


def _search_call(index: int) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="",
        output_tokens=5,
        finish_reasons=["tool_calls"],
        tool_calls_made=[
            {"id": f"call_{index}", "name": "search_memory", "input": {"query": "x"}}
        ],
    )


def _commit_call() -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="",
        output_tokens=8,
        finish_reasons=["tool_calls"],
        tool_calls_made=[
            {
                "id": "call_commit",
                "name": "create_observations_deductive",
                "input": {"observations": []},
            }
        ],
    )


class _ScriptedModel:
    """Keeps searching while `search_memory` is offered; commits once it is not.

    That is the behaviour the live log shows: discovery tool calls only, right
    up to the moment the loop takes the tools away.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        offered = [t["name"] for t in (kwargs.get("tools") or [])]
        self.calls.append(
            {
                "tools": deepcopy(kwargs.get("tools")),
                "tool_choice": kwargs.get("tool_choice"),
                "messages": deepcopy(kwargs.get("messages")),
            }
        )
        if "search_memory" in offered:
            return _search_call(len(self.calls))
        if "create_observations_deductive" in offered:
            return _commit_call()
        return HonchoLLMCallResponse(
            content="Unable to call further tools — here is the output I would "
            "have committed.",
            output_tokens=40,
            finish_reasons=["stop"],
            tool_calls_made=[],
        )


async def _run(
    model: _ScriptedModel,
    executed: list[str],
    *,
    max_tool_iterations: int,
    commit_tool_names: list[str] | None,
) -> HonchoLLMCallResponse[Any]:
    async def execute(name: str, _input: dict[str, Any]) -> str:
        executed.append(name)
        return "ok"

    with patch.object(tool_loop, "honcho_llm_call_inner", new=model):
        result = await execute_tool_loop(
            prompt="deduce",
            max_tokens=64,
            messages=[{"role": "user", "content": "deduce"}],
            tools=[SEARCH, COMMIT],
            tool_choice=None,
            tool_executor=execute,
            max_tool_iterations=max_tool_iterations,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _retry_state: None,
            stream_final=False,
            telemetry=None,
            commit_tool_names=commit_tool_names,
        )
    assert isinstance(result, HonchoLLMCallResponse)
    return result


@pytest.mark.asyncio
async def test_without_commit_tools_the_last_call_is_toolless(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unchanged behaviour for every caller that does not opt in."""
    model = _ScriptedModel()
    executed: list[str] = []

    with caplog.at_level(logging.WARNING, logger="src.llm.tool_loop"):
        await _run(
            model,
            executed,
            max_tool_iterations=3,
            commit_tool_names=None,
        )

    assert len(model.calls) == 4  # 3 discovery turns + the tool-less synthesis
    assert model.calls[-1]["tools"] is None
    assert executed == ["search_memory"] * 3
    assert "reached max iterations (3)" in caplog.text


@pytest.mark.asyncio
async def test_commit_turn_reserves_the_final_iteration_for_the_write_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _ScriptedModel()
    executed: list[str] = []

    with caplog.at_level(logging.WARNING, logger="src.llm.tool_loop"):
        result = await _run(
            model,
            executed,
            max_tool_iterations=3,
            commit_tool_names=["create_observations_deductive"],
        )

    # No fourth call: the commit turn IS the last turn, so no tool-less
    # synthesis call follows it.
    assert len(model.calls) == 3
    assert [t["name"] for t in model.calls[-1]["tools"]] == [
        "create_observations_deductive"
    ]
    # The model is told what the final turn is for before it is called.
    assert "final turn" in model.calls[-1]["messages"][-1]["content"]
    # And the write actually ran.
    assert executed == [
        "search_memory",
        "search_memory",
        "create_observations_deductive",
    ]
    assert [tc["tool_name"] for tc in result.tool_calls_made] == executed
    assert "Unable to call further tools" not in str(result.content)
    assert "reached max iterations" not in caplog.text


@pytest.mark.asyncio
async def test_commit_names_covering_every_tool_leave_the_loop_unchanged() -> None:
    """A filter that keeps the whole toolset is not a commit turn."""
    model = _ScriptedModel()
    executed: list[str] = []

    await _run(
        model,
        executed,
        max_tool_iterations=2,
        commit_tool_names=["search_memory", "create_observations_deductive"],
    )

    # Both turns offered the full toolset and the run still ended tool-less.
    assert [t["name"] for t in model.calls[1]["tools"]] == [
        "search_memory",
        "create_observations_deductive",
    ]
    assert model.calls[-1]["tools"] is None
    assert executed == ["search_memory"] * 2
