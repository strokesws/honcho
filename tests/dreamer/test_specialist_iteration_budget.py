"""The dream specialists' tool-iteration budget and commit-tool reservation.

`DREAM_MAX_TOOL_ITERATIONS` (`settings.DREAM.MAX_TOOL_ITERATIONS`, default 20)
documents itself as the agent iteration limit for the extended reasoning
workflow, but only `CardRefreshSpecialist` ever read it: deduction returned the
literal 12 and induction the literal 10. Measured on the pinned stack, 7 of 8
deduction runs hit that literal with zero write calls — the budget was spent on
discovery and the run ended tool-less, so `create_observations_deductive` was
unreachable by construction.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.dreamer.specialists import (
    BaseSpecialist,
    CardRefreshSpecialist,
    DeductionSpecialist,
    InductionSpecialist,
)
from src.llm import HonchoLLMCallResponse

# The literals the specialists used to return instead of reading the setting.
LEGACY_DEDUCTION_LITERAL = 12
LEGACY_INDUCTION_LITERAL = 10


class TestConfiguredBudget:
    @pytest.mark.parametrize(
        "specialist",
        [DeductionSpecialist(), InductionSpecialist()],
        ids=["deduction", "induction"],
    )
    def test_default_budget_is_the_configured_value(self, specialist: BaseSpecialist):
        assert specialist.get_max_iterations() == settings.DREAM.MAX_TOOL_ITERATIONS

    def test_deduction_no_longer_uses_the_legacy_literal(self):
        # The default setting (20) is above the old literal, so assert the
        # relationship rather than a number that a future default would break.
        assert DeductionSpecialist().get_max_iterations() != LEGACY_DEDUCTION_LITERAL
        assert DeductionSpecialist().get_max_iterations() == (
            settings.DREAM.MAX_TOOL_ITERATIONS
        )

    def test_induction_no_longer_uses_the_legacy_literal(self):
        assert InductionSpecialist().get_max_iterations() != LEGACY_INDUCTION_LITERAL
        assert InductionSpecialist().get_max_iterations() == (
            settings.DREAM.MAX_TOOL_ITERATIONS
        )

    @pytest.mark.parametrize("configured", [3, 7, 25, 50])
    def test_budget_follows_the_setting_in_both_directions(
        self, monkeypatch: pytest.MonkeyPatch, configured: int
    ):
        """A value below the legacy literal must bind too — no hidden floor."""
        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", configured)

        assert DeductionSpecialist().get_max_iterations() == configured
        assert InductionSpecialist().get_max_iterations() == configured

    def test_card_refresh_keeps_its_own_ceiling(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 4)
        assert CardRefreshSpecialist().get_max_iterations() == 4

        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 30)
        assert CardRefreshSpecialist().get_max_iterations() == 6


class TestCommitTools:
    def test_base_default_is_no_reservation(self):
        assert BaseSpecialist.commit_tool_names == ()

    @pytest.mark.parametrize(
        "specialist,write_tool",
        [
            (DeductionSpecialist(), "create_observations_deductive"),
            (InductionSpecialist(), "create_observations_inductive"),
            (CardRefreshSpecialist(), "update_peer_card"),
        ],
        ids=["deduction", "induction", "card_refresh"],
    )
    def test_every_commit_tool_is_offered(
        self, specialist: BaseSpecialist, write_tool: str
    ):
        offered = {t["name"] for t in specialist.get_tools()}
        assert specialist.commit_tool_names
        assert set(specialist.commit_tool_names) <= offered
        assert write_tool in specialist.commit_tool_names

    def test_commit_tools_are_a_strict_subset(self):
        """The loop only engages the commit turn for a strict subset: a
        reservation covering every tool would change the prompt for nothing."""
        for specialist in (
            DeductionSpecialist(),
            InductionSpecialist(),
            CardRefreshSpecialist(),
        ):
            offered = {t["name"] for t in specialist.get_tools()}
            assert 0 < len(specialist.commit_tool_names) < len(offered)

    def test_deduction_commit_tools_exclude_discovery(self):
        assert "search_memory" not in DeductionSpecialist.commit_tool_names
        assert "get_recent_observations" not in DeductionSpecialist.commit_tool_names

    def test_deduction_card_tool_drops_out_when_cards_are_disabled(self):
        """The loop intersects the reservation with the offered tools, so a run
        without card access simply has a smaller commit turn."""
        specialist = DeductionSpecialist()
        offered = {t["name"] for t in specialist.get_tools(peer_card_enabled=False)}
        commit_tools = offered & set(specialist.commit_tool_names)
        assert commit_tools == {"create_observations_deductive", "delete_observations"}


class TestBudgetReachesTheLlmCall:
    @pytest.mark.asyncio
    async def test_run_forwards_budget_and_commit_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.METRICS, "ENABLED", False)
        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 9)

        mock_llm_call = AsyncMock(
            return_value=HonchoLLMCallResponse(
                content="done",
                input_tokens=10,
                output_tokens=5,
                finish_reasons=["stop"],
            )
        )

        with (
            patch("src.dreamer.specialists.crud.get_peer", new=AsyncMock()),
            patch(
                "src.dreamer.specialists.crud.get_peer_card",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "src.dreamer.specialists.create_tool_executor",
                new=AsyncMock(return_value=AsyncMock()),
            ),
            patch(
                "src.dreamer.specialists.honcho_llm_call",
                new=mock_llm_call,
            ),
        ):
            result = await DeductionSpecialist().run(
                workspace_name="workspace",
                observer="alice",
                observed="alice",
                session_name=None,
            )

        assert result.success is True
        assert mock_llm_call.await_args is not None
        kwargs = mock_llm_call.await_args.kwargs
        assert kwargs["max_tool_iterations"] == 9
        assert kwargs["commit_tool_names"] == [
            "create_observations_deductive",
            "delete_observations",
            "update_peer_card",
        ]
