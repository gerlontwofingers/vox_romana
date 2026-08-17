"""Focused tests for perspective-aware research wiring.

These cover the next-chunk work after perspective-aware fact checking:
  - the live reference-material path (DebateResearcher),
  - the structured analysis path (DataResearcher),
  - bootstrap forwarding of explicit perspective overrides into auto-research.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from vox_romana.core.data_researcher import DataResearcher
from vox_romana.core.search import SearchResult
from vox_romana.shows.debates.researcher import DebateResearcher


class TestDebateResearcherPerspective:
    def test_formulate_queries_includes_perspective_context(self):
        client = Mock()
        client.chat.return_value = '["guatemala border policy"]'

        researcher = DebateResearcher(client)
        researcher.formulate_queries(
            "Local news on migration",
            perspective="from Guatemala",
        )

        user_msg = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "PERSPECTIVE CONTEXT" in user_msg
        assert "from Guatemala" in user_msg

    def test_compile_results_includes_perspective_context(self):
        client = Mock()
        client.chat.return_value = "## Context\n- Fact"

        researcher = DebateResearcher(client)
        researcher.compile_results(
            "Local news on migration",
            [SearchResult(title="A", url="u", snippet="Fact snippet")],
            perspective="from Guatemala",
        )

        user_msg = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "PERSPECTIVE CONTEXT" in user_msg
        assert "from Guatemala" in user_msg

    def test_run_from_knowledge_includes_perspective_context(self):
        client = Mock()
        client.chat.return_value = "## Context\n- Fact"

        researcher = DebateResearcher(client)
        researcher.run_from_knowledge(
            "Explain the northern border",
            perspective="from Guatemala",
        )

        user_msg = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "PERSPECTIVE CONTEXT" in user_msg
        assert "from Guatemala" in user_msg


class TestDataResearcherPerspective:
    def test_run_includes_perspective_in_web_analysis_and_summary_prompts(self):
        client = Mock()
        client.chat.side_effect = [
            '["guatemala migration background"]',
            '{"findings": "Context", "fact_cards": []}',
            "Summary",
        ]
        search = Mock()
        search.search.return_value = [
            SearchResult(title="A", url="u", snippet="Snippet"),
        ]

        researcher = DataResearcher(client=client, catalog=None, search_provider=search)
        researcher.run("Migration trends", perspective="from Guatemala")

        messages = [call.kwargs["messages"] for call in client.chat.call_args_list]
        assert "from Guatemala" in messages[0][1]["content"]
        assert "from Guatemala" in messages[1][1]["content"]
        assert "from Guatemala" in messages[2][1]["content"]

    def test_plan_analysis_includes_perspective_context(self):
        client = Mock()
        client.chat.return_value = '{"recipe": "compare_means", "params": {}, "rationale": "best"}'

        researcher = DataResearcher(client=client, catalog=None)
        researcher._plan_analysis(
            "Employment analysis",
            "Columns: x, y",
            {"compare_means": {"description": "desc", "params": []}},
            "(no prior studies found)",
            perspective="from Guatemala",
        )

        user_msg = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "PERSPECTIVE CONTEXT" in user_msg
        assert "from Guatemala" in user_msg


class TestBootstrapResearchPerspectiveForwarding:
    def test_debate_bootstrap_forwards_perspective_override(self):
        from vox_romana.shows.debates.bootstrap import DebateBootstrap

        bootstrap = DebateBootstrap(client=Mock())
        bootstrap._architect.run = Mock(return_value=SimpleNamespace(
            participants=[SimpleNamespace(name="Alice", side="pro")],
            show_host=None,
            topic="Topic",
            debate_format="public_forum",
            tone="balanced",
            target_turns=10,
            director_brief=None,
        ))
        bootstrap._profiler.run = Mock(side_effect=lambda stub, **kwargs: stub)
        bootstrap._director.run = Mock(return_value="director")

        with patch("vox_romana.shows.debates.bootstrap.DebateResearcher") as researcher_cls:
            researcher_cls.return_value.run.return_value = "facts"

            bootstrap.from_brief(
                "test debate",
                auto_research=True,
                overrides={"perspective": "from Guatemala"},
            )

        researcher_cls.return_value.run.assert_called_once_with(
            "test debate",
            perspective="from Guatemala",
        )

    def test_news_bootstrap_forwards_perspective_override(self):
        from vox_romana.shows.news.bootstrap import NewsBootstrap

        bootstrap = NewsBootstrap(client=Mock())
        bootstrap._run_architect = Mock(return_value=SimpleNamespace(
            anchors=[SimpleNamespace(name="Anchor")],
            show_host=SimpleNamespace(name="Host"),
            topic="Topic",
            tone="steady",
            director_brief=None,
        ))
        bootstrap._profile_anchor = Mock(side_effect=lambda anchor, **kwargs: anchor)
        bootstrap._profile_host = Mock()
        bootstrap._run_director = Mock(return_value="director")
        bootstrap._apply_overrides = Mock()

        with patch("vox_romana.shows.debates.researcher.DebateResearcher") as researcher_cls:
            researcher_cls.return_value.run.return_value = "facts"

            bootstrap.from_brief(
                "test news",
                auto_research=True,
                overrides={"perspective": "from Guatemala"},
            )

        researcher_cls.return_value.run.assert_called_once_with(
            "test news",
            perspective="from Guatemala",
        )

    def test_audiobook_bootstrap_forwards_perspective_override(self):
        from vox_romana.shows.audiobooks.bootstrap import AudiobookBootstrap

        bootstrap = AudiobookBootstrap(client=Mock())
        bootstrap._run_architect = Mock(return_value=SimpleNamespace(
            narrator=SimpleNamespace(name="Narrator"),
            genre="history",
            tone="reflective",
            director_brief=None,
        ))
        bootstrap._profile_narrator = Mock(side_effect=lambda narrator, **kwargs: narrator)
        bootstrap._run_director = Mock(return_value="director")
        bootstrap._apply_overrides = Mock()

        with patch("vox_romana.shows.debates.researcher.DebateResearcher") as researcher_cls:
            researcher_cls.return_value.run.return_value = "facts"

            bootstrap.from_brief(
                "test audiobook",
                auto_research=True,
                overrides={"perspective": "from Guatemala"},
            )

        researcher_cls.return_value.run.assert_called_once_with(
            "test audiobook",
            perspective="from Guatemala",
        )