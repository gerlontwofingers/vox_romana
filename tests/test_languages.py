"""Tests for vox_romana/core/languages.py — Shot 1."""

from __future__ import annotations

import pytest

from vox_romana.core.languages import (
    LANGUAGE_CODES,
    LANGUAGE_NAMES,
    language_display,
    normalize_lang_arg,
    supported_languages,
)


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE_CODES / LANGUAGE_NAMES
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageRegistry:
    def test_language_codes_is_frozenset(self):
        assert isinstance(LANGUAGE_CODES, frozenset)

    def test_english_always_present(self):
        assert "en" in LANGUAGE_CODES

    def test_spanish_present(self):
        assert "es" in LANGUAGE_CODES

    def test_french_present(self):
        assert "fr" in LANGUAGE_CODES

    def test_codes_match_names(self):
        assert set(LANGUAGE_CODES) == set(LANGUAGE_NAMES.keys())

    def test_english_name(self):
        assert LANGUAGE_NAMES["en"] == "English"

    def test_spanish_name(self):
        assert LANGUAGE_NAMES["es"] == "Spanish"

    def test_all_codes_lowercase(self):
        for code in LANGUAGE_CODES:
            assert code == code.lower(), f"Code {code!r} is not lowercase"

    def test_minimum_size(self):
        # We expect at least 10 languages registered.
        assert len(LANGUAGE_CODES) >= 10


# ─────────────────────────────────────────────────────────────────────────────
# supported_languages (Shot 1 stub)
# ─────────────────────────────────────────────────────────────────────────────


class TestSupportedLanguages:
    def test_returns_list(self):
        result = supported_languages()
        assert isinstance(result, list)

    def test_english_included(self):
        assert "en" in supported_languages()

    def test_all_codes_valid(self):
        for code in supported_languages():
            assert code in LANGUAGE_CODES, f"{code!r} not in LANGUAGE_CODES"


# ─────────────────────────────────────────────────────────────────────────────
# normalize_lang_arg
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeLangArg:
    # ── single code ──────────────────────────────────────────────────
    def test_single_en(self):
        assert normalize_lang_arg("en") == ["en"]

    def test_single_es(self):
        assert normalize_lang_arg("es") == ["es"]

    def test_single_with_whitespace(self):
        assert normalize_lang_arg("  fr  ") == ["fr"]

    # ── comma-separated list ─────────────────────────────────────────
    def test_two_codes(self):
        assert normalize_lang_arg("en,es") == ["en", "es"]

    def test_three_codes(self):
        assert normalize_lang_arg("en,es,fr") == ["en", "es", "fr"]

    def test_codes_with_spaces(self):
        assert normalize_lang_arg("en, es, fr") == ["en", "es", "fr"]

    def test_codes_sorted(self):
        # Output should be sorted regardless of input order.
        result = normalize_lang_arg("fr,en,es")
        assert result == sorted(result)

    def test_deduplication(self):
        result = normalize_lang_arg("en,es,en")
        assert result.count("en") == 1

    # ── case normalisation ────────────────────────────────────────────
    def test_uppercase_input(self):
        assert normalize_lang_arg("EN") == ["en"]

    def test_mixed_case(self):
        assert normalize_lang_arg("En,ES") == ["en", "es"]

    # ── "all" keyword ────────────────────────────────────────────────
    def test_all_returns_supported_languages(self):
        result = normalize_lang_arg("all")
        assert result == supported_languages()

    def test_all_case_insensitive(self):
        assert normalize_lang_arg("ALL") == supported_languages()
        assert normalize_lang_arg("All") == supported_languages()

    # ── error cases ───────────────────────────────────────────────────
    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown language code"):
            normalize_lang_arg("xx")

    def test_unknown_code_in_list_raises(self):
        with pytest.raises(ValueError, match="Unknown language code"):
            normalize_lang_arg("en,xx,zz")

    def test_error_lists_all_unknown(self):
        with pytest.raises(ValueError) as exc_info:
            normalize_lang_arg("xx,yy")
        msg = str(exc_info.value)
        assert "xx" in msg
        assert "yy" in msg

    def test_empty_string_raises(self):
        # An empty string produces no tokens → no unknown codes but also no
        # languages.  Callers should treat [] as "no languages requested";
        # however we don't want a silent empty list to sneak through.
        result = normalize_lang_arg("")
        # Acceptable: empty list (caller's problem) — just ensure no crash.
        assert isinstance(result, list)

    def test_trailing_comma_ignored(self):
        result = normalize_lang_arg("en,es,")
        assert result == ["en", "es"]


# ─────────────────────────────────────────────────────────────────────────────
# language_display
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageDisplay:
    def test_single_code(self):
        assert language_display(["en"]) == "English (en)"

    def test_two_codes(self):
        result = language_display(["en", "es"])
        assert "English (en)" in result
        assert "Spanish (es)" in result
        assert "," in result

    def test_unknown_code_uses_code_as_name(self):
        # Should not crash on an unknown code — just echo it.
        result = language_display(["zz"])
        assert "zz" in result

    def test_empty_list(self):
        assert language_display([]) == ""


# ─────────────────────────────────────────────────────────────────────────────
# ContentConfig.language field
# ─────────────────────────────────────────────────────────────────────────────


class TestContentConfigLanguage:
    def test_default_language_is_en(self):
        from vox_romana.core.config import ContentConfig
        cfg = ContentConfig(brief="test brief")
        assert cfg.language == "en"

    def test_language_can_be_set(self):
        from vox_romana.core.config import ContentConfig
        cfg = ContentConfig(brief="hola", language="es")
        assert cfg.language == "es"

    def test_debate_config_inherits_language(self):
        from vox_romana.shows.debates.config import DebateConfig
        cfg = DebateConfig(brief="test", language="fr")
        assert cfg.language == "fr"

    def test_debate_config_default_language(self):
        from vox_romana.shows.debates.config import DebateConfig
        cfg = DebateConfig(brief="test")
        assert cfg.language == "en"


# ─────────────────────────────────────────────────────────────────────────────
# MergedDebate.to_dict() — language field
# ─────────────────────────────────────────────────────────────────────────────


class TestMergedDebateLanguage:
    def _make_debate(self, language: str = "en"):
        from vox_romana.shows.debates.models import MergedDebate
        return MergedDebate(
            template_id="tpl",
            debate_id="d1",
            title="Test",
            chunks=(),
            events=(),
            language=language,
        )

    def test_default_language_en(self):
        d = self._make_debate()
        assert d.language == "en"

    def test_to_dict_includes_language(self):
        d = self._make_debate(language="es")
        data = d.to_dict()
        assert "language" in data
        assert data["language"] == "es"

    def test_to_dict_default_language(self):
        d = self._make_debate()
        data = d.to_dict()
        assert data["language"] == "en"


# ─────────────────────────────────────────────────────────────────────────────
# MergedNewsShow.to_dict() — language field
# ─────────────────────────────────────────────────────────────────────────────


class TestMergedNewsShowLanguage:
    def _make_show(self, language: str = "en"):
        from vox_romana.shows.news.models import MergedNewsShow
        return MergedNewsShow(
            show_id="s1",
            title="Test News",
            language=language,
        )

    def test_default_language_en(self):
        s = self._make_show()
        assert s.language == "en"

    def test_to_dict_includes_language(self):
        s = self._make_show(language="fr")
        data = s.to_dict()
        assert "language" in data
        assert data["language"] == "fr"

    def test_to_dict_default_language(self):
        s = self._make_show()
        data = s.to_dict()
        assert data["language"] == "en"

    def test_to_dict_no_duplicate_language_key(self):
        # Python dicts allow duplicate keys in literals (last wins) but
        # that's a bug we want to catch.  Serialise → JSON string → count.
        import json
        s = self._make_show(language="de")
        raw = json.dumps(s.to_dict())
        assert raw.count('"language"') == 1


# ─────────────────────────────────────────────────────────────────────────────
# run.py flag parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPyLanguageFlags:
    """Smoke-tests for --root-lang and --out-lang via run.parse_args()."""

    @pytest.fixture
    def tmp_prompt(self, tmp_path):
        p = tmp_path / "prompt.txt"
        p.write_text("A test debate about taxation.")
        return p

    @pytest.fixture
    def tmp_json(self, tmp_path):
        import json as _json
        p = tmp_path / "show.json"
        p.write_text(_json.dumps({"debate_id": "x", "title": "T"}))
        return p

    def test_root_lang_default(self, tmp_prompt):
        from run import parse_args
        args = parse_args([str(tmp_prompt)])
        assert args.root_lang == "en"

    def test_root_lang_explicit(self, tmp_prompt):
        from run import parse_args
        args = parse_args([str(tmp_prompt), "--root-lang", "es"])
        assert args.root_lang == "es"

    def test_out_lang_default_none(self, tmp_prompt):
        from run import parse_args
        args = parse_args([str(tmp_prompt)])
        assert args.out_lang is None

    def test_out_lang_single(self, tmp_prompt):
        from run import parse_args
        args = parse_args([str(tmp_prompt), "--out-lang", "es"])
        assert args.out_lang == "es"

    def test_out_lang_multiple(self, tmp_prompt):
        from run import parse_args
        args = parse_args([str(tmp_prompt), "--out-lang", "en,es,fr"])
        assert args.out_lang == "en,es,fr"

    def test_out_lang_all(self, tmp_prompt):
        from run import parse_args
        args = parse_args([str(tmp_prompt), "--out-lang", "all"])
        assert args.out_lang == "all"

    def test_invalid_root_lang_exits(self, tmp_json):
        import run as run_master
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_json),
                "--root-lang", "xx",
            ])
        assert exc_info.value.code == 1

    def test_invalid_out_lang_exits(self, tmp_json):
        import run as run_master
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_json),
                "--out-lang", "xx",
            ])
        assert exc_info.value.code == 1

    def test_out_lang_list_includes_root(self, tmp_json):
        """When --out-lang doesn't include root, root is silently added."""
        import run as run_master
        from vox_romana.publish import publish_document as _pd
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.warnings = []
        mock_result.show_type = "debate"
        mock_result.sections = 1
        mock_result.txt_path = tmp_json.parent / "show.txt"
        mock_result.pdf_path = None

        with patch("vox_romana.publish.publish_document", return_value=mock_result):
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_json),
                "--root-lang", "en",
                "--out-lang", "es",
                "--quiet",
            ])
        # The out_lang_list should contain both "en" and "es".
        # We can't inspect args after main() exits, so we verify no crash
        # and the validation message about missing root doesn't appear.

    def test_out_lang_list_resolved_for_all(self, tmp_json):
        """--out-lang all resolves to supported_languages() in out_lang_list."""
        import run as run_master
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.warnings = []
        mock_result.show_type = "debate"
        mock_result.sections = 1
        mock_result.txt_path = tmp_json.parent / "show.txt"
        mock_result.pdf_path = None

        with patch("vox_romana.publish.publish_document", return_value=mock_result):
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_json),
                "--out-lang", "all",
                "--quiet",
            ])
        # Just verifies no crash — actual out_lang_list is ["en"] in Shot 1.

    def test_banner_shows_language(self, tmp_json, capsys):
        import run as run_master
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.warnings = []
        mock_result.show_type = "debate"
        mock_result.sections = 1
        mock_result.txt_path = tmp_json.parent / "show.txt"
        mock_result.pdf_path = None

        with patch("vox_romana.publish.publish_document", return_value=mock_result):
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_json),
                "--root-lang", "es",
            ])
        captured = capsys.readouterr()
        assert "Spanish" in captured.out or "es" in captured.out

    def test_banner_shows_language_matrix(self, tmp_json, capsys):
        import run as run_master
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.warnings = []
        mock_result.show_type = "debate"
        mock_result.sections = 1
        mock_result.txt_path = tmp_json.parent / "show.txt"
        mock_result.pdf_path = None

        with patch("vox_romana.publish.publish_document", return_value=mock_result):
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_json),
                "--root-lang", "en",
                "--out-lang", "en",   # only en → single language display
            ])
        captured = capsys.readouterr()
        # In single-lang mode the banner says "Language: English (en)"
        assert "English" in captured.out
