"""Tests for Shot 5 — CLI Orchestration of multi-language pipeline.

Covers:
  * --keep-translated / --translation-model flags parsed onto args.
  * _suffix_audio_output appends _{lang} before the suffix.
  * _mode_translate: writes translated JSON, sets language field, names file.
  * _run_language_matrix:
      - root language uses original JSON
      - non-root languages call _mode_translate, run render/publish
      - cleanup deletes translated JSON unless --keep-translated
      - explicit --audio-output is suffixed per language
      - publish receives root_lang aligned to the per-language pass
  * main() dispatch fans out across languages for publish / render / full.

All tests mock _mode_translate, _mode_render, _mode_publish, and the
DeepSeek client so no real LLM or TTS calls happen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import run as run_mod


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ns(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for matrix/translate helpers."""
    base = dict(
        root_lang="en",
        out_lang_list=["en"],
        keep_translated=False,
        translation_model=None,
        quiet=True,
        audio_output=None,
        publish_format="txt",
        publish_output=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_json(tmp_path: Path, data: dict, name: str = "show.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Argparse: --keep-translated and --translation-model
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageFlags:
    def test_keep_translated_default_false(self):
        args = run_mod.parse_args(
            ["prompts/x.txt", "--mode", "generate"]
        )
        assert args.keep_translated is False

    def test_keep_translated_set_true(self):
        args = run_mod.parse_args(
            ["prompts/x.txt", "--mode", "generate", "--keep-translated"]
        )
        assert args.keep_translated is True

    def test_translation_model_default_none(self):
        args = run_mod.parse_args(
            ["prompts/x.txt", "--mode", "generate"]
        )
        assert args.translation_model is None

    def test_translation_model_explicit(self):
        args = run_mod.parse_args(
            ["prompts/x.txt", "--mode", "generate",
             "--translation-model", "deepseek-reasoner"]
        )
        assert args.translation_model == "deepseek-reasoner"


# ─────────────────────────────────────────────────────────────────────────────
# _suffix_audio_output
# ─────────────────────────────────────────────────────────────────────────────


class TestSuffixAudioOutput:
    def test_none_returns_none(self):
        assert run_mod._suffix_audio_output(None, "es") is None

    def test_simple_suffix_inserted(self):
        out = run_mod._suffix_audio_output(Path("debate.mp3"), "es")
        assert out == Path("debate_es.mp3")

    def test_with_directory(self):
        out = run_mod._suffix_audio_output(
            Path("/tmp/output/debate.wav"), "fr"
        )
        assert out == Path("/tmp/output/debate_fr.wav")

    def test_dotted_stem(self):
        out = run_mod._suffix_audio_output(
            Path("show.v2.mp3"), "de"
        )
        # Path.stem strips only last suffix → "show.v2" + "_de" + ".mp3"
        assert out == Path("show.v2_de.mp3")


# ─────────────────────────────────────────────────────────────────────────────
# _mode_translate
# ─────────────────────────────────────────────────────────────────────────────


class TestModeTranslate:
    def test_writes_translated_json_with_suffix(self, tmp_path):
        src = _write_json(tmp_path, {
            "title": "Hello",
            "language": "en",
            "events": [],
        })
        args = _ns(root_lang="en")

        fake_client = MagicMock()
        fake_translator = MagicMock()

        with patch.object(run_mod, "DeepSeekClient", create=True,
                          return_value=fake_client) if False else \
                patch("vox_romana.core.client.DeepSeekClient",
                      return_value=fake_client), \
             patch("vox_romana.translate.LLMTranslator",
                   return_value=fake_translator), \
             patch("vox_romana.translate.translate_show_data",
                   return_value={"title": "Hola", "language": "es",
                                 "events": []}) as mock_trans:
            out = run_mod._mode_translate(src, "es", args)

        assert out == tmp_path / "show_es.json"
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["language"] == "es"
        assert loaded["title"] == "Hola"
        mock_trans.assert_called_once()
        # source_lang propagated from args.root_lang
        kwargs = mock_trans.call_args.kwargs
        assert kwargs.get("source_lang") == "en"

    def test_passes_translation_model_override(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en", "events": []})
        args = _ns(translation_model="deepseek-reasoner")

        with patch("vox_romana.core.client.DeepSeekClient"), \
             patch("vox_romana.translate.LLMTranslator") as mock_cls, \
             patch("vox_romana.translate.translate_show_data",
                   return_value={"language": "fr", "events": []}):
            run_mod._mode_translate(src, "fr", args)

        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("model") == "deepseek-reasoner"


# ─────────────────────────────────────────────────────────────────────────────
# _run_language_matrix
# ─────────────────────────────────────────────────────────────────────────────


class TestRunLanguageMatrixSingleLang:
    def test_root_only_no_translation(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en"])

        with patch.object(run_mod, "_mode_translate") as mt, \
             patch.object(run_mod, "_mode_render") as mr, \
             patch.object(run_mod, "_mode_publish") as mp:
            paths = run_mod._run_language_matrix(
                src, args, do_render=True, do_publish=False
            )

        mt.assert_not_called()
        mr.assert_called_once()
        mp.assert_not_called()
        assert paths == {"en": src}


class TestRunLanguageMatrixMultiLang:
    def test_translates_each_non_root(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en", "es", "fr"], keep_translated=True)

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate) as mt, \
             patch.object(run_mod, "_mode_render"), \
             patch.object(run_mod, "_mode_publish"):
            paths = run_mod._run_language_matrix(
                src, args, do_render=True, do_publish=True
            )

        assert mt.call_count == 2  # only es and fr, not en
        called_langs = [c.args[1] for c in mt.call_args_list]
        assert called_langs == ["es", "fr"]
        assert paths["en"] == src
        assert paths["es"].name == "show_es.json"
        assert paths["fr"].name == "show_fr.json"

    def test_render_called_per_language(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en", "es"], keep_translated=True)

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate), \
             patch.object(run_mod, "_mode_render") as mr, \
             patch.object(run_mod, "_mode_publish"):
            run_mod._run_language_matrix(
                src, args, do_render=True, do_publish=False
            )

        assert mr.call_count == 2
        rendered_paths = [c.args[0] for c in mr.call_args_list]
        assert rendered_paths[0] == src
        assert rendered_paths[1].name == "show_es.json"

    def test_publish_called_per_language(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en", "es"], keep_translated=True)

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate), \
             patch.object(run_mod, "_mode_render"), \
             patch.object(run_mod, "_mode_publish") as mp:
            run_mod._run_language_matrix(
                src, args, do_render=False, do_publish=True
            )

        assert mp.call_count == 2

    def test_publish_root_lang_aligned_to_pass(self, tmp_path):
        """When publishing the Spanish translation, args.root_lang is
        temporarily set to 'es' so publish_document doesn't double-suffix."""
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en", "es"], keep_translated=True)

        captured_root_langs = []

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        def fake_publish(json_path, a):
            captured_root_langs.append(a.root_lang)

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate), \
             patch.object(run_mod, "_mode_render"), \
             patch.object(run_mod, "_mode_publish",
                          side_effect=fake_publish):
            run_mod._run_language_matrix(
                src, args, do_render=False, do_publish=True
            )

        # First call (root en): root_lang == "en"
        # Second call (es pass): root_lang temporarily == "es"
        assert captured_root_langs == ["en", "es"]
        # Restored after the matrix completes
        assert args.root_lang == "en"

    def test_explicit_audio_output_suffixed_per_language(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        explicit = tmp_path / "myshow.mp3"
        args = _ns(out_lang_list=["en", "es"], keep_translated=True,
                   audio_output=explicit)

        captured_audio = []

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        def fake_render(json_path, a):
            captured_audio.append(a.audio_output)

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate), \
             patch.object(run_mod, "_mode_render",
                          side_effect=fake_render), \
             patch.object(run_mod, "_mode_publish"):
            run_mod._run_language_matrix(
                src, args, do_render=True, do_publish=False
            )

        # Root pass uses original; Spanish pass uses _es suffixed name.
        assert captured_audio[0] == explicit
        assert captured_audio[1] == tmp_path / "myshow_es.mp3"
        # Original restored after the matrix completes.
        assert args.audio_output == explicit

    def test_translated_json_deleted_when_keep_false(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en", "es"], keep_translated=False)

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate), \
             patch.object(run_mod, "_mode_render"), \
             patch.object(run_mod, "_mode_publish"):
            run_mod._run_language_matrix(
                src, args, do_render=True, do_publish=False
            )

        assert not (tmp_path / "show_es.json").exists()
        assert src.exists()  # root never deleted

    def test_translated_json_kept_when_keep_true(self, tmp_path):
        src = _write_json(tmp_path, {"language": "en"})
        args = _ns(out_lang_list=["en", "es"], keep_translated=True)

        def fake_translate(s, lang, a):
            tgt = s.with_name(f"{s.stem}_{lang}.json")
            tgt.write_text("{}")
            return tgt

        with patch.object(run_mod, "_mode_translate",
                          side_effect=fake_translate), \
             patch.object(run_mod, "_mode_render"), \
             patch.object(run_mod, "_mode_publish"):
            run_mod._run_language_matrix(
                src, args, do_render=True, do_publish=False
            )

        assert (tmp_path / "show_es.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# main() dispatch — language matrix integration
# ─────────────────────────────────────────────────────────────────────────────


class TestMainDispatchLanguageMatrix:
    def test_publish_mode_invokes_matrix(self, tmp_path):
        script = _write_json(tmp_path, {"language": "en"})
        argv = [
            "--mode", "publish",
            "--script-input", str(script),
            "--out-lang", "es",
            "--quiet",
        ]
        with patch.object(run_mod, "_run_language_matrix") as mtx:
            run_mod.main(argv)

        mtx.assert_called_once()
        kwargs = mtx.call_args.kwargs
        assert kwargs["do_render"] is False
        assert kwargs["do_publish"] is True

    def test_render_mode_invokes_matrix(self, tmp_path):
        script = _write_json(tmp_path, {"language": "en"})
        argv = [
            "--mode", "render",
            "--script-input", str(script),
            "--out-lang", "es,fr",
            "--quiet",
        ]
        with patch.object(run_mod, "_run_language_matrix") as mtx:
            run_mod.main(argv)

        mtx.assert_called_once()
        passed_args = mtx.call_args.args[1]
        assert set(passed_args.out_lang_list) >= {"en", "es", "fr"}
        assert mtx.call_args.kwargs["do_render"] is True
        assert mtx.call_args.kwargs["do_publish"] is False

    def test_full_mode_invokes_matrix_with_render_only(self, tmp_path):
        prompt = tmp_path / "p.txt"
        prompt.write_text("topic")
        json_out = tmp_path / "out.json"
        json_out.write_text("{}")

        argv = [
            str(prompt),
            "--mode", "full",
            "--out-lang", "es",
            "--quiet",
        ]
        with patch.object(run_mod, "_mode_generate", return_value=json_out), \
             patch.object(run_mod, "_run_language_matrix") as mtx:
            run_mod.main(argv)

        mtx.assert_called_once()
        assert mtx.call_args.kwargs["do_render"] is True
        assert mtx.call_args.kwargs["do_publish"] is False

    def test_full_mode_with_publish_invokes_matrix_render_and_publish(
        self, tmp_path,
    ):
        prompt = tmp_path / "p.txt"
        prompt.write_text("topic")
        json_out = tmp_path / "out.json"
        json_out.write_text("{}")

        argv = [
            str(prompt),
            "--mode", "full",
            "--out-lang", "es",
            "--publish",
            "--quiet",
        ]
        with patch.object(run_mod, "_mode_generate", return_value=json_out), \
             patch.object(run_mod, "_run_language_matrix") as mtx:
            run_mod.main(argv)

        assert mtx.call_args.kwargs["do_render"] is True
        assert mtx.call_args.kwargs["do_publish"] is True

    def test_generate_mode_single_lang_no_matrix(self, tmp_path, capsys):
        prompt = tmp_path / "p.txt"
        prompt.write_text("topic")
        json_out = tmp_path / "out.json"
        json_out.write_text("{}")

        argv = [
            str(prompt),
            "--mode", "generate",
            "--quiet",
        ]
        with patch.object(run_mod, "_mode_generate", return_value=json_out), \
             patch.object(run_mod, "_run_language_matrix") as mtx:
            run_mod.main(argv)

        # No translation needed in single-lang generate mode, no publish.
        mtx.assert_not_called()

    def test_generate_mode_multi_lang_invokes_matrix(self, tmp_path):
        prompt = tmp_path / "p.txt"
        prompt.write_text("topic")
        json_out = tmp_path / "out.json"
        json_out.write_text("{}")

        argv = [
            str(prompt),
            "--mode", "generate",
            "--out-lang", "es",
            "--quiet",
        ]
        with patch.object(run_mod, "_mode_generate", return_value=json_out), \
             patch.object(run_mod, "_run_language_matrix") as mtx:
            run_mod.main(argv)

        mtx.assert_called_once()
        assert mtx.call_args.kwargs["do_render"] is False
        assert mtx.call_args.kwargs["do_publish"] is False

    def test_generate_multi_lang_force_keeps_translated(self, tmp_path):
        """In generate mode, the translated JSONs are the deliverable —
        --keep-translated should be effectively forced True for the matrix
        call when we're not also publishing."""
        prompt = tmp_path / "p.txt"
        prompt.write_text("topic")
        json_out = tmp_path / "out.json"
        json_out.write_text("{}")

        captured_keep = []

        def fake_matrix(src, args, *, do_render, do_publish):
            captured_keep.append(args.keep_translated)

        argv = [
            str(prompt),
            "--mode", "generate",
            "--out-lang", "es",
            "--quiet",
        ]
        with patch.object(run_mod, "_mode_generate", return_value=json_out), \
             patch.object(run_mod, "_run_language_matrix",
                          side_effect=fake_matrix):
            run_mod.main(argv)

        assert captured_keep == [True]
