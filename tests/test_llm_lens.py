"""Tests for the LLM lens rack (Chunk 1)."""

from __future__ import annotations

import pytest

from vox_romana.core.client import DeepSeekClient, DeepSeekCloudClient, LLMClient
from vox_romana.core.lens import (
    CLOUD_DEEPSEEK,
    LOCAL_LLAMA_DIM,
    LOCAL_QWEN_POLYGLOT,
    LOCAL_R1_DISTILL,
    LLMLens,
    LLMLensRack,
    build_default_rack,
)
from vox_romana.core.llm_config import parse_lens_overrides


# ─── Protocol conformance ───────────────────────────────────────────────────

def test_deepseek_alias_points_to_same_class():
    assert DeepSeekCloudClient is DeepSeekClient


def test_deepseek_client_satisfies_protocol(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    c = DeepSeekClient()
    # runtime_checkable Protocol should accept the client
    assert isinstance(c, LLMClient)
    assert c.model == "deepseek-chat"


# ─── LLMLens ─────────────────────────────────────────────────────────────────

def test_lens_lazy_factory_called_once():
    calls = {"n": 0}

    class FakeClient:
        model = "fake"
        def chat(self, messages, **kw):
            return ""

    def factory():
        calls["n"] += 1
        return FakeClient()

    lens = LLMLens(name="x", backend="local", model="fake", factory=factory)
    a = lens.client
    b = lens.client
    assert a is b
    assert calls["n"] == 1


def test_lens_to_dict_contains_metadata():
    lens = LLMLens(
        name="local-llama-dim",
        backend="local",
        model="llama3.2:3b-instruct",
        personality_hint="naive",
        cost_tier="free",
        language_strengths=("en",),
    )
    d = lens.to_dict()
    assert d["name"] == "local-llama-dim"
    assert d["backend"] == "local"
    assert d["personality_hint"] == "naive"
    assert d["language_strengths"] == ["en"]


def test_lens_without_factory_raises():
    lens = LLMLens(name="bare", backend="local", model="x")
    with pytest.raises(RuntimeError):
        _ = lens.client


# ─── LLMLensRack routing precedence ─────────────────────────────────────────

def _make_rack():
    rack = build_default_rack()
    return rack


def test_default_rack_has_four_builtins():
    rack = _make_rack()
    names = {l["name"] for l in rack.list_lenses()}
    assert {CLOUD_DEEPSEEK, LOCAL_R1_DISTILL,
            LOCAL_QWEN_POLYGLOT, LOCAL_LLAMA_DIM} <= names


def test_select_falls_back_to_default():
    rack = _make_rack()
    assert rack.select().name == CLOUD_DEEPSEEK
    assert rack.select(role="conductor").name == CLOUD_DEEPSEEK


def test_select_respects_role_assignment():
    rack = _make_rack()
    rack.assign("translator", LOCAL_QWEN_POLYGLOT)
    assert rack.select(role="translator").name == LOCAL_QWEN_POLYGLOT
    assert rack.select(role="conductor").name == CLOUD_DEEPSEEK


def test_character_route_wins_over_role():
    rack = _make_rack()
    rack.assign("conductor", LOCAL_R1_DISTILL)
    rack.assign("character:Salvatore", LOCAL_LLAMA_DIM)
    sel = rack.select(role="conductor", character="Salvatore")
    assert sel.name == LOCAL_LLAMA_DIM


def test_character_match_is_case_insensitive():
    rack = _make_rack()
    rack.assign("character:Salvatore", LOCAL_LLAMA_DIM)
    assert rack.select(character="salvatore").name == LOCAL_LLAMA_DIM
    assert rack.select(character="SALVATORE").name == LOCAL_LLAMA_DIM


def test_assign_unknown_lens_raises():
    rack = _make_rack()
    with pytest.raises(KeyError):
        rack.assign("conductor", "no-such-lens")


# ─── from_config ────────────────────────────────────────────────────────────

def test_from_config_with_none_yields_cloud_default():
    rack = LLMLensRack.from_config(None)
    assert rack.default_lens_name == CLOUD_DEEPSEEK


def test_from_config_default_and_roles():
    cfg = {
        "default": CLOUD_DEEPSEEK,
        "roles": {
            "translator": LOCAL_QWEN_POLYGLOT,
            "character:Salvatore": LOCAL_LLAMA_DIM,
        },
    }
    rack = LLMLensRack.from_config(cfg)
    assert rack.select(role="translator").name == LOCAL_QWEN_POLYGLOT
    assert rack.select(character="Salvatore").name == LOCAL_LLAMA_DIM


def test_from_config_unknown_default_warns_and_falls_back():
    cfg = {"default": "no-such-lens"}
    rack = LLMLensRack.from_config(cfg)
    assert rack.default_lens_name == CLOUD_DEEPSEEK


def test_from_config_skips_bad_role_silently():
    cfg = {"roles": {"conductor": "no-such-lens"}}
    rack = LLMLensRack.from_config(cfg)
    # Should still build, just without that route.
    assert "conductor" not in rack.routes()


def test_backend_override_cloud():
    rack = LLMLensRack.from_config(
        {"default": LOCAL_R1_DISTILL},
        backend_override="cloud",
    )
    assert rack.default_lens_name == CLOUD_DEEPSEEK


def test_backend_override_local():
    rack = LLMLensRack.from_config(None, backend_override="local")
    assert rack.default_lens_name == LOCAL_R1_DISTILL


def test_lens_overrides_win_over_config():
    cfg = {"roles": {"translator": CLOUD_DEEPSEEK}}
    rack = LLMLensRack.from_config(
        cfg,
        lens_overrides={"translator": LOCAL_QWEN_POLYGLOT},
    )
    assert rack.select(role="translator").name == LOCAL_QWEN_POLYGLOT


def test_inline_lens_definition():
    cfg = {
        "lenses": [
            {
                "name": "local-mistral-small",
                "backend": "local",
                "model": "mistral-small:latest",
                "personality_hint": "Terse French",
                "language_strengths": ["en", "fr"],
            }
        ],
        "roles": {"translator": "local-mistral-small"},
    }
    rack = LLMLensRack.from_config(cfg)
    sel = rack.select(role="translator")
    assert sel.name == "local-mistral-small"
    assert sel.model == "mistral-small:latest"


# ─── parse_lens_overrides ───────────────────────────────────────────────────

def test_parse_lens_overrides_basic():
    out = parse_lens_overrides([
        "translator=local-qwen-polyglot",
        "character:Salvatore=local-llama-dim",
    ])
    assert out == {
        "translator": "local-qwen-polyglot",
        "character:Salvatore": "local-llama-dim",
    }


def test_parse_lens_overrides_empty():
    assert parse_lens_overrides(None) == {}
    assert parse_lens_overrides([]) == {}


@pytest.mark.parametrize("bad", ["no-equals", "=lens", "role=", "  =  "])
def test_parse_lens_overrides_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_lens_overrides([bad])
