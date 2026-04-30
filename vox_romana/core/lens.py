"""LLM Lens Rack — per-role / per-character LLM backend selection.

The "lens rack" is the central abstraction for picking *which* LLM is used
for *which* job in vox_romana.  It treats local and cloud models as a set
of named **lenses**, each with its own personality and strengths.  A
weaker or weirder model is not a fallback — it is the right voice for
some characters (cf. Salvatore the polyglot lay brother in *The Name of
the Rose* who mangles Latin / Italian / Provençal across every sentence).

KEY CONCEPTS
────────────

* :class:`LLMLens` — a *named* :class:`~vox_romana.core.client.LLMClient`
  with creative metadata: ``personality_hint``, ``cost_tier``,
  ``language_strengths``.  The metadata is consumed by the casting
  tools (``scripts/lens_character_probe.py`` in Chunk 2) and the MCP
  ``vox_list_lenses`` tool (Chunk 3); it does not affect inference.

* :class:`LLMLensRack` — a registry of lenses plus a routing table that
  maps **roles** to lens names.  Roles are dotted strings:

  ===========================  =====================================
  Role                         Used by
  ===========================  =====================================
  ``"default"``                Fallback for everything not overridden
  ``"director"``               AnimationDirector / show director
  ``"conductor"``              Per-turn dialogue generation
  ``"editor"``                 Editor passes
  ``"fact_checker"``           FactChecker
  ``"translator"``             LLMTranslator
  ``"character:<Name>"``       A specific character's voice — the
                               headline use-case for the rack
  ===========================  =====================================

  :meth:`LLMLensRack.select` looks up the most specific role first
  (``character:Salvatore``), then progressively less specific ones
  (``character``, the bare role like ``conductor``), and finally
  ``default``.

DEFAULT BEHAVIOUR
─────────────────

A lens rack built with :func:`build_default_rack` registers four named
lenses but only the cloud lens is *constructed* eagerly.  The local
lenses are constructed lazily — see :meth:`LLMLens.client` — so a user
who never opts into local inference does not need Ollama installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from vox_romana.core.client import DeepSeekCloudClient, LLMClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lens dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMLens:
    """A named LLM client plus creative metadata.

    The ``factory`` callable is invoked the first time :attr:`client` is
    accessed, allowing local-model lenses to be declared without
    triggering any network or ``ollama`` import at startup.

    Attributes:
        name:                Unique lens id (e.g. ``"local-llama-dim"``).
        backend:             ``"cloud"`` or ``"local"``.
        model:               Underlying model identifier (e.g.
                             ``"deepseek-chat"``, ``"llama3.2:3b"``).
        personality_hint:    One-line description of the lens's voice;
                             read by casting tools and humans.
        cost_tier:           ``"free"`` (local) or ``"paid"`` (cloud).
        language_strengths:  BCP-47 codes the lens handles well.
        factory:             Zero-arg callable that produces the
                             underlying :class:`LLMClient`.  Called
                             once and cached on :attr:`_client`.
    """

    name: str
    backend: str
    model: str
    personality_hint: str = ""
    cost_tier: str = "paid"
    language_strengths: tuple[str, ...] = ("en",)
    factory: Callable[[], LLMClient] | None = None
    _client: LLMClient | None = field(default=None, repr=False, compare=False)

    @property
    def client(self) -> LLMClient:
        """Return (and lazily construct) the underlying LLM client."""
        if self._client is None:
            if self.factory is None:
                raise RuntimeError(
                    f"Lens '{self.name}' has no factory and no pre-built client."
                )
            self._client = self.factory()
        return self._client

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the MCP ``vox_list_lenses`` tool / introspection."""
        return {
            "name": self.name,
            "backend": self.backend,
            "model": self.model,
            "personality_hint": self.personality_hint,
            "cost_tier": self.cost_tier,
            "language_strengths": list(self.language_strengths),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Built-in lens catalogue
# ─────────────────────────────────────────────────────────────────────────────
#
# Local lens definitions are described declaratively here so they can be
# enumerated by ``vox_list_lenses`` even when Ollama is not installed.
# The factory is only called when a caller actually selects a local
# lens, at which point ``vox_romana.local.ollama`` is imported.

CLOUD_DEEPSEEK = "cloud-deepseek"
LOCAL_R1_DISTILL = "local-r1-distill"
LOCAL_QWEN_POLYGLOT = "local-qwen-polyglot"
LOCAL_LLAMA_DIM = "local-llama-dim"


def _make_cloud_deepseek_factory() -> Callable[[], LLMClient]:
    def factory() -> LLMClient:
        return DeepSeekCloudClient()
    return factory


def _make_ollama_factory(model: str) -> Callable[[], LLMClient]:
    def factory() -> LLMClient:
        # Imported lazily so users who never opt into local inference
        # don't pay any import cost.
        from vox_romana.local.ollama import OllamaClient
        return OllamaClient(model=model)
    return factory


def _builtin_lens_specs() -> list[LLMLens]:
    """The four lenses that ship with vox_romana out of the box.

    Returned as a fresh list every call so tests can mutate without
    bleeding state across cases.
    """
    return [
        LLMLens(
            name=CLOUD_DEEPSEEK,
            backend="cloud",
            model="deepseek-chat",
            personality_hint=(
                "Sober, capable generalist. The default voice for every "
                "role unless explicitly recast."
            ),
            cost_tier="paid",
            language_strengths=("en", "zh", "es", "fr", "de"),
            factory=_make_cloud_deepseek_factory(),
        ),
        LLMLens(
            name=LOCAL_R1_DISTILL,
            backend="local",
            model="deepseek-r1-distill-qwen-7b",
            personality_hint=(
                "Sober reasoner; structured-JSON friendly. Closest local "
                "match to the cloud DeepSeek family."
            ),
            cost_tier="free",
            language_strengths=("en", "zh"),
            factory=_make_ollama_factory("deepseek-r1-distill-qwen-7b"),
        ),
        LLMLens(
            name=LOCAL_QWEN_POLYGLOT,
            backend="local",
            model="qwen2.5:7b-instruct",
            personality_hint=(
                "Strong multilingual; codeswitches well. Good for "
                "characters who slide between languages."
            ),
            cost_tier="free",
            language_strengths=("en", "zh", "es", "fr", "de", "ja", "ru", "ar"),
            factory=_make_ollama_factory("qwen2.5:7b-instruct"),
        ),
        LLMLens(
            name=LOCAL_LLAMA_DIM,
            backend="local",
            model="llama3.2:3b-instruct",
            personality_hint=(
                "Small, brittle, sometimes naive. Use for innocents, "
                "drunks, code-switching monks — the Salvatore lens."
            ),
            cost_tier="free",
            language_strengths=("en",),
            factory=_make_ollama_factory("llama3.2:3b-instruct"),
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Lens rack
# ─────────────────────────────────────────────────────────────────────────────

class LLMLensRack:
    """Registry of named lenses plus a role → lens routing table.

    Routing precedence in :meth:`select` (most specific first):

    1. ``character:<exact-name>``  (case-insensitive match)
    2. The bare ``role`` argument (e.g. ``"conductor"``)
    3. ``"default"``

    The default rack always has a ``"default"`` entry so :meth:`select`
    is total — it never returns ``None``.
    """

    def __init__(self, default_lens_name: str = CLOUD_DEEPSEEK) -> None:
        self._lenses: dict[str, LLMLens] = {}
        self._routes: dict[str, str] = {}
        self._default = default_lens_name

    # ── Registration ────────────────────────────────────────────────

    def register(self, lens: LLMLens) -> None:
        """Add a lens to the rack.  Replaces any prior lens of the same name."""
        self._lenses[lens.name] = lens

    def assign(self, role: str, lens_name: str) -> None:
        """Bind ``role`` (e.g. ``"character:Salvatore"``) to a registered lens."""
        if lens_name not in self._lenses:
            raise KeyError(
                f"Cannot assign role '{role}' → unknown lens '{lens_name}'. "
                f"Known lenses: {sorted(self._lenses)}"
            )
        self._routes[role.lower()] = lens_name

    def set_default(self, lens_name: str) -> None:
        """Change the rack-wide default lens."""
        if lens_name not in self._lenses:
            raise KeyError(f"Unknown lens '{lens_name}'.")
        self._default = lens_name

    # ── Lookup ──────────────────────────────────────────────────────

    def select(
        self,
        role: str | None = None,
        character: str | None = None,
    ) -> LLMLens:
        """Resolve a ``(role, character)`` pair to a concrete lens.

        Args:
            role:      Coarse-grained role (e.g. ``"conductor"``,
                       ``"director"``, ``"translator"``).  ``None`` is
                       treated as the rack default.
            character: Fine-grained speaker name (e.g. ``"Salvatore"``).
                       Wins over ``role`` when a route exists for it.

        Returns:
            The matching :class:`LLMLens`.  Falls back to the rack
            default — which is always present — if no rule matches.
        """
        # 1. character:<name> match (case-insensitive)
        if character:
            key = f"character:{character}".lower()
            if key in self._routes:
                return self._lenses[self._routes[key]]
            # 2. bare 'character' key (catch-all for any character)
            if "character" in self._routes:
                return self._lenses[self._routes["character"]]

        # 3. role match
        if role and role.lower() in self._routes:
            return self._lenses[self._routes[role.lower()]]

        # 4. default
        return self._lenses[self._default]

    def list_lenses(self) -> list[dict[str, Any]]:
        """Return all registered lenses as dicts (for MCP / introspection)."""
        return [lens.to_dict() for lens in self._lenses.values()]

    def routes(self) -> dict[str, str]:
        """Return a copy of the role → lens-name routing table."""
        return dict(self._routes)

    @property
    def default_lens_name(self) -> str:
        return self._default

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        backend_override: str | None = None,
        lens_overrides: dict[str, str] | None = None,
    ) -> "LLMLensRack":
        """Build a rack from a parsed ``llm:`` block.

        Args:
            config:           The ``llm`` sub-dict from
                              :mod:`vox_romana.core.llm_config`, or
                              ``None`` for all defaults.
            backend_override: Override from ``--llm-backend``.  Values:
                              ``"cloud"`` (force cloud-deepseek as the
                              default), ``"local"`` (force
                              local-r1-distill), ``"auto"`` / ``None``
                              (honour the config).
            lens_overrides:   Per-role overrides from ``--llm-lens``,
                              e.g. ``{"character:Salvatore": "local-llama-dim"}``.

        Returns:
            A fully populated rack.  All four built-in lenses are
            registered; a config may add more.
        """
        rack = cls()

        # Always seed the four built-in lenses.
        for lens in _builtin_lens_specs():
            rack.register(lens)

        cfg = dict(config or {})

        # Optional: extra lenses defined inline in YAML.
        for spec in cfg.get("lenses") or []:
            rack.register(_lens_from_yaml_spec(spec))

        # Default lens.
        default = cfg.get("default") or CLOUD_DEEPSEEK
        if default not in rack._lenses:
            logger.warning(
                "llm.default '%s' is not registered; falling back to '%s'.",
                default, CLOUD_DEEPSEEK,
            )
            default = CLOUD_DEEPSEEK
        rack.set_default(default)

        # Role bindings from YAML.
        for role, lens_name in (cfg.get("roles") or {}).items():
            try:
                rack.assign(role, lens_name)
            except KeyError as exc:
                logger.warning("Skipping bad llm.roles entry: %s", exc)

        # Apply --llm-backend override.
        if backend_override == "cloud":
            rack.set_default(CLOUD_DEEPSEEK)
        elif backend_override == "local":
            rack.set_default(LOCAL_R1_DISTILL)
        # "auto" / None → honour config

        # Apply --llm-lens overrides last so they win over everything.
        for role, lens_name in (lens_overrides or {}).items():
            try:
                rack.assign(role, lens_name)
            except KeyError as exc:
                logger.warning("Skipping bad --llm-lens override: %s", exc)

        return rack


def _lens_from_yaml_spec(spec: dict[str, Any]) -> LLMLens:
    """Build an :class:`LLMLens` from an inline YAML lens definition.

    Only Ollama-backed lenses can be added this way at the moment; cloud
    lenses must be one of the built-ins.
    """
    name = spec["name"]
    backend = spec.get("backend", "local")
    model = spec.get("model") or name
    if backend == "cloud":
        factory: Callable[[], LLMClient] = _make_cloud_deepseek_factory()
    else:
        factory = _make_ollama_factory(model)
    return LLMLens(
        name=name,
        backend=backend,
        model=model,
        personality_hint=spec.get("personality_hint", ""),
        cost_tier=spec.get("cost_tier", "free" if backend == "local" else "paid"),
        language_strengths=tuple(spec.get("language_strengths") or ("en",)),
        factory=factory,
    )


def build_default_rack() -> LLMLensRack:
    """Convenience helper: rack with built-in lenses + cloud default."""
    return LLMLensRack.from_config(None)
