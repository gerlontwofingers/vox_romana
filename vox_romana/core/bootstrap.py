"""Common base for show-type bootstrap orchestrators.

Each show type (debates, news, podcast, audiobooks) has its own
``XxxBootstrap`` class that turns a natural-language brief into a fully
populated ``ContentConfig`` subclass by chaining LLM agents
(Architect → Profile Builder → Director). The orchestration steps
differ per show type, but two pieces of plumbing are universal:

  * Constructing the shared :class:`DeepSeekClient` once and reusing it
    across every agent the bootstrap creates.
  * Applying user-supplied ``overrides`` to the resulting config after
    the agents have run, silently ignoring keys that don't exist.

``BaseShowBootstrap`` captures both pieces. Show-specific bootstraps
inherit and add their own ``from_brief`` that calls the agents.
"""

from __future__ import annotations

from .client import DeepSeekClient
from .config import ContentConfig


class BaseShowBootstrap:
    """Shared init + override-application logic for all show bootstraps.

    Subclasses construct their own LLM agents from ``self._client`` and
    expose a ``from_brief()`` method that builds and returns a populated
    config. They should call :meth:`_apply_overrides` at the end of
    ``from_brief()`` so the user gets the last word.
    """

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        """Initialize the bootstrap with a shared LLM client.

        Args:
            client:  An existing DeepSeekClient to use. If not provided,
                     creates a new one using ``api_key`` or the
                     ``DEEPSEEK_API_KEY`` env var.
            api_key: API key for DeepSeekClient. Only used if ``client``
                     is None.
        """
        self._client = client or DeepSeekClient(api_key=api_key)

    @staticmethod
    def _apply_overrides(config: ContentConfig, overrides: dict | None) -> None:
        """Apply user overrides to ``config`` in place.

        Only sets attributes that actually exist on the config — unknown
        keys are silently skipped. They might be for future fields, and
        complaining about them would be annoying for callers who reuse
        an override dict across config versions.

        Args:
            config:    The config to modify in place.
            overrides: Dict of ``{field_name: value}``. ``None`` is a no-op.
        """
        if not overrides:
            return
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
