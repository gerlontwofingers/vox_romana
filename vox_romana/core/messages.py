"""Typed inter-agent messages and routing helpers.

The conductor already owns every persistent AgentSession during a run.
This module adds a small typed protocol on top of those sessions so
downstream agents can exchange structured broadcasts, directives, and
challenge/justification round-trips without coupling the FactChecker or
Editor to any specific show implementation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .session import AgentSession


class MessageType(str, Enum):
    """Kinds of inter-agent messages understood by the router."""

    DIRECTIVE = "directive"
    CHALLENGE = "challenge"
    JUSTIFICATION = "justification"
    BROADCAST = "broadcast"
    ISSUE_REPORT = "issue_report"
    REJECTION = "rejection"


@dataclass
class AgentMessage:
    """A typed message passed between persistent agents in a run."""

    type: MessageType
    from_agent: str
    to_agent: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    reply_to: str | None = None
    message_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.type, MessageType):
            self.type = MessageType(str(self.type))
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def format_for_context(self) -> str:
        """Render this message into a stable text block for session injection."""
        lines = [
            f"MESSAGE TYPE: {self.type.value}",
            f"FROM: {self.from_agent}",
        ]
        if self.to_agent:
            lines.append(f"TO: {self.to_agent}")
        if self.reply_to:
            lines.append(f"IN REPLY TO: {self.reply_to}")
        lines.append("PAYLOAD:")
        if self.payload:
            lines.append(json.dumps(self.payload, indent=2, sort_keys=True, ensure_ascii=True))
        else:
            lines.append("{}")
        return "\n".join(lines)


class AgentRouter:
    """Mixin that routes typed messages across registered agent sessions."""

    _session_registry: dict[str, AgentSession]

    def _log_router_event(
        self,
        *,
        channel: str,
        message: AgentMessage,
        recipients: list[str],
    ) -> None:
        """Record a routed message when the owning conductor supports it."""
        append_log = getattr(self, "_append_production_log", None)
        if not callable(append_log):
            return
        append_log({
            "event_type": "agent_message",
            "channel": channel,
            "recipients": recipients,
            "message": message.to_dict(),
        })

    def _get_registered_session(self, agent_name: str) -> AgentSession:
        """Return the registered session for *agent_name* or raise clearly."""
        try:
            return self._session_registry[agent_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._session_registry)) or "(none)"
            raise KeyError(
                f"No registered session for '{agent_name}'. Available: {available}."
            ) from exc

    def _broadcast(self, message: AgentMessage) -> None:
        """Inject a message into every registered session."""
        if message.to_agent is not None:
            raise ValueError("Broadcast messages must not target a single agent.")
        rendered = message.format_for_context()
        recipients = list(self._session_registry)
        for session in self._session_registry.values():
            session.inject_context(rendered)
        self._log_router_event(
            channel="broadcast",
            message=message,
            recipients=recipients,
        )

    def _route_message(self, message: AgentMessage) -> None:
        """Route a one-way message to a specific session or broadcast it."""
        if message.type == MessageType.CHALLENGE:
            raise ValueError(
                "Challenge messages expect a reply. Use _challenge_loop() instead."
            )
        if message.to_agent is None:
            self._broadcast(message)
            return
        session = self._get_registered_session(message.to_agent)
        session.inject_context(message.format_for_context())
        self._log_router_event(
            channel="direct",
            message=message,
            recipients=[message.to_agent],
        )

    def _challenge_loop(
        self,
        *,
        from_agent: str,
        to_agent: str,
        payload: dict[str, Any],
        reply_to: str | None = None,
    ) -> AgentMessage:
        """Send a challenge that expects a justification response."""
        challenge = AgentMessage(
            type=MessageType.CHALLENGE,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            reply_to=reply_to,
        )
        session = self._get_registered_session(to_agent)
        self._log_router_event(
            channel="challenge",
            message=challenge,
            recipients=[to_agent],
        )
        response = session.send(challenge.format_for_context())
        justification = AgentMessage(
            type=MessageType.JUSTIFICATION,
            from_agent=to_agent,
            to_agent=from_agent,
            payload={"response": response},
            reply_to=challenge.message_id,
        )
        self._log_router_event(
            channel="justification",
            message=justification,
            recipients=[from_agent],
        )
        return justification