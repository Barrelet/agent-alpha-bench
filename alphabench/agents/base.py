"""Agent interface. Every competitor — rule-based control or LLM — implements
`decide(payload) -> Decision`. The engine treats them identically."""

from __future__ import annotations

from ..schema import Decision


class Agent:
    name: str = "agent"
    #: symbols this agent may trade. None = the standard universe. Controls that
    #: hold the benchmark override this; LLM agents never may.
    tradeable: set[str] | None = None
    #: False for agents that never read candles (random / rule controls that only use
    #: the summary table) — lets the replay skip building candle blocks for held names.
    needs_detail: bool = True

    def decide(self, payload: dict) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        """Called once before a replay so stateful agents start clean."""
