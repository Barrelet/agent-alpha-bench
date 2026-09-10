"""Decision schema — the contract every agent (rule-based or LLM) must satisfy.

Structural validation lives here (types, ranges, required fields). *Rule*
validation (max positions, one new position per cycle, no averaging down, etc.)
lives in the engine, because it depends on portfolio state and must be applied
identically to every agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

Action = Literal["open_long", "open_short", "add", "close", "hold"]

MIN_CONFIDENCE_TO_OPEN = 0.80
MIN_PCT, MAX_PCT = 10.0, 100.0
MAX_DECISIONS = 5


class DecisionItem(BaseModel):
    symbol: str
    action: Action
    percent_of_equity: float | None = Field(default=None, ge=MIN_PCT, le=MAX_PCT)
    percent_of_position: float | None = Field(default=None, gt=0.0, le=100.0)  # for partial close
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    thesis: str | None = None
    invalidation: str | None = None
    invalidation_price: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _requirements_by_action(self):
        if self.action in ("open_long", "open_short", "add"):
            missing = [f for f in ("percent_of_equity", "confidence", "thesis", "invalidation", "invalidation_price")
                       if getattr(self, f) in (None, "")]
            if missing:
                raise ValueError(f"{self.action} on {self.symbol} missing {missing}")
        return self


class Decision(BaseModel):
    """What an agent returns for one cycle."""
    reasoning: str = ""
    market_context: str = ""
    #: trades to execute this cycle; an EMPTY list means hold. maxItems is enforced
    #: by grammar-constrained backends (Ollama), which stops models enumerating the universe.
    decisions: list[DecisionItem] = Field(default_factory=list, max_length=MAX_DECISIONS)

    @model_validator(mode="after")
    def _at_most_one_hold(self):
        holds = [d for d in self.decisions if d.action == "hold"]
        others = [d for d in self.decisions if d.action != "hold"]
        if holds and others:
            raise ValueError("'hold' cannot be combined with other actions")
        return self

    @classmethod
    def hold(cls, reason: str = "") -> "Decision":
        return cls(reasoning=reason, decisions=[DecisionItem(symbol="*", action="hold")])


def parse_decision(obj: dict | str) -> tuple[Decision | None, str | None]:
    """Validate a dict or JSON string. Returns (decision, error_message)."""
    try:
        d = Decision.model_validate_json(obj) if isinstance(obj, str) else Decision.model_validate(obj)
        return d, None
    except ValidationError as e:
        return None, str(e)


# ---- LLM-facing contract ------------------------------------------------------
# Rule agents use Decision/DecisionItem directly (optional fields, 'hold' allowed).
# LLMs get a STRICT schema: every trade field required, 'close' as its own shape,
# no 'hold' (an empty list is hold), max MAX_DECISIONS items. Grammar-constrained
# backends (Ollama) enforce required keys and enums; pydantic enforces the ranges.

class LLMTradeItem(BaseModel):
    symbol: str
    action: Literal["open_long", "open_short", "add"]
    percent_of_equity: float = Field(ge=MIN_PCT, le=MAX_PCT, description="PERCENT of equity, 10-100 (20 means 20%)")
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    invalidation: str
    invalidation_price: float = Field(gt=0.0)


class LLMCloseItem(BaseModel):
    symbol: str
    action: Literal["close"]


class LLMDecision(BaseModel):
    reasoning: str
    market_context: str
    decisions: list[LLMTradeItem | LLMCloseItem] = Field(max_length=MAX_DECISIONS)

    def to_decision(self) -> Decision:
        return Decision(reasoning=self.reasoning, market_context=self.market_context,
                        decisions=[DecisionItem(**d.model_dump()) for d in self.decisions])


def parse_llm_decision(obj: dict | str) -> tuple[Decision | None, str | None]:
    """Validate an LLM response against the strict contract; return a Decision."""
    try:
        d = LLMDecision.model_validate_json(obj) if isinstance(obj, str) else LLMDecision.model_validate(obj)
        return d.to_decision(), None
    except ValidationError as e:
        return None, str(e)


def json_schema(for_llm: bool = False) -> dict:
    """JSON Schema of the decision contract: the permissive Decision for rule
    agents, or the strict LLMDecision when for_llm=True."""
    return LLMDecision.model_json_schema() if for_llm else Decision.model_json_schema()
