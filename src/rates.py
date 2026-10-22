"""Rate card: unit rates for the measured BOQ.

All rates are market rates in INR (rupees) covering material + labour.
A market multiplier adapts them to the project location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config, get_config


@dataclass
class RateCard:
    currency_symbol: str = "₹"
    market: str = "india_urban"
    multiplier: float = 1.0
    concrete: dict[str, float] = field(default_factory=dict)
    steel: dict[str, float] = field(default_factory=dict)
    formwork: dict[str, float] = field(default_factory=dict)
    masonry_per_m3: float = 5400.0

    def concrete_rate(self, grade: str) -> float:
        return float(self.concrete.get(grade, 0.0)) * self.multiplier

    def steel_rate(self, grade: str) -> float:
        """Rate per tonne of reinforcement steel."""
        return float(self.steel.get(grade, 0.0)) * self.multiplier

    def formwork_rate(self, kind: str) -> float:
        return float(self.formwork.get(kind, 550.0)) * self.multiplier

    def masonry_rate(self) -> float:
        return self.masonry_per_m3 * self.multiplier

    def format(self, amount: float) -> str:
        return f"{self.currency_symbol}{amount:,.0f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency_symbol": self.currency_symbol,
            "market": self.market,
            "multiplier": self.multiplier,
            "concrete": self.concrete,
            "steel": self.steel,
            "formwork": self.formwork,
        }


def load_rate_card(config: Config | None = None, market: str | None = None) -> RateCard:
    """Build a rate card from the configuration, optionally overriding the market."""
    cfg = config or get_config()
    rates = cfg.section("rates")
    market = market or rates.get("market", "india_urban")
    multipliers = rates.get("market_multipliers", {}) or {}
    mult = float(multipliers.get(market, 1.0))
    return RateCard(
        currency_symbol=rates.get("currency_symbol", "₹"),
        market=market,
        multiplier=mult,
        concrete={k: float(v) for k, v in (rates.get("concrete_per_m3", {}) or {}).items()},
        steel={k: float(v) for k, v in (rates.get("steel_per_tonne", {}) or {}).items()},
        formwork={k: float(v) for k, v in (rates.get("formwork_per_m2", {}) or {}).items()},
        masonry_per_m3=float(rates.get("masonry_per_m3", 5400.0)),
    )


AVAILABLE_MARKETS = ["india_urban", "india_rural", "gcc", "msec", "international"]
