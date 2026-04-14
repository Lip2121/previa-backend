from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def _get(obj: Any, key: str, default=None):
    """Works for dicts and Pydantic objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_date(d: Any) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        try:
            return datetime.fromisoformat(d).date()
        except ValueError:
            return None
    return None


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if abs(n) == 1 else plural


def _fmt_currency(x: float, currency: str) -> str:
    sign = "-" if x < 0 else ""
    x_abs = abs(x)
    return f"{sign}{currency}{x_abs:,.0f}"


def generate_decision_impact(
    baseline: Any,
    scenario: Any,
    action_label: str,
    *,
    currency_symbol: str = "€",
    material_days: int = 1,
    material_amount: float = 50.0,
) -> str:
    b_first_neg = _parse_date(_get(baseline, "first_negative_date"))
    s_first_neg = _parse_date(_get(scenario, "first_negative_date"))

    b_days = _get(baseline, "days_until_negative")
    s_days = _get(scenario, "days_until_negative")

    b_low = _get(baseline, "lowest_balance")
    s_low = _get(scenario, "lowest_balance")

    # Normalize numeric fields
    try:
        b_low = float(b_low) if b_low is not None else None
        s_low = float(s_low) if s_low is not None else None
    except (TypeError, ValueError):
        b_low, s_low = None, None

    # Compute deltas
    days_gained = None
    if isinstance(b_days, (int, float)) and isinstance(s_days, (int, float)):
        days_gained = int(round(s_days - b_days))

    worst_change = None
    if b_low is not None and s_low is not None:
        worst_change = s_low - b_low

    negative_removed = (b_first_neg is not None) and (s_first_neg is None)
    still_negative = s_first_neg is not None

    # Precompute words safely
    if days_gained is not None:
        day_word = _plural(days_gained, "day", "days")
        abs_day_word = _plural(abs(days_gained), "day", "days")
    else:
        day_word = "days"
        abs_day_word = "days"

    # Case: removes negative cash within forecast horizon
    if negative_removed:
        if worst_change is not None and abs(worst_change) >= material_amount:
            return (
                f"{action_label} removes negative cash entirely within the forecast period "
                f"and improves worst-case liquidity by {_fmt_currency(worst_change, currency_symbol)}."
            )
        return f"{action_label} removes negative cash entirely within the forecast period."

    # Case: worsens
    if (days_gained is not None and days_gained < -material_days) or (
        worst_change is not None and worst_change < -material_amount
    ):
        parts = [f"{action_label} worsens liquidity"]
        if worst_change is not None:
            parts.append(f"by {_fmt_currency(abs(worst_change), currency_symbol)}")
        if days_gained is not None:
            parts.append(f"and causes cash to turn negative {abs(days_gained)} {abs_day_word} earlier")
        return " ".join(parts) + "."

    # Case: improves (but risk remains)
    if (days_gained is not None and days_gained > material_days) or (
        worst_change is not None and worst_change > material_amount
    ):
        if still_negative and days_gained is not None and days_gained > 0:
            if worst_change is not None and worst_change > 0:
                return (
                    f"{action_label} postpones negative cash by {days_gained} {day_word} "
                    f"and improves worst-case liquidity by {_fmt_currency(worst_change, currency_symbol)}, "
                    f"but does not remove long-term risk."
                )
            return (
                f"{action_label} postpones negative cash by {days_gained} {day_word}, "
                f"but does not remove long-term risk."
            )

        if worst_change is not None and worst_change > 0:
            return (
                f"{action_label} improves worst-case liquidity by {_fmt_currency(worst_change, currency_symbol)}, "
                f"but risk remains within the forecast period."
            )

    return f"{action_label} does not materially change cash risk within the forecast period." 