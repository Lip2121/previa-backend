import numpy as np


def calculate_liquidity_risk(forecast):
    if not forecast:
        return {
            "score": 0,
            "level": "Low",
            "lowest_balance": 0.0,
            "first_negative_day": None
        }

    balances = [f["balance"] for f in forecast]
    lowest_balance = min(balances)

    negative_days = [i for i, b in enumerate(balances) if b < 0]
    first_negative_day = negative_days[0] if negative_days else None

    volatility = float(np.std(balances))

    score = 0

    # --- Days until negative ---
    if first_negative_day is not None:
        if first_negative_day < 7:
            score += 40
        elif first_negative_day < 30:
            score += 25
        elif first_negative_day < 60:
            score += 10

    # --- Lowest balance severity ---
    if lowest_balance < -50000:
        score += 30
    elif lowest_balance < -10000:
        score += 20
    elif lowest_balance < 0:
        score += 10

    # --- Volatility penalty ---
    if volatility > 50000:
        score += 20
    elif volatility > 20000:
        score += 10

    score = min(score, 100)

    if score < 30:
        level = "Low"
    elif score < 60:
        level = "Medium"
    elif score < 80:
        level = "High"
    else:
        level = "Critical"

    return {
        "score": score,
        "level": level,
        "lowest_balance": float(lowest_balance),
        "first_negative_day": first_negative_day
    }