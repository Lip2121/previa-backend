import numpy as np


def monte_carlo_liquidity_failure(
    hist_flows,
    opening_balance: float,
    horizon_days: int,
    simulations: int = 5000,
    seed: int = 42,
):
    hist = np.array(list(hist_flows), dtype=float)

    if horizon_days <= 0:
        return {
            "probability": 0.0,
            "horizon_days": int(horizon_days),
            "simulations": int(simulations),
            "p10_min_balance": float(opening_balance),
            "p50_min_balance": float(opening_balance),
            "p90_min_balance": float(opening_balance),
        }

    if hist.size < 5:
        return {
            "probability": None,
            "horizon_days": int(horizon_days),
            "simulations": int(simulations),
            "error": "Not enough historical days to run Monte Carlo (need >= 5).",
        }

    mean_flow = float(np.mean(hist))
    std_flow = float(np.std(hist)) * 0.5

    if std_flow == 0:
        balances = np.full((simulations, horizon_days), float(opening_balance + mean_flow))
        balances = np.cumsum(
            np.full((simulations, horizon_days), mean_flow),
            axis=1,
        ) + float(opening_balance)
    else:
        rng = np.random.default_rng(seed)

        simulated_flows = rng.normal(
            loc=mean_flow,
            scale=std_flow,
            size=(simulations, horizon_days),
        )

        lower = np.percentile(hist, 10)
        upper = np.percentile(hist, 90)
        simulated_flows = np.clip(simulated_flows, lower, upper)

        balances = float(opening_balance) + np.cumsum(simulated_flows, axis=1)

    failed = (balances < 0).any(axis=1)
    prob = float(failed.mean())

    path_mins = balances.min(axis=1)
    p10 = float(np.percentile(path_mins, 10))
    p50 = float(np.percentile(path_mins, 50))
    p90 = float(np.percentile(path_mins, 90))

    return {
        "probability": prob,
        "horizon_days": int(horizon_days),
        "simulations": int(simulations),
        "p10_min_balance": p10,
        "p50_min_balance": p50,
        "p90_min_balance": p90,
    }