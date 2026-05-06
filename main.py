from services.liquidity_failure import monte_carlo_liquidity_failure
from fastapi import FastAPI, UploadFile, File, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from io import StringIO
import pandas as pd
import os

from apscheduler.schedulers.background import BackgroundScheduler
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from pydantic import BaseModel


def parse_uploaded_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig", errors="ignore")
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file. Make sure the file is a valid CSV. Error: {str(e)}",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV is empty. Add at least one row with date and amount values.",
        )

    return df


def normalize_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean == "date":
            rename_map[col] = "date"
        elif col_clean == "amount":
            rename_map[col] = "amount"
        elif col_clean == "customer":
            rename_map[col] = "customer"
    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame):
    required = {"date", "amount"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column(s): {', '.join(missing)}. Your CSV must include at minimum 'date' and 'amount'.",
        )


def format_action_label_from_type(scenario_type: str | None) -> str:
    if scenario_type == "delay_outflows":
        return "Delaying outflows"
    if scenario_type == "accelerate_inflows":
        return "Accelerating inflows"
    if scenario_type == "timing_adjustment":
        return "Adjusting cash timing"
    if scenario_type == "cash_injection":
        return "Injecting cash"
    return "Scenario action"


def detect_action_label(forecast: dict) -> str:
    scenario = forecast.get("scenario")
    if not scenario:
        return "Scenario"
    return format_action_label_from_type(scenario.get("scenario_type"))


def _lowest_balance_from_date(series: list[dict], start_date: str) -> float | None:
    if not series:
        return None

    start_ts = pd.to_datetime(start_date)
    balances = []

    for row in series:
        row_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(row_date):
            continue
        if row_date >= start_ts:
            balances.append(float(row.get("balance", 0) or 0))

    if not balances:
        return None

    return float(min(balances))


def _lowest_balance_date_from_date(series: list[dict], start_date: str) -> str | None:
    if not series:
        return None

    start_ts = pd.to_datetime(start_date)
    filtered = []

    for row in series:
        row_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(row_date):
            continue
        if row_date >= start_ts:
            filtered.append(row)

    if not filtered:
        return None

    lowest_row = min(filtered, key=lambda r: float(r.get("balance", 0) or 0))
    return lowest_row.get("date")


def compute_balance_improvement_from_comparison_window(
    baseline: dict,
    scenario: dict,
) -> tuple[float, float, float, str | None]:
    historical_end = baseline.get("historical_end_date") or scenario.get("historical_end_date")
    compare_start = None

    if historical_end:
        compare_start = (pd.to_datetime(historical_end) + pd.Timedelta(days=1)).date().isoformat()

    if compare_start:
        baseline_lowest = _lowest_balance_from_date(baseline.get("series", []), compare_start)
        scenario_lowest = _lowest_balance_from_date(scenario.get("series", []), compare_start)
        scenario_lowest_date = _lowest_balance_date_from_date(scenario.get("series", []), compare_start)
    else:
        baseline_lowest = None
        scenario_lowest = None
        scenario_lowest_date = None

    if baseline_lowest is None:
        baseline_lowest = float(baseline.get("lowest_balance", 0) or 0)

    if scenario_lowest is None:
        scenario_lowest = float(scenario.get("lowest_balance", 0) or 0)

    if scenario_lowest_date is None:
        scenario_lowest_date = scenario.get("lowest_balance_date")

    improvement = float(scenario_lowest - baseline_lowest)

    return baseline_lowest, scenario_lowest, improvement, scenario_lowest_date


def generate_decision_impact(baseline: dict, scenario: dict, action_label: str, currency_symbol: str) -> dict:
    baseline_negative = baseline.get("first_negative_date")
    scenario_negative = scenario.get("first_negative_date")

    days_gained = None
    if baseline_negative and scenario_negative:
        baseline_date = pd.to_datetime(baseline_negative)
        scenario_date = pd.to_datetime(scenario_negative)
        days_gained = int((scenario_date - baseline_date).days)

    baseline_lowest_balance, scenario_lowest_balance, balance_improvement, scenario_lowest_date = (
        compute_balance_improvement_from_comparison_window(baseline, scenario)
    )

    return {
        "action": action_label,
        "baseline_first_negative": baseline_negative,
        "scenario_first_negative": scenario_negative,
        "days_gained": days_gained,
        "baseline_lowest_balance": baseline_lowest_balance,
        "scenario_lowest_balance": scenario_lowest_balance,
        "scenario_lowest_balance_date": scenario_lowest_date,
        "balance_improvement": balance_improvement,
        "currency_symbol": currency_symbol,
    }


def build_executive_summary(forecast: dict):
    days = forecast.get("days_until_negative")
    first_neg = forecast.get("first_negative_date")
    lowest = forecast.get("lowest_balance")
    lowest_date = forecast.get("lowest_balance_date")

    below = forecast.get("below_threshold_at_start", False)
    warning_threshold = float(forecast.get("warning_threshold", 0.0) or 0.0)

    risk = forecast.get("risk", {}) or {}
    risk_level = risk.get("level")
    risk_score = risk.get("score")

    failure = forecast.get("liquidity_failure", {}) or {}
    failure_probability = failure.get("probability")
    if failure_probability is not None:
        failure_probability = round(float(failure_probability) * 100)

    if below and warning_threshold and first_neg is not None:
        return {
            "headline": "Cash is already below the warning threshold",
            "details": (
                f"Your cash balance starts below the warning threshold ({warning_threshold}). "
                f"Cash is projected to turn negative on {first_neg}."
            ),
            "suggestions": [
                "Review near-term outflows immediately",
                "Accelerate incoming payments where possible",
                "Consider short-term financing",
            ],
        }

    if first_neg is not None:
        return {
            "headline": f"Cash turns negative in {days} day{'s' if days != 1 else ''}",
            "details": (
                f"Your cash balance is projected to turn negative on {first_neg}. "
                f"The lowest expected balance is {lowest}. Without intervention, liquidity risk becomes immediate."
            ),
            "suggestions": [
                f"Review and potentially delay major outflows before {first_neg}",
                "Accelerate incoming payments where possible",
                "Consider securing short-term financing",
            ],
        }

    if risk_level == "Watch" or (failure_probability is not None and failure_probability >= 30):
        return {
            "headline": "Liquidity remains positive, but downside risk is building",
            "details": (
                f"Cash does not turn negative within the forecast horizon, but the risk profile remains elevated. "
                f"The lowest projected balance is {lowest}"
                f"{f' on {lowest_date}' if lowest_date else ''}. "
                f"Liquidity risk is currently {risk_level or 'elevated'}"
                f"{f' ({risk_score})' if risk_score is not None else ''}"
                f"{f', with an estimated {failure_probability}% failure probability in simulation.' if failure_probability is not None else '.'}"
            ),
            "suggestions": [
                "Test protective scenarios before major payment periods",
                "Review large planned outflows and timing sensitivity",
                "Monitor liquidity closely over the next cycle",
            ],
        }

    return {
        "headline": "No immediate liquidity risk detected",
        "details": "Based on the current data, your cash balance does not turn negative within the forecast period.",
        "suggestions": [],
    }


def get_risk_level(score):
    if score >= 80:
        return "Safe"
    elif score >= 50:
        return "Watch"
    elif score >= 20:
        return "High"
    return "Critical"


def compute_liquidity_risk(forecast):
    lowest_balance = float(forecast.get("lowest_balance", 0) or 0)
    days_until_negative = forecast.get("days_until_negative")
    warning_threshold = float(forecast.get("warning_threshold", 0) or 0)

    if days_until_negative is None:
        time_score = 100
    else:
        time_score = max(0, min(100, days_until_negative * 4))

    if lowest_balance <= 0:
        buffer_score = 0
    elif lowest_balance >= 5000:
        buffer_score = 100
    else:
        buffer_score = (lowest_balance / 5000) * 100

    if warning_threshold > 0:
        if lowest_balance <= warning_threshold:
            threshold_score = 0
        elif lowest_balance >= warning_threshold + 5000:
            threshold_score = 100
        else:
            threshold_score = ((lowest_balance - warning_threshold) / 5000) * 100
    else:
        threshold_score = buffer_score

    score = 0.45 * time_score + 0.30 * buffer_score + 0.25 * threshold_score

    return {
        "score": round(score),
        "level": get_risk_level(score),
    }


def identify_risk_drivers(forecast: dict, decision_impact: dict | None = None) -> dict:
    series = forecast.get("series", []) or []
    warning_threshold = float(forecast.get("warning_threshold", 0) or 0)
    lowest_balance = float(forecast.get("lowest_balance", 0) or 0)
    liquidity_failure = forecast.get("liquidity_failure", {}) or {}
    failure_probability = liquidity_failure.get("probability")
    scenario_type = forecast.get("scenario_type")

    if failure_probability is not None:
        failure_probability = float(failure_probability)

    drivers = []
    explanation_parts = []
    recommended_focus = []
    primary_driver = None

    if not series:
        return {
            "primary_driver": "insufficient_data",
            "drivers": ["Insufficient forecast series data"],
            "explanation": "Risk drivers could not be determined because no forecast series was available.",
            "recommended_focus": ["Check uploaded data and rerun the forecast"],
        }

    df = pd.DataFrame(series).copy()
    df["net_flow"] = pd.to_numeric(df["net_flow"], errors="coerce").fillna(0.0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    thin_buffer = False
    if lowest_balance > 0:
        if warning_threshold > 0 and lowest_balance <= warning_threshold + 1000:
            thin_buffer = True
        elif warning_threshold == 0 and lowest_balance <= 1000:
            thin_buffer = True

    if thin_buffer:
        drivers.append("Thin projected cash buffer")
        explanation_parts.append(
            f"The projected minimum cash balance falls to {round(lowest_balance, 2)}, leaving limited room for timing disruptions."
        )
        recommended_focus.append("Increase near-term liquidity buffer")
        if primary_driver is None:
            primary_driver = "thin_cash_buffer"

    negative_days_count = int((df["net_flow"] < 0).sum())
    negative_ratio = negative_days_count / max(1, len(df))

    if negative_ratio >= 0.45 and scenario_type is None:
        drivers.append("Sustained negative daily cash movement")
        explanation_parts.append(
            "A large share of projected days have negative net cash flow, which steadily weakens liquidity."
        )
        recommended_focus.append("Review recurring outflows and weak inflow coverage")
        if primary_driver is None:
            primary_driver = "persistent_negative_flow"

    neg_df = df[df["net_flow"] < 0].copy()
    if len(neg_df) >= 3 and scenario_type is None:
        largest_negatives = neg_df.nsmallest(5, "net_flow").copy()
        total_negative = abs(float(neg_df["net_flow"].sum())) if len(neg_df) else 0.0
        top_negative_share = (
            abs(float(largest_negatives["net_flow"].sum())) / total_negative
            if total_negative > 0
            else 0.0
        )

        if len(largest_negatives) >= 2:
            window_days = (largest_negatives["date"].max() - largest_negatives["date"].min()).days
        else:
            window_days = 999

        if top_negative_share >= 0.55 and window_days <= 14:
            drivers.append("Concentrated outflows in a short period")
            explanation_parts.append(
                "A small number of larger outflow days account for a meaningful share of downside pressure over a short time window."
            )
            recommended_focus.append("Review timing of larger near-term payments")
            if primary_driver is None:
                primary_driver = "outflow_concentration"

    if scenario_type == "cash_injection":
        drivers.append("Liquidity support required")
        explanation_parts.append(
            "Scenario testing shows that direct funding materially improves the projected cash floor, indicating a structural need for added liquidity support."
        )
        recommended_focus.append("Evaluate funding size and timing")
        recommended_focus.append("Assess whether a temporary cash buffer is required")
        if primary_driver is None:
            primary_driver = "funding_dependence"

    elif scenario_type in {"delay_outflows", "accelerate_inflows", "timing_adjustment"}:
        drivers.append("High sensitivity to cash timing")
        explanation_parts.append(
            "Scenario testing shows that shifting cash timing materially improves the downside liquidity profile."
        )
        recommended_focus.append("Prioritize timing-based interventions")
        if primary_driver is None:
            primary_driver = "timing_sensitivity"

    if failure_probability is not None and failure_probability >= 0.30 and scenario_type is None:
        drivers.append("Meaningful downside exposure in simulation")
        explanation_parts.append(
            f"Simulation results indicate elevated downside exposure, with an estimated {round(failure_probability * 100)}% probability of liquidity failure within the forecast horizon."
        )
        recommended_focus.append("Stress-test contingency actions before major payment periods")
        if primary_driver is None:
            primary_driver = "simulated_downside_risk"

    if primary_driver is None:
        return {
            "primary_driver": "stable_profile",
            "drivers": ["No material short-term driver detected"],
            "explanation": "The forecast does not currently indicate a strong short-term liquidity driver. Cash remains relatively stable within the forecast horizon.",
            "recommended_focus": ["Continue monitoring cash performance"],
        }

    return {
        "primary_driver": primary_driver,
        "drivers": list(dict.fromkeys(drivers)),
        "explanation": " ".join(explanation_parts),
        "recommended_focus": list(dict.fromkeys(recommended_focus)),
    }


def _compute_metrics_from_daily(daily: pd.Series, opening_balance: float, warning_threshold: float) -> dict:
    balance = float(opening_balance) + daily.cumsum()

    below_threshold_at_start = False
    if warning_threshold and float(warning_threshold) > 0:
        below_threshold_at_start = bool(balance.iloc[0] <= float(warning_threshold))

    lowest_balance = float(balance.min())
    lowest_date = balance.idxmin().date().isoformat()

    negative_days = balance[balance < 0]
    warning_threshold = float(warning_threshold)

    if len(negative_days) > 0:
        first_negative_ts = negative_days.index[0]
        first_negative_date = first_negative_ts.date().isoformat()
        days_until_negative = int((first_negative_ts - daily.index[0]).days)
    else:
        first_negative_ts = None
        first_negative_date = None
        days_until_negative = None

    if warning_threshold > 0:
        b = balance
        if first_negative_ts is not None:
            b = b[b.index < first_negative_ts]

        crossed = (b.shift(1) > warning_threshold) & (b <= warning_threshold)
        warning_cross_days = b[crossed]

        if len(warning_cross_days) > 0:
            first_warning_ts = warning_cross_days.index[0]
            first_warning_date = first_warning_ts.date().isoformat()
            days_until_warning = int((first_warning_ts - balance.index[0]).days)
        else:
            first_warning_date = None
            days_until_warning = None
    else:
        first_warning_date = None
        days_until_warning = None

    series = [
        {
            "date": d.date().isoformat(),
            "net_flow": float(daily.loc[d]),
            "balance": float(balance.loc[d]),
        }
        for d in daily.index
    ]

    return {
        "start_date": daily.index[0].date().isoformat(),
        "end_date": daily.index[-1].date().isoformat(),
        "lowest_balance": lowest_balance,
        "lowest_balance_date": lowest_date,
        "first_negative_date": first_negative_date,
        "days_until_negative": days_until_negative,
        "warning_threshold": float(warning_threshold),
        "first_warning_date": first_warning_date,
        "days_until_warning": days_until_warning,
        "series": series,
        "preview_first_14_days": series[: min(14, len(series))],
        "below_threshold_at_start": below_threshold_at_start,
    }


def _shift_outflows_forward(daily: pd.Series, shift_days: int) -> pd.Series:
    shift_days = int(max(0, shift_days))
    if shift_days == 0:
        return daily.copy()

    shifted = daily.copy()
    neg = shifted[shifted < 0].copy()
    shifted.loc[neg.index] = 0.0

    for d, val in neg.items():
        target = d + pd.Timedelta(days=shift_days)
        if target in shifted.index:
            shifted.loc[target] = float(shifted.loc[target]) + float(val)

    return shifted


def _shift_inflows_earlier(daily: pd.Series, shift_days: int) -> pd.Series:
    shift_days = int(max(0, shift_days))
    if shift_days == 0:
        return daily.copy()

    shifted = daily.copy()
    pos = shifted[shifted > 0].copy()
    shifted.loc[pos.index] = 0.0

    for d, val in pos.items():
        target = d - pd.Timedelta(days=shift_days)
        if target in shifted.index:
            shifted.loc[target] = float(shifted.loc[target]) + float(val)
        else:
            earliest = shifted.index.min()
            shifted.loc[earliest] = float(shifted.loc[earliest]) + float(val)

    return shifted.sort_index()


def _apply_cash_injection(daily: pd.Series, amount: float, injection_date: pd.Timestamp) -> pd.Series:
    amount = float(max(0, amount))
    if amount <= 0:
        return daily.copy()

    adjusted = daily.copy()
    if injection_date in adjusted.index:
        adjusted.loc[injection_date] = float(adjusted.loc[injection_date]) + amount
    return adjusted


def build_scenario_ranking(
    data: list[dict],
    opening_balance: float,
    horizon_days: int,
    baseline_window_days: int,
    warning_threshold: float,
    outflow_shift_days: int,
    inflow_shift_days: int,
    cash_injection_amount: float,
) -> dict | None:
    outflow_shift_days = int(max(0, outflow_shift_days))
    inflow_shift_days = int(max(0, inflow_shift_days))
    cash_injection_amount = float(max(0, cash_injection_amount))

    if outflow_shift_days == 0 and inflow_shift_days == 0 and cash_injection_amount == 0:
        return None

    baseline = forecast_cash(
        data,
        opening_balance=opening_balance,
        horizon_days=horizon_days,
        baseline_window_days=baseline_window_days,
        outflow_shift_days=0,
        inflow_shift_days=0,
        cash_injection_amount=0,
        warning_threshold=warning_threshold,
    )
    if "error" in baseline:
        return None

    compare_start = (
        pd.to_datetime(baseline["historical_end_date"]) + pd.Timedelta(days=1)
    ).date().isoformat()

    baseline_lowest = _lowest_balance_from_date(baseline.get("series", []), compare_start)
    if baseline_lowest is None:
        baseline_lowest = float(baseline.get("lowest_balance", 0) or 0)

    options = []

    def add_option(result: dict, key: str, label: str, outflow: int, inflow: int, injection: float):
        if not result.get("scenario"):
            return

        scen = result["scenario"]

        scenario_lowest = _lowest_balance_from_date(scen.get("series", []), compare_start)
        scenario_lowest_date = _lowest_balance_date_from_date(scen.get("series", []), compare_start)

        if scenario_lowest is None:
            scenario_lowest = float(scen.get("lowest_balance", 0) or 0)
            scenario_lowest_date = scen.get("lowest_balance_date")

        failure_probability = scen.get("liquidity_failure", {}).get("probability")
        if failure_probability is None:
            failure_probability = 1.0

        options.append(
            {
                "key": key,
                "label": label,
                "scenario_type": scen.get("scenario_type"),
                "lowest_balance": float(scenario_lowest),
                "lowest_balance_date": scenario_lowest_date,
                "improvement_vs_baseline": float(scenario_lowest - baseline_lowest),
                "risk": scen.get("risk"),
                "failure_probability": float(failure_probability),
                "outflow_shift_days": outflow,
                "inflow_shift_days": inflow,
                "cash_injection_amount": injection,
            }
        )

    if outflow_shift_days > 0:
        add_option(
            forecast_cash(
                data,
                opening_balance=opening_balance,
                horizon_days=horizon_days,
                baseline_window_days=baseline_window_days,
                outflow_shift_days=outflow_shift_days,
                inflow_shift_days=0,
                cash_injection_amount=0,
                warning_threshold=warning_threshold,
            ),
            "delay_outflows",
            "Delaying outflows",
            outflow_shift_days,
            0,
            0,
        )

    if inflow_shift_days > 0:
        add_option(
            forecast_cash(
                data,
                opening_balance=opening_balance,
                horizon_days=horizon_days,
                baseline_window_days=baseline_window_days,
                outflow_shift_days=0,
                inflow_shift_days=inflow_shift_days,
                cash_injection_amount=0,
                warning_threshold=warning_threshold,
            ),
            "accelerate_inflows",
            "Accelerating inflows",
            0,
            inflow_shift_days,
            0,
        )

    if outflow_shift_days > 0 and inflow_shift_days > 0:
        add_option(
            forecast_cash(
                data,
                opening_balance=opening_balance,
                horizon_days=horizon_days,
                baseline_window_days=baseline_window_days,
                outflow_shift_days=outflow_shift_days,
                inflow_shift_days=inflow_shift_days,
                cash_injection_amount=0,
                warning_threshold=warning_threshold,
            ),
            "timing_adjustment",
            "Adjusting cash timing",
            outflow_shift_days,
            inflow_shift_days,
            0,
        )

    if cash_injection_amount > 0:
        add_option(
            forecast_cash(
                data,
                opening_balance=opening_balance,
                horizon_days=horizon_days,
                baseline_window_days=baseline_window_days,
                outflow_shift_days=0,
                inflow_shift_days=0,
                cash_injection_amount=cash_injection_amount,
                warning_threshold=warning_threshold,
            ),
            "cash_injection",
            "Injecting cash",
            0,
            0,
            cash_injection_amount,
        )

    if not options:
        return None

    ranked = sorted(
        options,
        key=lambda item: (
            float(item.get("improvement_vs_baseline", 0) or 0),
            -float(item.get("failure_probability", 1) or 1),
            float(item.get("lowest_balance", 0) or 0),
        ),
        reverse=True,
    )

    standalone = [
        item
        for item in ranked
        if item["key"] in {"delay_outflows", "accelerate_inflows", "cash_injection"}
    ]

    best_standalone = standalone[0] if standalone else None
    best_overall = ranked[0]

    return {
        "comparison_start_date": compare_start,
        "baseline_lowest_balance": baseline_lowest,
        "best_overall": best_overall,
        "best_standalone": best_standalone,
        "ranked_options": ranked,
    }


app = FastAPI()

FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    os.getenv("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in FRONTEND_ORIGINS if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/quality-check")
def quality_check(data: list[dict]):
    df = pd.DataFrame(data)
    df = normalize_expected_columns(df)

    if df.empty:
        return {
            "rows": 0,
            "columns": [],
            "missing_values": {},
            "quality_score": 0,
            "warnings": ["No data received."],
        }

    missing_values = df.isnull().sum().to_dict()
    total_cells = max(1, int(df.shape[0] * df.shape[1]))
    missing_cells = int(df.isnull().sum().sum())
    missing_ratio = missing_cells / total_cells

    score = 100 - (missing_ratio * 60)
    score = max(0, round(score, 1))

    warnings = []
    if missing_cells > 0:
        warnings.append(f"Missing data: {missing_cells} empty cells ({missing_ratio:.1%} of all fields).")

    date_col = "date" if "date" in df.columns else None
    missing_days = None

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        bad_dates = int(dates.isna().sum())
        if bad_dates > 0:
            warnings.append(f"Unparseable dates: {bad_dates} row(s).")

        clean_dates = dates.dropna().sort_values()
        if len(clean_dates) >= 2:
            span_days = (clean_dates.iloc[-1] - clean_dates.iloc[0]).days
            if span_days >= 60:
                start = clean_dates.iloc[0].to_period("M").to_timestamp()
                end = clean_dates.iloc[-1].to_period("M").to_timestamp()
                all_months = pd.date_range(start, end, freq="MS")

                present_months = clean_dates.dt.to_period("M").astype(str).drop_duplicates()
                present_months = pd.to_datetime(present_months)

                missing_months = all_months.difference(present_months)
                if len(missing_months) > 0:
                    warnings.append(f"Missing months in range: {len(missing_months)} {'month' if len(missing_months) == 1 else 'months'}.")
            else:
                start = clean_dates.iloc[0].normalize()
                end = clean_dates.iloc[-1].normalize()
                all_days = pd.date_range(start, end, freq="D")
                present_days = clean_dates.dt.normalize().drop_duplicates()
                missing_days = all_days.difference(present_days)
        else:
            warnings.append("Not enough valid dates to check for missing days.")
    else:
        warnings.append("No 'date' column found (expected a column named 'date').")

    if missing_days is not None and len(missing_days) > 0:
        warnings.append(f"Missing days in range: {len(missing_days)} {'day' if len(missing_days) == 1 else 'days'}.")

    amount_col = "amount" if "amount" in df.columns else None
    if amount_col:
        amounts = pd.to_numeric(df[amount_col], errors="coerce")
        bad_amounts = int(amounts.isna().sum())
        if bad_amounts > 0:
            warnings.append(f"Non-numeric amounts: {bad_amounts} row(s).")
    else:
        warnings.append("No 'amount' column found (expected a column named 'amount').")

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": missing_values,
        "quality_score": score,
        "warnings": warnings,
    }


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = parse_uploaded_csv_bytes(content)
    df = normalize_expected_columns(df)
    validate_required_columns(df)
    data = df.to_dict(orient="records")
    return quality_check(data)


@app.post("/forecast-cash")
def forecast_cash(
    data: list[dict],
    opening_balance: float = 0.0,
    horizon_days: int = 30,
    baseline_window_days: int = 30,
    outflow_shift_days: int = 0,
    inflow_shift_days: int = 0,
    cash_injection_amount: float = 0.0,
    warning_threshold: float = 0.0,
):
    df = pd.DataFrame(data)
    df = normalize_expected_columns(df)

    if df.empty:
        return {"error": "No data received."}
    if "date" not in df.columns or "amount" not in df.columns:
        return {"error": "Expected columns: 'date' and 'amount'."}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).copy()

    if df.empty:
        return {"error": "After cleaning, no valid rows remained. Check that dates are valid and amounts are numeric."}

    daily_hist = df.groupby(df["date"].dt.normalize())["amount"].sum().sort_index()
    hist_start = daily_hist.index.min()
    hist_end = daily_hist.index.max()

    horizon_days = int(max(0, horizon_days))
    outflow_shift_days = int(max(0, outflow_shift_days))
    inflow_shift_days = int(max(0, inflow_shift_days))
    cash_injection_amount = float(max(0, cash_injection_amount))

    future_extension = max(outflow_shift_days, 0)
    start_extension = max(inflow_shift_days, 0)

    start = hist_start - pd.Timedelta(days=start_extension)
    end = hist_end + pd.Timedelta(days=horizon_days + future_extension)
    all_days = pd.date_range(start, end, freq="D")

    daily = daily_hist.reindex(all_days, fill_value=0.0)

    cutoff = hist_end - pd.Timedelta(days=baseline_window_days - 1)
    hist_window = daily_hist[(daily_hist.index >= cutoff) & (daily_hist != 0)]
    if len(hist_window) < 7:
        hist_window = daily_hist[daily_hist != 0]

    if len(hist_window) >= 14:
        weekday_means = hist_window.groupby(hist_window.index.weekday).mean()
        global_mean = float(hist_window.mean()) if len(hist_window) else 0.0

        def estimate_flow(ts: pd.Timestamp) -> float:
            return float(weekday_means.get(ts.weekday(), global_mean))

        forecast_method = "weekday_mean_over_recent_window"
    else:
        global_mean = float(hist_window.mean()) if len(hist_window) else 0.0

        def estimate_flow(ts: pd.Timestamp) -> float:
            return float(global_mean)

        forecast_method = "global_mean_over_recent_window"

    future_mask = all_days > hist_end
    for d in all_days[future_mask]:
        daily.loc[d] = estimate_flow(d)

    baseline_metrics = _compute_metrics_from_daily(
        daily,
        opening_balance=float(opening_balance),
        warning_threshold=float(warning_threshold),
    )

    baseline = {
        "opening_balance": float(opening_balance),
        "historical_end_date": hist_end.date().isoformat(),
        "horizon_days": horizon_days,
        "baseline_window_days": int(baseline_window_days),
        "forecast_method": forecast_method,
        **baseline_metrics,
    }

    try:
        baseline["risk"] = compute_liquidity_risk(baseline)
    except Exception as e:
        baseline["risk"] = {"error": f"Risk calc failed: {str(e)}"}

    try:
        baseline["liquidity_failure"] = monte_carlo_liquidity_failure(
            hist_flows=hist_window.values,
            opening_balance=float(baseline["series"][len(daily_hist) - 1]["balance"]),
            horizon_days=int(horizon_days),
            simulations=5000,
            seed=42,
        )
    except Exception as e:
        baseline["liquidity_failure"] = {"error": f"Monte Carlo failed: {str(e)}"}

    baseline["risk_drivers"] = identify_risk_drivers(baseline)

    scenario = None
    comparison = None

    if outflow_shift_days > 0 or inflow_shift_days > 0 or cash_injection_amount > 0:
        scenario_daily = daily.copy()
        scenario_type = None

        if outflow_shift_days > 0 and inflow_shift_days == 0 and cash_injection_amount == 0:
            scenario_daily = _shift_outflows_forward(scenario_daily, outflow_shift_days)
            scenario_type = "delay_outflows"
        elif inflow_shift_days > 0 and outflow_shift_days == 0 and cash_injection_amount == 0:
            scenario_daily = _shift_inflows_earlier(scenario_daily, inflow_shift_days)
            scenario_type = "accelerate_inflows"
        elif cash_injection_amount > 0 and outflow_shift_days == 0 and inflow_shift_days == 0:
            injection_date = hist_end + pd.Timedelta(days=1)
            scenario_daily = _apply_cash_injection(scenario_daily, cash_injection_amount, injection_date)
            scenario_type = "cash_injection"
        else:
            if outflow_shift_days > 0:
                scenario_daily = _shift_outflows_forward(scenario_daily, outflow_shift_days)
            if inflow_shift_days > 0:
                scenario_daily = _shift_inflows_earlier(scenario_daily, inflow_shift_days)
            if cash_injection_amount > 0:
                injection_date = hist_end + pd.Timedelta(days=1)
                scenario_daily = _apply_cash_injection(scenario_daily, cash_injection_amount, injection_date)

            if cash_injection_amount > 0 and (outflow_shift_days > 0 or inflow_shift_days > 0):
                scenario_type = "cash_injection"
            else:
                scenario_type = "timing_adjustment"

        scenario_metrics = _compute_metrics_from_daily(
            scenario_daily,
            opening_balance=float(opening_balance),
            warning_threshold=float(warning_threshold),
        )

        scenario = {
            "scenario_type": scenario_type,
            "outflow_shift_days": outflow_shift_days,
            "inflow_shift_days": inflow_shift_days,
            "cash_injection_amount": cash_injection_amount,
            "opening_balance": float(opening_balance),
            "historical_end_date": hist_end.date().isoformat(),
            "horizon_days": horizon_days,
            "baseline_window_days": int(baseline_window_days),
            "forecast_method": forecast_method,
            **scenario_metrics,
        }

        try:
            scenario["risk"] = compute_liquidity_risk(scenario)
        except Exception as e:
            scenario["risk"] = {"error": f"Risk calc failed: {str(e)}"}

        try:
            scenario["liquidity_failure"] = monte_carlo_liquidity_failure(
                hist_flows=hist_window.values,
                opening_balance=float(scenario["series"][len(daily_hist) - 1]["balance"]),
                horizon_days=int(horizon_days),
                simulations=5000,
                seed=42,
            )
        except Exception as e:
            scenario["liquidity_failure"] = {"error": f"Monte Carlo failed: {str(e)}"}

        scenario["risk_drivers"] = identify_risk_drivers(scenario)

        compare_start = (hist_end + pd.Timedelta(days=1)).date().isoformat()

        baseline_compare_lowest = _lowest_balance_from_date(baseline.get("series", []), compare_start)
        scenario_compare_lowest = _lowest_balance_from_date(scenario.get("series", []), compare_start)
        scenario_compare_lowest_date = _lowest_balance_date_from_date(scenario.get("series", []), compare_start)

        if baseline_compare_lowest is None:
            baseline_compare_lowest = float(baseline.get("lowest_balance", 0) or 0)

        if scenario_compare_lowest is None:
            scenario_compare_lowest = float(scenario.get("lowest_balance", 0) or 0)

        if scenario_compare_lowest_date is None:
            scenario_compare_lowest_date = scenario.get("lowest_balance_date")

        comparison = {
            "scenario_type": scenario_type,
            "outflow_shift_days": outflow_shift_days,
            "inflow_shift_days": inflow_shift_days,
            "cash_injection_amount": cash_injection_amount,
            "comparison_start_date": compare_start,
            "baseline_first_negative_date": baseline["first_negative_date"],
            "scenario_first_negative_date": scenario["first_negative_date"],
            "baseline_lowest_balance": float(baseline_compare_lowest),
            "scenario_lowest_balance": float(scenario_compare_lowest),
            "scenario_lowest_balance_date": scenario_compare_lowest_date,
            "delta_lowest_balance": float(scenario_compare_lowest - baseline_compare_lowest),
        }

    return {
        **baseline,
        "scenario": scenario,
        "comparison": comparison,
    }


def build_selected_scenario_summary(forecast: dict, decision_impact: dict | None):
    if not forecast.get("scenario"):
        return None

    scenario = forecast["scenario"]
    lowest_balance = decision_impact.get("scenario_lowest_balance") if decision_impact else scenario.get("lowest_balance")
    lowest_balance_date = decision_impact.get("scenario_lowest_balance_date") if decision_impact else scenario.get("lowest_balance_date")
    improvement = decision_impact.get("balance_improvement") if decision_impact else None
    action_label = decision_impact.get("action") if decision_impact else format_action_label_from_type(scenario.get("scenario_type"))

    return {
        "headline": "The selected scenario materially improves liquidity resilience.",
        "details": (
            f"No negative cash position is projected within the current forecast horizon. "
            f"The lowest projected balance is {round(float(lowest_balance or 0), 2)}"
            f"{f' on {lowest_balance_date}' if lowest_balance_date else ''}. "
            f"{action_label} improves worst-case liquidity by {round(float(improvement or 0), 2)} kr."
        ),
        "suggestions": [
            "The tested scenario improves downside liquidity and may be worth operational follow-up."
        ],
    }


@app.post("/upload-csv-forecast")
async def upload_csv_forecast(
    file: UploadFile = File(...),
    opening_balance: float = 0.0,
    horizon_days: int = 30,
    baseline_window_days: int = 30,
    outflow_shift_days: int = 0,
    inflow_shift_days: int = 0,
    cash_injection_amount: float = 0.0,
    warning_threshold: float = 0.0,
):
    print("📥 Received upload request")
    content = await file.read()

    def process():
        df = parse_uploaded_csv_bytes(content)
        df = normalize_expected_columns(df)
        validate_required_columns(df)

        data = df.to_dict(orient="records")
        quality = quality_check(data)

        forecast = forecast_cash(
            data,
            opening_balance=opening_balance,
            horizon_days=horizon_days,
            baseline_window_days=baseline_window_days,
            outflow_shift_days=outflow_shift_days,
            inflow_shift_days=inflow_shift_days,
            cash_injection_amount=cash_injection_amount,
            warning_threshold=warning_threshold,
        )

        ranking = build_scenario_ranking(
            data=data,
            opening_balance=opening_balance,
            horizon_days=horizon_days,
            baseline_window_days=baseline_window_days,
            warning_threshold=warning_threshold,
            outflow_shift_days=outflow_shift_days,
            inflow_shift_days=inflow_shift_days,
            cash_injection_amount=cash_injection_amount,
        )

        return quality, forecast, ranking

    try:
        quality, forecast, ranking = await run_in_threadpool(process)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if "error" in forecast:
        return {
            "quality": quality,
            "forecast": forecast,
            "summary": {
                "headline": "Forecast could not be generated",
                "details": forecast["error"],
                "suggestions": [
                    "Check that the CSV contains 'date' and 'amount' columns",
                    "Ensure all dates and amounts are valid",
                ],
            },
            "decision_impact": None,
            "risk": None,
            "risk_drivers": None,
            "scenario_ranking": None,
        }

    decision_impact = None
    if forecast.get("scenario"):
        action_label = detect_action_label(forecast)
        decision_impact = generate_decision_impact(
            baseline=forecast,
            scenario=forecast["scenario"],
            action_label=action_label,
            currency_symbol="kr.",
        )

        forecast["scenario"]["risk_drivers"] = identify_risk_drivers(
            forecast["scenario"],
            decision_impact=decision_impact,
        )

    if forecast.get("scenario"):
        summary = build_selected_scenario_summary(forecast, decision_impact)
        summary_target = forecast["scenario"]
    else:
        summary = build_executive_summary(forecast)
        summary_target = forecast

    print("📤 Returning forecast response")
    return {
        "quality": quality,
        "forecast": forecast,
        "summary": summary,
        "decision_impact": decision_impact,
        "risk": summary_target.get("risk"),
        "risk_drivers": summary_target.get("risk_drivers"),
        "scenario_ranking": ranking,
    }


@app.post("/upload/{company_token}")
async def upload_for_company(
    company_token: str = Path(..., description="Unique company identifier"),
    file: UploadFile = File(...),
    opening_balance: float = 0.0,
    horizon_days: int = 30,
    baseline_window_days: int = 30,
    outflow_shift_days: int = 0,
    inflow_shift_days: int = 0,
    cash_injection_amount: float = 0.0,
    warning_threshold: float = 0.0,
):
    content = await file.read()
    df = parse_uploaded_csv_bytes(content)
    df = normalize_expected_columns(df)
    validate_required_columns(df)

    data = df.to_dict(orient="records")
    quality = quality_check(data)
    forecast = forecast_cash(
        data,
        opening_balance=opening_balance,
        horizon_days=horizon_days,
        baseline_window_days=baseline_window_days,
        outflow_shift_days=outflow_shift_days,
        inflow_shift_days=inflow_shift_days,
        cash_injection_amount=cash_injection_amount,
        warning_threshold=warning_threshold,
    )

    if "error" in forecast:
        raise HTTPException(status_code=400, detail=forecast["error"])

    ranking = build_scenario_ranking(
        data=data,
        opening_balance=opening_balance,
        horizon_days=horizon_days,
        baseline_window_days=baseline_window_days,
        warning_threshold=warning_threshold,
        outflow_shift_days=outflow_shift_days,
        inflow_shift_days=inflow_shift_days,
        cash_injection_amount=cash_injection_amount,
    )

    decision_impact = None
    if forecast.get("scenario"):
        action_label = detect_action_label(forecast)
        decision_impact = generate_decision_impact(
            baseline=forecast,
            scenario=forecast["scenario"],
            action_label=action_label,
            currency_symbol="kr.",
        )

    if forecast.get("scenario"):
        summary = build_selected_scenario_summary(forecast, decision_impact)
        summary_target = forecast["scenario"]
    else:
        summary = build_executive_summary(forecast)
        summary_target = forecast

    return {
        "company": company_token,
        "quality": quality,
        "forecast": forecast,
        "summary": summary,
        "decision_impact": decision_impact,
        "risk": summary_target.get("risk"),
        "risk_drivers": summary_target.get("risk_drivers"),
        "scenario_ranking": ranking,
    }


COMPANIES = {}
LAST_ALERT_KEY = "last_alert_sent_first_warning_date"


def should_send_alert(company: dict, forecast: dict) -> bool:
    first_warning_date = forecast.get("first_warning_date")
    if first_warning_date is None:
        return False
    last_sent = company.get(LAST_ALERT_KEY)
    return last_sent != first_warning_date


def send_email_alert(to_email: str, subject: str, html_body: str):
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("ALERTS_FROM_EMAIL", "alerts@previa.local")

    if not api_key:
        print("⚠️ SENDGRID_API_KEY not set. Skipping email send.")
        return

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=html_body,
    )
    sg = SendGridAPIClient(api_key)
    try:
        sg.send(message)
    except Exception as e:
        print(f"⚠️ SendGrid error: {e}")


def run_forecasts_for_all_companies():
    for company_id, company in COMPANIES.items():
        data = company.get("data")
        if not data:
            continue

        forecast = forecast_cash(
            data,
            opening_balance=float(company.get("opening_balance", 0.0)),
            horizon_days=int(company.get("horizon_days", 30)),
            baseline_window_days=int(company.get("baseline_window_days", 30)),
            outflow_shift_days=int(company.get("outflow_shift_days", 0)),
            inflow_shift_days=int(company.get("inflow_shift_days", 0)),
            cash_injection_amount=float(company.get("cash_injection_amount", 0.0)),
            warning_threshold=float(company.get("warning_threshold", 0.0)),
        )

        summary_target = forecast["scenario"] if forecast.get("scenario") else forecast
        summary = build_executive_summary(summary_target)

        if should_send_alert(company, summary_target):
            subject = f"[Previa Alert] {summary['headline']}"
            html_body = f"""
            <h3>{summary['headline']}</h3>
            <p>{summary['details']}</p>
            <ul>
              {''.join(f"<li>{s}</li>" for s in summary["suggestions"])}
            </ul>
            <p><b>First warning date:</b> {summary_target.get("first_warning_date")}</p>
            <p><b>Warning threshold:</b> {summary_target.get("warning_threshold")}</p>
            <p><b>Liquidity Risk Score:</b> {summary_target.get("risk", {}).get("score", "N/A")}</p>
            <p><b>Risk Level:</b> {summary_target.get("risk", {}).get("level", "N/A")}</p>
            <p><b>Failure probability:</b> {summary_target.get("liquidity_failure", {}).get("probability", "N/A")}</p>
            """
            send_email_alert(company["email"], subject, html_body)
            company[LAST_ALERT_KEY] = summary_target.get("first_warning_date")
            print(f"✅ Alert sent to {company['email']} for company_id={company_id}")
        else:
            print(f"ℹ️ No new alert for company_id={company_id}")


class CompanyRegister(BaseModel):
    company_id: str
    email: str
    opening_balance: float = 0.0
    horizon_days: int = 30
    baseline_window_days: int = 30
    outflow_shift_days: int = 0
    inflow_shift_days: int = 0
    cash_injection_amount: float = 0.0
    warning_threshold: float = 0.0


@app.post("/register-company")
def register_company(payload: CompanyRegister):
    COMPANIES[payload.company_id] = {
        "email": payload.email,
        "opening_balance": payload.opening_balance,
        "horizon_days": payload.horizon_days,
        "baseline_window_days": payload.baseline_window_days,
        "outflow_shift_days": payload.outflow_shift_days,
        "inflow_shift_days": payload.inflow_shift_days,
        "cash_injection_amount": payload.cash_injection_amount,
        "warning_threshold": payload.warning_threshold,
        "data": None,
        LAST_ALERT_KEY: None,
    }
    return {"status": "registered", "company_id": payload.company_id}


@app.post("/upload-csv-forecast-company")
async def upload_csv_forecast_company(
    company_id: str,
    file: UploadFile = File(...),
):
    if company_id not in COMPANIES:
        raise HTTPException(status_code=404, detail="Unknown company_id. Register first.")

    content = await file.read()
    df = parse_uploaded_csv_bytes(content)
    df = normalize_expected_columns(df)
    validate_required_columns(df)

    data = df.to_dict(orient="records")
    COMPANIES[company_id]["data"] = data

    company = COMPANIES[company_id]
    forecast = forecast_cash(
        data,
        opening_balance=float(company.get("opening_balance", 0.0)),
        horizon_days=int(company.get("horizon_days", 30)),
        baseline_window_days=int(company.get("baseline_window_days", 30)),
        outflow_shift_days=int(company.get("outflow_shift_days", 0)),
        inflow_shift_days=int(company.get("inflow_shift_days", 0)),
        cash_injection_amount=float(company.get("cash_injection_amount", 0.0)),
        warning_threshold=float(company.get("warning_threshold", 0.0)),
    )

    ranking = build_scenario_ranking(
        data=data,
        opening_balance=float(company.get("opening_balance", 0.0)),
        horizon_days=int(company.get("horizon_days", 30)),
        baseline_window_days=int(company.get("baseline_window_days", 30)),
        warning_threshold=float(company.get("warning_threshold", 0.0)),
        outflow_shift_days=int(company.get("outflow_shift_days", 0)),
        inflow_shift_days=int(company.get("inflow_shift_days", 0)),
        cash_injection_amount=float(company.get("cash_injection_amount", 0.0)),
    )

    decision_impact = None
    if forecast.get("scenario"):
        action_label = detect_action_label(forecast)
        decision_impact = generate_decision_impact(
            baseline=forecast,
            scenario=forecast["scenario"],
            action_label=action_label,
            currency_symbol="kr.",
        )

    if forecast.get("scenario"):
        summary = build_selected_scenario_summary(forecast, decision_impact)
        summary_target = forecast["scenario"]
    else:
        summary = build_executive_summary(forecast)
        summary_target = forecast

    return {
        "status": "uploaded",
        "company_id": company_id,
        "quality": quality_check(data),
        "forecast": forecast,
        "summary": summary,
        "decision_impact": decision_impact,
        "risk": summary_target.get("risk"),
        "risk_drivers": summary_target.get("risk_drivers"),
        "scenario_ranking": ranking,
    }


scheduler = BackgroundScheduler()


@app.on_event("startup")
def _start_scheduler():
    if not scheduler.running:
        scheduler.add_job(
            run_forecasts_for_all_companies,
            "interval",
            hours=6,
            id="forecast_job",
            replace_existing=True,
        )
        scheduler.start()
        print("✅ Scheduler started")
    else:
        print("ℹ️ Scheduler already running")


@app.on_event("shutdown")
def _stop_scheduler():
    scheduler.shutdown()
    print("🛑 Scheduler stopped")