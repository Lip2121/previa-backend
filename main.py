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


def generate_decision_impact(baseline: dict, scenario: dict, action_label: str, currency_symbol: str) -> dict:
    baseline_negative = baseline.get("first_negative_date")
    scenario_negative = scenario.get("first_negative_date")

    days_gained = None
    if baseline_negative and scenario_negative:
        baseline_date = pd.to_datetime(baseline_negative)
        scenario_date = pd.to_datetime(scenario_negative)
        days_gained = int((scenario_date - baseline_date).days)

    return {
        "action": action_label,
        "baseline_first_negative": baseline_negative,
        "scenario_first_negative": scenario_negative,
        "days_gained": days_gained,
        "baseline_lowest_balance": baseline.get("lowest_balance"),
        "scenario_lowest_balance": scenario.get("lowest_balance"),
        "balance_improvement": float(scenario.get("lowest_balance", 0) - baseline.get("lowest_balance", 0)),
        "currency_symbol": currency_symbol,
    }


def build_executive_summary(forecast: dict):
    days = forecast["days_until_negative"]
    first_neg = forecast["first_negative_date"]
    lowest = forecast["lowest_balance"]

    below = forecast.get("below_threshold_at_start", False)
    warning_threshold = forecast.get("warning_threshold", 0.0)

    if below and warning_threshold and first_neg is not None:
        headline = "Cash is already below warning threshold"
        details = (
            f"Your cash balance starts below the warning threshold ({warning_threshold}). "
            f"Cash is projected to turn negative on {first_neg}."
        )
        suggestions = [
            "Review near-term outflows immediately",
            "Accelerate incoming payments where possible",
            "Consider short-term financing",
        ]
        return {
            "headline": headline,
            "details": details,
            "suggestions": suggestions,
        }

    if first_neg is None:
        headline = "No immediate liquidity risk detected"
        details = (
            "Based on the current data, your cash balance does not "
            "turn negative within the forecast period."
        )
        suggestions = []
    else:
        headline = f"Cash turns negative in {days} day{'s' if days != 1 else ''}"
        details = (
            f"Your cash balance is projected to turn negative on {first_neg}. "
            f"The lowest expected balance is {lowest}. "
            "Without intervention, liquidity risk becomes immediate."
        )
        suggestions = [
            f"Review and potentially delay major outflows before {first_neg}",
            "Accelerate incoming payments where possible",
            "Consider securing short-term financing",
        ]

    return {
        "headline": headline,
        "details": details,
        "suggestions": suggestions,
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

    # 1. Time-to-failure score
    if days_until_negative is None:
        time_score = 100
    else:
        time_score = max(0, min(100, days_until_negative * 4))

    # 2. Buffer score
    # Rewards strong minimum balances, penalizes thin buffers even if still positive
    if lowest_balance <= 0:
        buffer_score = 0
    elif lowest_balance >= 5000:
        buffer_score = 100
    else:
        buffer_score = (lowest_balance / 5000) * 100

    # 3. Threshold score
    # If a warning threshold exists, measure how safely above it the business remains
    if warning_threshold > 0:
        if lowest_balance <= warning_threshold:
            threshold_score = 0
        elif lowest_balance >= warning_threshold + 5000:
            threshold_score = 100
        else:
            threshold_score = ((lowest_balance - warning_threshold) / 5000) * 100
    else:
        threshold_score = buffer_score

    score = (
        0.45 * time_score +
        0.30 * buffer_score +
        0.25 * threshold_score
    )

    return {
        "score": round(score),
        "level": get_risk_level(score),
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

    warnings: list[str] = []
    if missing_cells > 0:
        warnings.append(
            f"Missing data: {missing_cells} empty cells ({missing_ratio:.1%} of all fields)."
        )

    date_col = next((c for c in ["date", "Date", "DATE"] if c in df.columns), None)

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
                    n_missing = len(missing_months)
                    month_word = "month" if n_missing == 1 else "months"
                    warnings.append(f"Missing months in range: {n_missing} {month_word}.")
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
        n_missing = len(missing_days)
        day_word = "day" if n_missing == 1 else "days"
        warnings.append(f"Missing days in range: {n_missing} {day_word}.")

    amount_col = next((c for c in ["amount", "Amount", "AMOUNT"] if c in df.columns), None)

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
    text = content.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(text))
    data = df.to_dict(orient="records")
    return quality_check(data)


def _compute_metrics_from_daily(
    daily: pd.Series,
    opening_balance: float,
    warning_threshold: float,
) -> dict:
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


def _shift_outflows_forward(daily: pd.Series, shift_days: int, hist_end: pd.Timestamp) -> pd.Series:
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


@app.post("/forecast-cash")
def forecast_cash(
    data: list[dict],
    opening_balance: float = 0.0,
    horizon_days: int = 30,
    baseline_window_days: int = 30,
    outflow_shift_days: int = 0,
    warning_threshold: float = 0.0,
):
    df = pd.DataFrame(data)

    if df.empty:
        return {"error": "No data received."}
    if "date" not in df.columns or "amount" not in df.columns:
        return {"error": "Expected columns: 'date' and 'amount'."}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).copy()
    if df.empty:
        return {"error": "After cleaning, no valid rows remained (check date/amount formats)."}

    daily_hist = df.groupby(df["date"].dt.normalize())["amount"].sum().sort_index()
    hist_start = daily_hist.index.min()
    hist_end = daily_hist.index.max()

    horizon_days = int(max(0, horizon_days))
    outflow_shift_days = int(max(0, outflow_shift_days))

    end = hist_end + pd.Timedelta(days=horizon_days + outflow_shift_days)
    all_days = pd.date_range(hist_start, end, freq="D")

    daily = daily_hist.reindex(all_days, fill_value=0.0)

    cutoff = hist_end - pd.Timedelta(days=baseline_window_days - 1)
    hist_window = daily_hist[(daily_hist.index >= cutoff) & (daily_hist != 0)]

    if len(hist_window) < 7:
        hist_window = daily_hist[daily_hist != 0]

    if len(hist_window) >= 14:
        weekday_means = hist_window.groupby(hist_window.index.weekday).mean()
        global_mean = float(hist_window.mean()) if len(hist_window) else 0.0

        def estimate_flow(ts: pd.Timestamp) -> float:
            wd = ts.weekday()
            return float(weekday_means.get(wd, global_mean))

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

    scenario = None
    comparison = None

    if outflow_shift_days > 0:
        scenario_daily = _shift_outflows_forward(daily, outflow_shift_days, hist_end)

        scenario_metrics = _compute_metrics_from_daily(
            scenario_daily,
            opening_balance=float(opening_balance),
            warning_threshold=float(warning_threshold),
        )

        scenario = {
            "scenario_type": "delay_outflows",
            "outflow_shift_days": outflow_shift_days,
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

        comparison = {
            "scenario_type": "delay_outflows",
            "outflow_shift_days": outflow_shift_days,
            "baseline_first_negative_date": baseline["first_negative_date"],
            "scenario_first_negative_date": scenario["first_negative_date"],
            "baseline_lowest_balance": baseline["lowest_balance"],
            "scenario_lowest_balance": scenario["lowest_balance"],
            "delta_lowest_balance": float(scenario["lowest_balance"] - baseline["lowest_balance"]),
        }

    return {
        **baseline,
        "scenario": scenario,
        "comparison": comparison,
    }


@app.post("/upload-csv-forecast")
async def upload_csv_forecast(
    file: UploadFile = File(...),
    opening_balance: float = 0.0,
    horizon_days: int = 30,
    baseline_window_days: int = 30,
    outflow_shift_days: int = 0,
    warning_threshold: float = 0.0,
):
    print("📥 Received upload request")

    content = await file.read()

    def process():
        text = content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(text))
        data = df.to_dict(orient="records")

        quality = quality_check(data)

        forecast = forecast_cash(
            data,
            opening_balance=opening_balance,
            horizon_days=horizon_days,
            baseline_window_days=baseline_window_days,
            outflow_shift_days=outflow_shift_days,
            warning_threshold=warning_threshold,
        )

        return quality, forecast

    quality, forecast = await run_in_threadpool(process)

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
        }

    decision_impact = None
    if forecast.get("scenario"):
        decision_impact = generate_decision_impact(
            baseline=forecast,
            scenario=forecast["scenario"],
            action_label="Delaying outflows",
            currency_symbol="kr.",
        )

    summary_target = forecast["scenario"] if forecast.get("scenario") else forecast

    print("📤 Returning forecast response")

    return {
        "quality": quality,
        "forecast": forecast,
        "summary": build_executive_summary(summary_target),
        "decision_impact": decision_impact,
        "risk": summary_target.get("risk"),
    }


@app.post("/upload/{company_token}")
async def upload_for_company(
    company_token: str = Path(..., description="Unique company identifier"),
    file: UploadFile = File(...),
    opening_balance: float = 0.0,
    horizon_days: int = 30,
    baseline_window_days: int = 30,
    outflow_shift_days: int = 0,
    warning_threshold: float = 0.0,
):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(text))

    data = df.to_dict(orient="records")

    quality = quality_check(data)
    forecast = forecast_cash(
        data,
        opening_balance=opening_balance,
        horizon_days=horizon_days,
        baseline_window_days=baseline_window_days,
        outflow_shift_days=outflow_shift_days,
        warning_threshold=warning_threshold,
    )

    if "error" in forecast:
        raise HTTPException(status_code=400, detail=forecast["error"])

    return {
        "company": company_token,
        "quality": quality,
        "forecast": forecast,
        "summary": build_executive_summary(forecast),
        "risk": forecast.get("risk"),
    }


COMPANIES: dict[str, dict] = {}
LAST_ALERT_KEY = "last_alert_sent_first_warning_date"


def should_send_alert(company: dict, forecast: dict) -> bool:
    first_warning_date = forecast.get("first_warning_date")
    if first_warning_date is None:
        return False

    last_sent = company.get(LAST_ALERT_KEY)
    if last_sent == first_warning_date:
        return False

    return True


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
            warning_threshold=float(company.get("warning_threshold", 0.0)),
        )

        summary = build_executive_summary(forecast)

        if should_send_alert(company, forecast):
            subject = f"[Previa Alert] {summary['headline']}"
            html_body = f"""
            <h3>{summary['headline']}</h3>
            <p>{summary['details']}</p>
            <ul>
              {''.join(f"<li>{s}</li>" for s in summary["suggestions"])}
            </ul>
            <p><b>First warning date:</b> {forecast.get("first_warning_date")}</p>
            <p><b>Warning threshold:</b> {forecast.get("warning_threshold")}</p>
            <p><b>Liquidity Risk Score:</b> {forecast.get("risk", {}).get("score", "N/A")}</p>
            <p><b>Risk Level:</b> {forecast.get("risk", {}).get("level", "N/A")}</p>
            <p><b>Failure probability:</b> {forecast.get("liquidity_failure", {}).get("probability", "N/A")}</p>
            """

            send_email_alert(company["email"], subject, html_body)
            company[LAST_ALERT_KEY] = forecast.get("first_warning_date")
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
    warning_threshold: float = 0.0


@app.post("/register-company")
def register_company(payload: CompanyRegister):
    COMPANIES[payload.company_id] = {
        "email": payload.email,
        "opening_balance": payload.opening_balance,
        "horizon_days": payload.horizon_days,
        "baseline_window_days": payload.baseline_window_days,
        "outflow_shift_days": payload.outflow_shift_days,
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
    text = content.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(text))
    data = df.to_dict(orient="records")

    COMPANIES[company_id]["data"] = data

    company = COMPANIES[company_id]
    forecast = forecast_cash(
        data,
        opening_balance=float(company.get("opening_balance", 0.0)),
        horizon_days=int(company.get("horizon_days", 30)),
        baseline_window_days=int(company.get("baseline_window_days", 30)),
        outflow_shift_days=int(company.get("outflow_shift_days", 0)),
        warning_threshold=float(company.get("warning_threshold", 0.0)),
    )

    return {
        "status": "uploaded",
        "company_id": company_id,
        "quality": quality_check(data),
        "forecast": forecast,
        "summary": build_executive_summary(forecast),
        "risk": forecast.get("risk"),
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