from services.liquidity_failure import monte_carlo_liquidity_failure

from fastapi import FastAPI, UploadFile, File, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from io import StringIO
import csv
import os
import json
import math
import secrets
from sqlalchemy import create_engine, inspect, text
import uuid
from datetime import datetime, timezone
import pandas as pd

from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


# ------------------------------------------------------------
# CSV parsing / validation
# ------------------------------------------------------------

def parse_uploaded_csv_bytes(content: bytes) -> pd.DataFrame:
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    decoded_text = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded_text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the CSV. Export it as UTF-8 CSV and try again.",
        )

    text_sample = decoded_text[:65536]
    delimiters = [",", ";", "\t", "|"]
    try:
        sniffed = csv.Sniffer().sniff(text_sample, delimiters="".join(delimiters)).delimiter
        delimiters = [sniffed] + [item for item in delimiters if item != sniffed]
    except csv.Error:
        pass

    candidates = []
    parse_errors = []
    for delimiter in delimiters:
        try:
            candidate = pd.read_csv(
                StringIO(decoded_text),
                sep=delimiter,
                engine="python",
                dtype=str,
                keep_default_na=False,
                skipinitialspace=True,
            )
            candidate.columns = [str(column).strip().lstrip("\ufeff") for column in candidate.columns]
            candidates.append(candidate)
        except Exception as error:
            parse_errors.append(str(error))

    if not candidates:
        detail = parse_errors[0] if parse_errors else "Unknown CSV formatting error"
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file. Export it as CSV and try again. Error: {detail}",
        )

    # The correct separator is normally the candidate producing the most real columns.
    df = max(candidates, key=lambda candidate: (len(candidate.columns), len(candidate.index)))

    if len(df.columns) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Previa found only one column. Re-export the file as CSV using comma, "
                "semicolon, or tab separators, then upload it again."
            ),
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

        if col_clean in {"date", "dato", "transaction date", "posting date", "bogføringsdato", "bogfoeringsdato", "bilagsdato", "betalingsdato", "forfaldsdato"}:
            rename_map[col] = "date"
        elif col_clean in {"amount", "beløb", "belob", "beloeb", "amount dkk", "beløb dkk", "beloeb dkk", "value", "transaction amount", "saldoændring", "saldoaendring"}:
            rename_map[col] = "amount"
        elif col_clean in {"customer", "kunde", "client", "kundenavn"}:
            rename_map[col] = "customer"
        elif col_clean in {"description", "tekst", "text", "message", "details", "posteringstekst", "beskrivelse", "bilagstekst", "notat"}:
            rename_map[col] = "description"
        elif col_clean in {"category", "kategori", "type", "konto", "kontonavn", "kontonummer"}:
            rename_map[col] = "category"
        elif col_clean in {"counterparty", "modpart", "merchant", "supplier", "vendor", "name", "leverandør", "leverandoer", "leverandørnavn", "leverandoernavn"}:
            rename_map[col] = "counterparty"

    return df.rename(columns=rename_map)


def apply_column_mapping(
    df: pd.DataFrame,
    date_column: str | None = None,
    amount_column: str | None = None,
    description_column: str | None = None,
    counterparty_column: str | None = None,
    category_column: str | None = None,
    customer_column: str | None = None,
) -> pd.DataFrame:
    """Map user-selected CSV columns into Previa's expected internal column names."""
    if df.empty:
        return df

    mapped = df.copy()

    mapping = {
        date_column: "date",
        amount_column: "amount",
        description_column: "description",
        counterparty_column: "counterparty",
        category_column: "category",
        customer_column: "customer",
    }

    for source_col, target_col in mapping.items():
        if not source_col:
            continue

        source_col = str(source_col).strip()

        if source_col not in mapped.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Mapped column '{source_col}' was not found in the uploaded CSV.",
            )

        mapped[target_col] = mapped[source_col]

    return mapped


def validate_required_columns(df: pd.DataFrame):
    required = {"date", "amount"}
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column(s): {', '.join(missing)}. Your CSV must include at minimum 'date' and 'amount'.",
        )


def parse_amount_series(series: pd.Series) -> pd.Series:
    """Parse ordinary and Danish/European money strings into numeric values."""
    def parse_value(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        if isinstance(value, (int, float)):
            return float(value)

        raw = str(value).strip()
        if not raw:
            return float("nan")

        negative_parentheses = raw.startswith("(") and raw.endswith(")")
        trailing_minus = raw.endswith("-")
        cleaned = (
            raw.replace("\u00a0", "")
            .replace(" ", "")
            .replace("DKK", "")
            .replace("dkk", "")
            .replace("kr.", "")
            .replace("kr", "")
            .replace("(", "")
            .replace(")", "")
            .rstrip("-")
        )
        cleaned = "".join(char for char in cleaned if char.isdigit() or char in ",.+-")

        if "," in cleaned and "." in cleaned:
            decimal_separator = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
            thousands_separator = "." if decimal_separator == "," else ","
            cleaned = cleaned.replace(thousands_separator, "").replace(decimal_separator, ".")
        elif "," in cleaned:
            parts = cleaned.split(",")
            cleaned = "".join(parts) if len(parts[-1]) == 3 else "".join(parts[:-1]) + "." + parts[-1]
        elif cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = "".join(parts) if len(parts[-1]) == 3 else "".join(parts[:-1]) + "." + parts[-1]

        try:
            number = float(cleaned)
            return -abs(number) if negative_parentheses or trailing_minus else number
        except ValueError:
            return float("nan")

    return series.map(parse_value)


# ------------------------------------------------------------
# Formatting / labels
# ------------------------------------------------------------

def format_action_label_from_type(scenario_type: str | None) -> str:
    if scenario_type == "delay_outflows":
        return "Delaying outflows"
    if scenario_type == "accelerate_inflows":
        return "Accelerating inflows"
    if scenario_type == "timing_adjustment":
        return "Adjusting cash timing"
    if scenario_type == "cash_injection":
        return "Adding liquidity buffer"

    return "Scenario action"


def detect_action_label(forecast: dict) -> str:
    scenario = forecast.get("scenario")

    if not scenario:
        return "Scenario"

    return format_action_label_from_type(scenario.get("scenario_type"))


# ------------------------------------------------------------
# Transaction categorization v1
# ------------------------------------------------------------

CATEGORY_DEFINITIONS = {
    "customer_receivables": {
        "label": "Customer receipts",
        "keywords": [
            "customer", "client", "invoice paid", "invoice payment", "receivable",
            "sales", "revenue", "stripe payout", "shopify payout", "payout",
            "kunde", "kunden", "debitor", "indbetaling", "salg", "omsætning",
        ],
    },
    "supplier_payments": {
        "label": "Supplier payments",
        "keywords": [
            "supplier", "vendor", "invoice", "materials", "inventory", "stock",
            "wholesale", "supplies", "purchase", "faktura", "leverandør",
            "leverandor", "varekøb", "varekob", "materialer", "indkøb", "indkob",
        ],
    },
    "payroll": {
        "label": "Payroll / wages",
        "keywords": [
            "payroll", "salary", "wages", "employee", "staff", "pension",
            "lønn", "løn", "lon", "medarbejder", "personale", "feriepenge",
        ],
    },
    "tax_vat": {
        "label": "Tax / VAT",
        "keywords": [
            "tax", "vat", "moms", "skat", "skatt", "hmrc", "irs",
            "atp", "am-bidrag", "arbejdsmarkedsbidrag", "a-skat",
        ],
    },
    "rent_facilities": {
        "label": "Rent / facilities",
        "keywords": [
            "rent", "lease", "office", "workspace", "facility", "utilities",
            "electricity", "water", "heating", "husleje", "leje", "kontor",
            "el", "varme", "vand",
        ],
    },
    "software_tools": {
        "label": "Software / tools",
        "keywords": [
            "software", "subscription", "saas", "hosting", "cloud", "aws",
            "google", "microsoft", "adobe", "shopify", "slack", "notion",
            "figma", "stripe", "openai", "abonnement",
        ],
    },
    "loan_financing": {
        "label": "Loan / financing",
        "keywords": [
            "loan", "repayment", "interest", "financing", "credit", "bank fee",
            "rente", "afdrag", "lån", "laan", "kredit", "gebyr",
        ],
    },
    "marketing_sales": {
        "label": "Marketing / sales",
        "keywords": [
            "marketing", "ads", "advertising", "facebook", "meta", "google ads",
            "campaign", "seo", "agency", "reklame", "annoncer", "bureau",
        ],
    },
    "travel_expenses": {
        "label": "Travel / expenses",
        "keywords": [
            "travel", "hotel", "flight", "uber", "taxi", "train", "fuel",
            "parking", "rejse", "hotel", "fly", "tog", "brændstof", "braendstof",
            "parkering",
        ],
    },
}


def _clean_text_value(value) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def _category_label(category_key: str) -> str:
    if category_key in CATEGORY_DEFINITIONS:
        return CATEGORY_DEFINITIONS[category_key]["label"]

    if category_key == "other_income":
        return "Other income"

    if category_key == "other_outflow":
        return "Other outflows"

    return str(category_key).replace("_", " ").title()


INFLOW_CATEGORY_KEYS = {
    "customer_receivables",
}

OUTFLOW_CATEGORY_KEYS = {
    "supplier_payments",
    "payroll",
    "tax_vat",
    "rent_facilities",
    "software_tools",
    "loan_financing",
    "marketing_sales",
    "travel_expenses",
}


def _normalize_category_text(value: str) -> str:
    return (
        _clean_text_value(value)
        .strip()
        .lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _category_is_allowed_for_direction(category_key: str, amount: float) -> bool:
    if amount >= 0:
        return category_key in INFLOW_CATEGORY_KEYS

    return category_key in OUTFLOW_CATEGORY_KEYS


def categorize_transaction(row: pd.Series) -> tuple[str, str, str]:
    amount = float(row.get("amount", 0) or 0)

    explicit_category = _clean_text_value(row.get("category", ""))
    description = _clean_text_value(row.get("description", ""))
    counterparty = _clean_text_value(row.get("counterparty", ""))
    customer = _clean_text_value(row.get("customer", ""))

    # Important: include the mapped category text in the detection search.
    # Messy Danish files often provide the useful signal in Kategori rather than
    # Description/Counterparty. Without this, values like "Leverandør",
    # "Husleje", "Software", "Løn", or "Moms" fall back to Other.
    search_text = " ".join(
        part.lower()
        for part in [explicit_category, description, counterparty, customer]
        if part
    )

    allowed_category_keys = INFLOW_CATEGORY_KEYS if amount >= 0 else OUTFLOW_CATEGORY_KEYS

    if explicit_category:
        normalized_explicit = _normalize_category_text(explicit_category)

        # 1) Exact category key / label match, e.g. "supplier_payments" or
        # "Supplier payments".
        for key, definition in CATEGORY_DEFINITIONS.items():
            definition_label = _normalize_category_text(definition["label"])

            if normalized_explicit in {key, definition_label}:
                if _category_is_allowed_for_direction(key, amount):
                    return key, definition["label"], "provided"
                break

        # 2) Keyword match inside the provided category value, e.g. Danish
        # "Leverandør", "Husleje", "Løn", "Moms", "Abonnement".
        for key in allowed_category_keys:
            definition = CATEGORY_DEFINITIONS[key]
            for keyword in definition["keywords"]:
                if keyword.lower() in search_text:
                    return key, definition["label"], "provided"

    # If a positive transaction has a customer/counterparty/customer-like value,
    # classify it as customer receipts before falling back to Other income.
    if amount >= 0 and customer:
        return "customer_receivables", _category_label("customer_receivables"), "detected"

    for key in allowed_category_keys:
        definition = CATEGORY_DEFINITIONS[key]

        for keyword in definition["keywords"]:
            if keyword.lower() in search_text:
                return key, definition["label"], "detected"

    if amount >= 0:
        return "other_income", "Other income", "fallback"

    return "other_outflow", "Other outflows", "fallback"

def build_transaction_intelligence(
    df: pd.DataFrame,
    hist_end: pd.Timestamp,
    baseline_window_days: int,
) -> dict:
    def format_kr(value: float) -> str:
        rounded = int(round(float(value or 0)))
        return f"{rounded:,.0f}".replace(",", ".") + " kr."

    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return {
            "available": False,
            "summary": "Transaction intelligence could not be calculated because dated amount data was unavailable.",
        }

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
    work["amount"] = parse_amount_series(work["amount"])
    work = work.dropna(subset=["date", "amount"]).copy()

    if work.empty:
        return {
            "available": False,
            "summary": "Transaction intelligence could not be calculated because no valid transactions remained after cleaning.",
        }

    for optional_col in ["description", "category", "counterparty", "customer"]:
        if optional_col not in work.columns:
            work[optional_col] = ""

    categorized = work.apply(categorize_transaction, axis=1, result_type="expand")
    work["detected_category"] = categorized[0]
    work["detected_category_label"] = categorized[1]
    work["category_source"] = categorized[2]

    baseline_window_days = int(max(1, baseline_window_days or 30))
    recent_start = pd.to_datetime(hist_end) - pd.Timedelta(days=baseline_window_days - 1)
    recent = work[(work["date"] >= recent_start) & (work["date"] <= hist_end)].copy()

    if recent.empty:
        recent = work.copy()
        recent_start = work["date"].min()

    outflows = recent[recent["amount"] < 0].copy()
    inflows = recent[recent["amount"] > 0].copy()

    total_outflows = abs(float(outflows["amount"].sum())) if not outflows.empty else 0.0
    total_inflows = float(inflows["amount"].sum()) if not inflows.empty else 0.0

    def grouped_categories(frame: pd.DataFrame, direction: str) -> list[dict]:
        if frame.empty:
            return []

        grouped = (
            frame.groupby(["detected_category", "detected_category_label"], dropna=False)
            .agg(
                amount=("amount", "sum"),
                transaction_count=("amount", "count"),
            )
            .reset_index()
        )

        if direction == "outflow":
            grouped["absolute_amount"] = grouped["amount"].abs()
            denominator = max(total_outflows, 1.0)
        else:
            grouped["absolute_amount"] = grouped["amount"].clip(lower=0)
            denominator = max(total_inflows, 1.0)

        grouped = grouped.sort_values("absolute_amount", ascending=False)

        results = []
        for _, row in grouped.head(6).iterrows():
            absolute_amount = float(row["absolute_amount"] or 0)
            results.append(
                {
                    "category": row["detected_category"],
                    "label": row["detected_category_label"],
                    "amount": float(row["amount"] or 0),
                    "absolute_amount": absolute_amount,
                    "transaction_count": int(row["transaction_count"] or 0),
                    "share": float(absolute_amount / denominator) if denominator > 0 else 0.0,
                }
            )

        return results

    top_outflow_categories = grouped_categories(outflows, "outflow")
    top_inflow_categories = grouped_categories(inflows, "inflow")

    def top_counterparties(frame: pd.DataFrame) -> list[dict]:
        if frame.empty:
            return []

        temp = frame.copy()
        temp["counterparty_clean"] = temp["counterparty"].apply(_clean_text_value)
        temp.loc[temp["counterparty_clean"] == "", "counterparty_clean"] = temp["description"].apply(_clean_text_value)
        temp = temp[temp["counterparty_clean"] != ""]

        if temp.empty:
            return []

        grouped = (
            temp.groupby("counterparty_clean", dropna=False)
            .agg(amount=("amount", "sum"), transaction_count=("amount", "count"))
            .reset_index()
        )
        grouped["absolute_amount"] = grouped["amount"].abs()
        grouped = grouped.sort_values("absolute_amount", ascending=False)

        return [
            {
                "name": row["counterparty_clean"],
                "amount": float(row["amount"] or 0),
                "absolute_amount": float(row["absolute_amount"] or 0),
                "transaction_count": int(row["transaction_count"] or 0),
            }
            for _, row in grouped.head(5).iterrows()
        ]

    largest_outflow_counterparties = top_counterparties(outflows)
    largest_inflow_counterparties = top_counterparties(inflows)

    dominant_outflow = top_outflow_categories[0] if top_outflow_categories else None
    dominant_inflow = top_inflow_categories[0] if top_inflow_categories else None

    recommended_focus = []

    if dominant_outflow:
        recommended_focus.append(
            f"Review {dominant_outflow['label'].lower()} because it is the largest recent outflow category."
        )

    if dominant_inflow:
        recommended_focus.append(
            f"Monitor {dominant_inflow['label'].lower()} to ensure expected inflows arrive before the pressure window."
        )

    if dominant_outflow and dominant_inflow:
        recommended_focus.append(
            f"Compare {dominant_outflow['label'].lower()} timing against expected {dominant_inflow['label'].lower()} before the weakest cash period."
        )

    if not recommended_focus:
        recommended_focus.append("Add description, category, or counterparty columns to improve transaction intelligence.")

    if dominant_outflow:
        summary = (
            f"Recent cash pressure is mainly linked to {dominant_outflow['label'].lower()}, "
            f"representing {format_kr(dominant_outflow['absolute_amount'])} of outflows in the learning window."
        )
    elif dominant_inflow:
        summary = (
            f"Recent cash movement is mainly linked to {dominant_inflow['label'].lower()}, "
            f"representing {format_kr(dominant_inflow['absolute_amount'])} of inflows in the learning window."
        )
    else:
        summary = "No dominant transaction category was detected. Add descriptions, categories, or counterparties to improve the analysis."

    return {
        "available": True,
        "analysis_window_start": recent_start.date().isoformat(),
        "analysis_window_end": pd.to_datetime(hist_end).date().isoformat(),
        "total_inflows": total_inflows,
        "total_outflows": total_outflows,
        "dominant_outflow_category": dominant_outflow,
        "dominant_inflow_category": dominant_inflow,
        "top_outflow_categories": top_outflow_categories,
        "top_inflow_categories": top_inflow_categories,
        "largest_outflow_counterparties": largest_outflow_counterparties,
        "largest_inflow_counterparties": largest_inflow_counterparties,
        "recommended_focus": recommended_focus,
        "summary": summary,
    }


# ------------------------------------------------------------
# Forecast helpers
# ------------------------------------------------------------

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
        compare_start = (
            pd.to_datetime(historical_end) + pd.Timedelta(days=1)
        ).date().isoformat()

    if compare_start:
        baseline_lowest = _lowest_balance_from_date(
            baseline.get("series", []),
            compare_start,
        )
        scenario_lowest = _lowest_balance_from_date(
            scenario.get("series", []),
            compare_start,
        )
        scenario_lowest_date = _lowest_balance_date_from_date(
            scenario.get("series", []),
            compare_start,
        )
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


def generate_decision_impact(
    baseline: dict,
    scenario: dict,
    action_label: str,
    currency_symbol: str,
) -> dict:
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


# ------------------------------------------------------------
# Executive summary
# ------------------------------------------------------------

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


def build_selected_scenario_summary(forecast: dict, decision_impact: dict | None):
    if not forecast.get("scenario"):
        return None

    scenario = forecast["scenario"]

    lowest_balance = (
        decision_impact.get("scenario_lowest_balance")
        if decision_impact
        else scenario.get("lowest_balance")
    )

    lowest_balance_date = (
        decision_impact.get("scenario_lowest_balance_date")
        if decision_impact
        else scenario.get("lowest_balance_date")
    )

    improvement = (
        decision_impact.get("balance_improvement")
        if decision_impact
        else None
    )

    action_label = (
        decision_impact.get("action")
        if decision_impact
        else format_action_label_from_type(scenario.get("scenario_type"))
    )

    lowest_value = float(lowest_balance or 0)
    improvement_value = float(improvement or 0)
    first_negative_date = scenario.get("first_negative_date")
    days_until_negative = scenario.get("days_until_negative")
    risk = scenario.get("risk") or {}
    risk_level = risk.get("level")
    failure = scenario.get("liquidity_failure") or {}
    failure_probability = failure.get("probability")

    try:
        failure_probability_value = float(failure_probability) if failure_probability is not None else None
    except Exception:
        failure_probability_value = None

    lowest_text = f"{round(lowest_value, 2)}"
    lowest_date_text = f" on {lowest_balance_date}" if lowest_balance_date else ""
    improvement_text = f"{round(improvement_value, 2)} kr."

    if first_negative_date or lowest_value < 0:
        day_text = ""
        if days_until_negative is not None:
            day_text = f" in {days_until_negative} day{'s' if int(days_until_negative) != 1 else ''}"

        return {
            "headline": "The selected scenario improves liquidity, but residual risk remains.",
            "details": (
                f"Cash is still projected to turn negative{day_text}"
                f"{f' on {first_negative_date}' if first_negative_date else ''}. "
                f"The lowest projected balance is {lowest_text}{lowest_date_text}. "
                f"{action_label} improves worst-case liquidity by {improvement_text}, "
                f"but additional protection may still be required."
            ),
            "suggestions": [
                "Treat this scenario as partial protection and test additional actions before relying on it operationally."
            ],
        }

    if risk_level in {"Watch", "High", "Critical"} or (
        failure_probability_value is not None and failure_probability_value >= 0.30
    ):
        probability_text = ""
        if failure_probability_value is not None:
            probability_text = f" Simulation still estimates a {round(failure_probability_value * 100)}% liquidity failure probability."

        return {
            "headline": "The selected scenario improves liquidity, but downside risk should still be monitored.",
            "details": (
                f"No negative cash position is projected within the current forecast horizon. "
                f"The lowest projected balance is {lowest_text}{lowest_date_text}. "
                f"{action_label} improves worst-case liquidity by {improvement_text}."
                f"{probability_text}"
            ),
            "suggestions": [
                "Continue monitoring the weakest forecast period and compare the action against operational feasibility."
            ],
        }

    return {
        "headline": "The selected scenario strengthens short-term liquidity.",
        "details": (
            f"No negative cash position is projected within the current forecast horizon. "
            f"The lowest projected balance is {lowest_text}{lowest_date_text}. "
            f"{action_label} improves worst-case liquidity by {improvement_text}."
        ),
        "suggestions": [
            "The tested scenario improves downside liquidity and may be worth operational follow-up."
        ],
    }


# ------------------------------------------------------------
# Risk score
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Risk drivers
# ------------------------------------------------------------

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
            window_days = (
                largest_negatives["date"].max() - largest_negatives["date"].min()
            ).days
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
            "Scenario testing shows that additional liquidity materially improves the projected cash floor, indicating a need for added liquidity support."
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


# ------------------------------------------------------------
# Liquidity Pressure Breakdown v1
# ------------------------------------------------------------

def build_pressure_drivers(forecast: dict, window_before_days: int = 14, window_after_days: int = 0) -> dict:
    def format_kr(value: float) -> str:
        if value is None:
            return "0 kr."
        rounded = int(round(float(value)))
        return f"{rounded:,.0f}".replace(",", ".") + " kr."

    def format_date(value) -> str:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return ""
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        return f"{ts.day:02d} {month_names[ts.month - 1]} {ts.year}"

    series = forecast.get("series", []) or []
    warning_threshold = float(forecast.get("warning_threshold", 0) or 0)
    historical_end_date = forecast.get("historical_end_date")

    if not series:
        return {
            "available": False,
            "summary": "Liquidity pressure breakdown could not be calculated because no forecast series was available.",
        }

    df = pd.DataFrame(series).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["net_flow"] = pd.to_numeric(df["net_flow"], errors="coerce").fillna(0.0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date"]).sort_values("date")

    if df.empty:
        return {
            "available": False,
            "summary": "Liquidity pressure breakdown could not be calculated because no valid dated forecast rows were available.",
        }

    # Prefer the future forecast period, not the historical period
    future_df = df.copy()
    if historical_end_date:
        future_start = pd.to_datetime(historical_end_date, errors="coerce") + pd.Timedelta(days=1)
        if not pd.isna(future_start):
            future_df = df[df["date"] >= future_start].copy()

    if future_df.empty:
        future_df = df.copy()

    weakest_row = future_df.loc[future_df["balance"].idxmin()]
    weakest_date = weakest_row["date"]
    weakest_balance = float(weakest_row["balance"])

    window_start = weakest_date - pd.Timedelta(days=window_before_days)
    window_end = weakest_date + pd.Timedelta(days=window_after_days)

    available_start = future_df["date"].min()
    available_end = future_df["date"].max()

    if window_start < available_start:
        window_start = available_start

    if window_end > available_end:
        window_end = available_end

    pressure_window = future_df[
        (future_df["date"] >= window_start) & (future_df["date"] <= window_end)
    ].copy()

    if pressure_window.empty:
        pressure_window = future_df.copy()

    total_inflows = float(pressure_window.loc[pressure_window["net_flow"] > 0, "net_flow"].sum())
    total_outflows = abs(float(pressure_window.loc[pressure_window["net_flow"] < 0, "net_flow"].sum()))
    net_pressure = float(total_inflows - total_outflows)

    outflow_days = pressure_window[pressure_window["net_flow"] < 0].copy()
    largest_outflow_days = []

    if not outflow_days.empty:
        largest_outflow_days = [
            {
                "date": row["date"].date().isoformat(),
                "amount": float(row["net_flow"]),
                "absolute_amount": abs(float(row["net_flow"])),
            }
            for _, row in outflow_days.nsmallest(5, "net_flow").iterrows()
        ]

    minimum_buffer_to_zero = max(0.0, -weakest_balance)
    minimum_buffer_to_safe_threshold = max(0.0, warning_threshold - weakest_balance)

    if net_pressure < 0:
        summary = (
            f"Liquidity pressure is concentrated between {format_date(window_start)} and "
            f"{format_date(window_end)}. During this period, projected outflows exceed inflows "
            f"by {format_kr(abs(net_pressure))}. The weakest projected cash position occurs on "
            f"{format_date(weakest_date)}."
        )
    else:
        summary = (
            f"No concentrated pressure period is detected. The weakest projected cash position "
            f"occurs on {format_date(weakest_date)}, but inflows cover outflows within the main "
            f"pressure window."
        )

    recommended_focus = []

    if minimum_buffer_to_zero > 0:
        recommended_focus.append(
            f"At least {format_kr(minimum_buffer_to_zero)} is needed to avoid falling below zero."
        )

    if minimum_buffer_to_safe_threshold > 0:
        recommended_focus.append(
            f"At least {format_kr(minimum_buffer_to_safe_threshold)} is needed to remain above the safe cash threshold."
        )

    if total_outflows > total_inflows:
        recommended_focus.append("Review large outflows before the weakest cash date.")
        recommended_focus.append("Test whether customer payments can be accelerated into the pressure window.")

    transaction_intelligence = forecast.get("transaction_intelligence") or {}
    category_pressure = transaction_intelligence.get("top_outflow_categories", []) or []
    dominant_pressure_category = transaction_intelligence.get("dominant_outflow_category")

    if dominant_pressure_category:
        recommended_focus.append(
            f"Investigate {str(dominant_pressure_category.get('label', 'the dominant outflow category')).lower()} because it is the largest recent outflow category."
        )

    if not recommended_focus:
        recommended_focus.append("Continue monitoring the weakest forecast period.")

    pressure_source_summary = None
    if dominant_pressure_category:
        pressure_source_summary = (
            f"The largest recent outflow category is {dominant_pressure_category.get('label')}, "
            f"which helps explain what may be driving projected liquidity pressure."
        )

    return {
        "available": True,
        "weakest_date": weakest_date.date().isoformat(),
        "weakest_balance": weakest_balance,
        "pressure_window_start": window_start.date().isoformat(),
        "pressure_window_end": window_end.date().isoformat(),
        "total_inflows": total_inflows,
        "total_outflows": total_outflows,
        "net_pressure": net_pressure,
        "largest_outflow_days": largest_outflow_days,
        "minimum_buffer_to_zero": minimum_buffer_to_zero,
        "minimum_buffer_to_safe_threshold": minimum_buffer_to_safe_threshold,
        "category_pressure": category_pressure[:5],
        "dominant_pressure_category": dominant_pressure_category,
        "pressure_source_summary": pressure_source_summary,
        "summary": summary,
        "recommended_focus": list(dict.fromkeys(recommended_focus)),
    }


# ------------------------------------------------------------
# Minimum Required Action v1
# ------------------------------------------------------------

def build_minimum_required_action(forecast: dict, max_test_days: int = 30) -> dict:
    def format_kr(value: float) -> str:
        if value is None:
            return "0 kr."
        rounded = int(round(float(value)))
        return f"{rounded:,.0f}".replace(",", ".") + " kr."

    def format_days(value: int | None) -> str:
        if value is None:
            return "—"
        return f"{int(value)} day{'s' if int(value) != 1 else ''}"

    series = forecast.get("series", []) or []
    historical_end_date = forecast.get("historical_end_date")
    warning_threshold = float(forecast.get("warning_threshold", 0) or 0)
    opening_balance = float(forecast.get("opening_balance", 0) or 0)

    if not series:
        return {
            "available": False,
            "summary": "Minimum required action could not be calculated because no forecast series was available.",
        }

    df = pd.DataFrame(series).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["net_flow"] = pd.to_numeric(df["net_flow"], errors="coerce").fillna(0.0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date"]).sort_values("date")

    if df.empty:
        return {
            "available": False,
            "summary": "Minimum required action could not be calculated because no valid dated forecast rows were available.",
        }

    future_df = df.copy()
    compare_start = None

    if historical_end_date:
        future_start = pd.to_datetime(historical_end_date, errors="coerce") + pd.Timedelta(days=1)
        if not pd.isna(future_start):
            compare_start = future_start
            future_df = df[df["date"] >= future_start].copy()

    if future_df.empty:
        future_df = df.copy()
        compare_start = df["date"].min()

    weakest_row = future_df.loc[future_df["balance"].idxmin()]
    weakest_balance = float(weakest_row["balance"])
    weakest_date = weakest_row["date"].date().isoformat()

    buffer_to_avoid_negative = max(0.0, -weakest_balance)
    buffer_to_reach_safe_cash = max(0.0, warning_threshold - weakest_balance)

    daily = pd.Series(
        data=df["net_flow"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["date"]),
    ).sort_index()

    def lowest_balance_after_start(adjusted_daily: pd.Series) -> float:
        balance = opening_balance + adjusted_daily.cumsum()

        if compare_start is not None:
            future_balance = balance[balance.index >= compare_start]
            if not future_balance.empty:
                return float(future_balance.min())

        return float(balance.min())

    minimum_supplier_delay_days = None
    minimum_customer_acceleration_days = None

    for days in range(1, int(max_test_days) + 1):
        delayed = _shift_outflows_forward(daily, days)
        delayed_lowest = lowest_balance_after_start(delayed)

        if delayed_lowest >= 0:
            minimum_supplier_delay_days = days
            break

    for days in range(1, int(max_test_days) + 1):
        accelerated = _shift_inflows_earlier(daily, days)
        accelerated_lowest = lowest_balance_after_start(accelerated)

        if accelerated_lowest >= 0:
            minimum_customer_acceleration_days = days
            break

    practical_options = []

    if buffer_to_avoid_negative > 0:
        practical_options.append({
            "type": "liquidity_buffer",
            "label": "Add liquidity buffer",
            "value": buffer_to_avoid_negative,
            "unit": "kr.",
            "display_value": format_kr(buffer_to_avoid_negative),
        })

    if minimum_supplier_delay_days is not None:
        practical_options.append({
            "type": "supplier_delay",
            "label": "Delay supplier payments",
            "value": minimum_supplier_delay_days,
            "unit": "days",
            "display_value": format_days(minimum_supplier_delay_days),
        })

    if minimum_customer_acceleration_days is not None:
        practical_options.append({
            "type": "customer_acceleration",
            "label": "Accelerate customer payments",
            "value": minimum_customer_acceleration_days,
            "unit": "days",
            "display_value": format_days(minimum_customer_acceleration_days),
        })

    # Prefer the smallest direct buffer if there is an actual shortfall.
    # If no shortfall exists, recommend monitoring or staying above the safe cash threshold.
    if buffer_to_avoid_negative > 0:
        recommended_action = practical_options[0] if practical_options else None
        summary = (
            f"The smallest direct protection needed to avoid a projected shortfall is "
            f"{format_kr(buffer_to_avoid_negative)}. To stay above the safe cash threshold, "
            f"the forecast requires {format_kr(buffer_to_reach_safe_cash)}."
        )
    elif buffer_to_reach_safe_cash > 0:
        recommended_action = {
            "type": "liquidity_buffer",
            "label": "Add liquidity buffer",
            "value": buffer_to_reach_safe_cash,
            "unit": "kr.",
            "display_value": format_kr(buffer_to_reach_safe_cash),
        }
        summary = (
            f"No negative cash position is projected, but {format_kr(buffer_to_reach_safe_cash)} "
            f"is needed to stay above the safe cash threshold."
        )
    else:
        recommended_action = {
            "type": "monitoring",
            "label": "No immediate action required",
            "value": 0,
            "unit": "kr.",
            "display_value": "0 kr.",
        }
        summary = (
            "No minimum corrective action is required within the current forecast horizon. "
            "Cash remains above zero and above the safe cash threshold."
        )

    return {
        "available": True,
        "weakest_date": weakest_date,
        "weakest_balance": weakest_balance,
        "buffer_to_avoid_negative": buffer_to_avoid_negative,
        "buffer_to_reach_safe_cash": buffer_to_reach_safe_cash,
        "minimum_supplier_delay_days": minimum_supplier_delay_days,
        "minimum_customer_acceleration_days": minimum_customer_acceleration_days,
        "recommended_action": recommended_action,
        "practical_options": practical_options,
        "summary": summary,
    }


# ------------------------------------------------------------
# Metrics and scenario transformations
# ------------------------------------------------------------

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


def _apply_cash_injection(
    daily: pd.Series,
    amount: float,
    injection_date: pd.Timestamp,
) -> pd.Series:
    amount = float(max(0, amount))

    if amount <= 0:
        return daily.copy()

    adjusted = daily.copy()

    if injection_date in adjusted.index:
        adjusted.loc[injection_date] = float(adjusted.loc[injection_date]) + amount

    return adjusted


# ------------------------------------------------------------
# Forecast engine
# ------------------------------------------------------------

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

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["amount"] = parse_amount_series(df["amount"])
    df = df.dropna(subset=["date", "amount"]).copy()

    if df.empty:
        return {
            "error": "After cleaning, no valid rows remained. Check that dates are valid and amounts are numeric."
        }

    daily_hist = df.groupby(df["date"].dt.normalize())["amount"].sum().sort_index()

    hist_start = daily_hist.index.min()
    hist_end = daily_hist.index.max()

    horizon_days = int(max(0, horizon_days))
    baseline_window_days = int(max(1, baseline_window_days))
    outflow_shift_days = int(max(0, outflow_shift_days))
    inflow_shift_days = int(max(0, inflow_shift_days))
    cash_injection_amount = float(max(0, cash_injection_amount))

    transaction_intelligence = build_transaction_intelligence(
        df=df,
        hist_end=hist_end,
        baseline_window_days=baseline_window_days,
    )

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
        "transaction_intelligence": transaction_intelligence,
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
    baseline["pressure_drivers"] = build_pressure_drivers(baseline)
    baseline["minimum_required_action"] = build_minimum_required_action(baseline)

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
            scenario_daily = _apply_cash_injection(
                scenario_daily,
                cash_injection_amount,
                injection_date,
            )
            scenario_type = "cash_injection"

        else:
            if outflow_shift_days > 0:
                scenario_daily = _shift_outflows_forward(scenario_daily, outflow_shift_days)

            if inflow_shift_days > 0:
                scenario_daily = _shift_inflows_earlier(scenario_daily, inflow_shift_days)

            if cash_injection_amount > 0:
                injection_date = hist_end + pd.Timedelta(days=1)
                scenario_daily = _apply_cash_injection(
                    scenario_daily,
                    cash_injection_amount,
                    injection_date,
                )

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
            "transaction_intelligence": transaction_intelligence,
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
        scenario["pressure_drivers"] = build_pressure_drivers(scenario)
        scenario["minimum_required_action"] = build_minimum_required_action(scenario)

        compare_start = (hist_end + pd.Timedelta(days=1)).date().isoformat()

        baseline_compare_lowest = _lowest_balance_from_date(
            baseline.get("series", []),
            compare_start,
        )

        scenario_compare_lowest = _lowest_balance_from_date(
            scenario.get("series", []),
            compare_start,
        )

        scenario_compare_lowest_date = _lowest_balance_date_from_date(
            scenario.get("series", []),
            compare_start,
        )

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


# ------------------------------------------------------------
# Scenario ranking
# ------------------------------------------------------------

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

    baseline_lowest = _lowest_balance_from_date(
        baseline.get("series", []),
        compare_start,
    )

    if baseline_lowest is None:
        baseline_lowest = float(baseline.get("lowest_balance", 0) or 0)

    options = []

    def add_option(
        result: dict,
        key: str,
        label: str,
        outflow: int,
        inflow: int,
        injection: float,
    ):
        if not result.get("scenario"):
            return

        scen = result["scenario"]

        scenario_lowest = _lowest_balance_from_date(
            scen.get("series", []),
            compare_start,
        )

        scenario_lowest_date = _lowest_balance_date_from_date(
            scen.get("series", []),
            compare_start,
        )

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
            "Add liquidity buffer",
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



# ------------------------------------------------------------
# Forecast history database / workspace v1
# PostgreSQL-ready through SQLAlchemy
# ------------------------------------------------------------

DB_PATH = os.getenv("PREVIA_DB_PATH", "previa_history.db")
DEFAULT_COMPANY_ID = os.getenv("PREVIA_DEFAULT_COMPANY_ID", "demo-company")
DEFAULT_COMPANY_NAME = os.getenv("PREVIA_DEFAULT_COMPANY_NAME", "Demo Company")


def _normalize_database_url(raw_url: str | None) -> str:
    """Use DATABASE_URL when provided, otherwise local SQLite.

    Render/Railway/Supabase commonly provide Postgres URLs. Some providers use
    postgres:// while SQLAlchemy expects postgresql://, so normalize it here.
    """
    url = (raw_url or "").strip()

    if not url:
        return f"sqlite:///{DB_PATH}"

    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)

    return url


DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL"))
IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    pool_pre_ping=True,
    future=True,
)


def normalize_company_id(company_id: str | None = None) -> str:
    cleaned = str(company_id or "").strip()
    return cleaned or DEFAULT_COMPANY_ID


def _db_label() -> str:
    if IS_SQLITE:
        return f"SQLite ({DB_PATH})"
    return "PostgreSQL / external DATABASE_URL"


def _table_columns(table_name: str) -> set[str]:
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            return set()
        return {column["name"] for column in inspector.get_columns(table_name)}
    except Exception:
        return set()


def _mapping_to_dict(row):
    if row is None:
        return None
    return dict(row)


def init_history_db():
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL DEFAULT 'demo-company',
                    created_at TEXT NOT NULL,
                    filename TEXT,
                    run_type TEXT NOT NULL,
                    opening_balance REAL,
                    horizon_days INTEGER,
                    baseline_window_days INTEGER,
                    warning_threshold REAL,
                    outflow_shift_days INTEGER,
                    inflow_shift_days INTEGER,
                    cash_injection_amount REAL,
                    risk_score REAL,
                    risk_level TEXT,
                    failure_probability REAL,
                    lowest_balance REAL,
                    lowest_balance_date TEXT,
                    first_negative_date TEXT,
                    first_warning_date TEXT,
                    summary_headline TEXT,
                    summary_details TEXT,
                    raw_response_json TEXT
                )
                """
            )
        )

        existing_company = conn.execute(
            text("SELECT id FROM companies WHERE id = :id LIMIT 1"),
            {"id": DEFAULT_COMPANY_ID},
        ).mappings().first()

        if existing_company is None:
            conn.execute(
                text(
                    """
                    INSERT INTO companies (id, name, created_at)
                    VALUES (:id, :name, :created_at)
                    """
                ),
                {
                    "id": DEFAULT_COMPANY_ID,
                    "name": DEFAULT_COMPANY_NAME,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    # Lightweight migration for old local SQLite databases created before
    # workspace/full-result support. This also keeps the schema compatible
    # when DATABASE_URL later points to PostgreSQL.
    forecast_columns = _table_columns("forecast_runs")

    with engine.begin() as conn:
        if "company_id" not in forecast_columns:
            conn.execute(text("ALTER TABLE forecast_runs ADD COLUMN company_id TEXT"))
            conn.execute(
                text(
                    """
                    UPDATE forecast_runs
                    SET company_id = :company_id
                    WHERE company_id IS NULL OR TRIM(company_id) = ''
                    """
                ),
                {"company_id": DEFAULT_COMPANY_ID},
            )

        if "raw_response_json" not in forecast_columns:
            conn.execute(text("ALTER TABLE forecast_runs ADD COLUMN raw_response_json TEXT"))

        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_forecast_runs_created_at ON forecast_runs(created_at)")
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_forecast_runs_company_created_at ON forecast_runs(company_id, created_at)")
        )


def ensure_company(company_id: str | None = None, name: str | None = None) -> dict:
    company_id = normalize_company_id(company_id)
    company_name = str(name or "").strip() or (
        DEFAULT_COMPANY_NAME if company_id == DEFAULT_COMPANY_ID else company_id
    )

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id, name, created_at FROM companies WHERE id = :id LIMIT 1"),
            {"id": company_id},
        ).mappings().first()

        if existing is None:
            conn.execute(
                text(
                    """
                    INSERT INTO companies (id, name, created_at)
                    VALUES (:id, :name, :created_at)
                    """
                ),
                {
                    "id": company_id,
                    "name": company_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            existing = conn.execute(
                text("SELECT id, name, created_at FROM companies WHERE id = :id LIMIT 1"),
                {"id": company_id},
            ).mappings().first()

    return _mapping_to_dict(existing) or {"id": company_id, "name": company_name}


def list_companies() -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, name, created_at FROM companies ORDER BY created_at ASC")
        ).mappings().all()
    return [dict(row) for row in rows]


class WorkspaceCompanyCreate(BaseModel):
    id: str | None = None
    name: str


def _to_jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # Support numpy scalar values without importing numpy directly.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def _json_safe(value):
    return json.dumps(_to_jsonable(value), ensure_ascii=False)


def save_forecast_run_to_history(
    *,
    company_id: str | None = None,
    filename: str | None,
    response_payload: dict,
    opening_balance: float,
    horizon_days: int,
    baseline_window_days: int,
    warning_threshold: float,
    outflow_shift_days: int,
    inflow_shift_days: int,
    cash_injection_amount: float,
):
    """Save a compact forecast-run summary for history views."""
    forecast = response_payload.get("forecast") or {}
    summary = response_payload.get("summary") or {}

    run_type = "scenario" if forecast.get("scenario") else "baseline"
    summary_target = forecast.get("scenario") if forecast.get("scenario") else forecast

    if not isinstance(summary_target, dict):
        summary_target = {}

    risk = summary_target.get("risk") or {}
    liquidity_failure = summary_target.get("liquidity_failure") or {}
    decision_impact = response_payload.get("decision_impact") or {}

    probability = liquidity_failure.get("probability")
    if probability is not None:
        try:
            probability = float(probability)
        except Exception:
            probability = None

    # For scenario runs, history should use the same future comparison window
    # shown in the saved scenario summary.
    history_lowest_balance = summary_target.get("lowest_balance")
    history_lowest_balance_date = summary_target.get("lowest_balance_date")

    if run_type == "scenario" and isinstance(decision_impact, dict):
        history_lowest_balance = decision_impact.get(
            "scenario_lowest_balance",
            history_lowest_balance,
        )
        history_lowest_balance_date = decision_impact.get(
            "scenario_lowest_balance_date",
            history_lowest_balance_date,
        )

    company_id = normalize_company_id(company_id)
    ensure_company(company_id)

    record = {
        "id": str(uuid.uuid4()),
        "company_id": company_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "run_type": run_type,
        "opening_balance": float(opening_balance or 0),
        "horizon_days": int(horizon_days or 0),
        "baseline_window_days": int(baseline_window_days or 0),
        "warning_threshold": float(warning_threshold or 0),
        "outflow_shift_days": int(outflow_shift_days or 0),
        "inflow_shift_days": int(inflow_shift_days or 0),
        "cash_injection_amount": float(cash_injection_amount or 0),
        "risk_score": risk.get("score"),
        "risk_level": risk.get("level"),
        "failure_probability": probability,
        "lowest_balance": history_lowest_balance,
        "lowest_balance_date": history_lowest_balance_date,
        "first_negative_date": summary_target.get("first_negative_date"),
        "first_warning_date": summary_target.get("first_warning_date"),
        "summary_headline": summary.get("headline"),
        "summary_details": summary.get("details"),
        "raw_response_json": _json_safe(response_payload),
    }

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO forecast_runs (
                    id,
                    company_id,
                    created_at,
                    filename,
                    run_type,
                    opening_balance,
                    horizon_days,
                    baseline_window_days,
                    warning_threshold,
                    outflow_shift_days,
                    inflow_shift_days,
                    cash_injection_amount,
                    risk_score,
                    risk_level,
                    failure_probability,
                    lowest_balance,
                    lowest_balance_date,
                    first_negative_date,
                    first_warning_date,
                    summary_headline,
                    summary_details,
                    raw_response_json
                )
                VALUES (
                    :id,
                    :company_id,
                    :created_at,
                    :filename,
                    :run_type,
                    :opening_balance,
                    :horizon_days,
                    :baseline_window_days,
                    :warning_threshold,
                    :outflow_shift_days,
                    :inflow_shift_days,
                    :cash_injection_amount,
                    :risk_score,
                    :risk_level,
                    :failure_probability,
                    :lowest_balance,
                    :lowest_balance_date,
                    :first_negative_date,
                    :first_warning_date,
                    :summary_headline,
                    :summary_details,
                    :raw_response_json
                )
                """
            ),
            record,
        )

    return record["id"]


def list_forecast_history(limit: int = 20, company_id: str | None = None):
    limit = max(1, min(int(limit or 20), 100))
    company_id = normalize_company_id(company_id)
    ensure_company(company_id)

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    id,
                    company_id,
                    created_at,
                    filename,
                    run_type,
                    opening_balance,
                    horizon_days,
                    baseline_window_days,
                    warning_threshold,
                    outflow_shift_days,
                    inflow_shift_days,
                    cash_injection_amount,
                    risk_score,
                    risk_level,
                    failure_probability,
                    lowest_balance,
                    lowest_balance_date,
                    first_negative_date,
                    first_warning_date,
                    summary_headline,
                    summary_details
                FROM forecast_runs
                WHERE company_id = :company_id
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"company_id": company_id, "limit": limit},
        ).mappings().all()

    return [dict(row) for row in rows]


def get_forecast_history_run(run_id: str, company_id: str | None = None):
    """Return one saved forecast run for the clickable History detail view."""
    company_id = normalize_company_id(company_id)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                    id,
                    company_id,
                    created_at,
                    filename,
                    run_type,
                    opening_balance,
                    horizon_days,
                    baseline_window_days,
                    warning_threshold,
                    outflow_shift_days,
                    inflow_shift_days,
                    cash_injection_amount,
                    risk_score,
                    risk_level,
                    failure_probability,
                    lowest_balance,
                    lowest_balance_date,
                    first_negative_date,
                    first_warning_date,
                    summary_headline,
                    summary_details,
                    raw_response_json
                FROM forecast_runs
                WHERE id = :run_id AND company_id = :company_id
                LIMIT 1
                """
            ),
            {"run_id": run_id, "company_id": company_id},
        ).mappings().first()

    if row is None:
        return None

    item = dict(row)
    raw_response_json = item.pop("raw_response_json", None)

    if raw_response_json:
        try:
            item["raw_response"] = json.loads(raw_response_json)
        except Exception:
            item["raw_response"] = None
    else:
        item["raw_response"] = None

    return item


def clear_forecast_history_for_company(company_id: str | None = None):
    company_id = normalize_company_id(company_id)

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM forecast_runs WHERE company_id = :company_id"),
            {"company_id": company_id},
        )

    return company_id

# ------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------

app = FastAPI()

APP_ENV = os.getenv("PREVIA_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"
PILOT_PASSWORD = os.getenv("PILOT_PASSWORD", "").strip()

# These routes must remain reachable without pilot credentials. Render uses
# /health for service checks, and the frontend uses /pilot-access/verify before
# unlocking the private application.
PUBLIC_PATHS = {"/health", "/pilot-access/verify"}


@app.middleware("http")
async def require_pilot_access(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    # Local development remains convenient when no password is configured.
    # Production fails closed if PILOT_PASSWORD was accidentally omitted.
    if not PILOT_PASSWORD:
        if IS_PRODUCTION:
            return JSONResponse(
                status_code=503,
                content={"detail": "Pilot access is not configured."},
            )
        return await call_next(request)

    supplied_password = request.headers.get("X-Previa-Pilot-Key", "")
    if not secrets.compare_digest(supplied_password, PILOT_PASSWORD):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid pilot access password."},
        )

    return await call_next(request)

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


@app.post("/pilot-access/verify")
def verify_pilot_access(request: Request):
    if not PILOT_PASSWORD:
        if IS_PRODUCTION:
            raise HTTPException(status_code=503, detail="Pilot access is not configured.")
        return {"status": "ok", "mode": "development"}

    supplied_password = request.headers.get("X-Previa-Pilot-Key", "")
    if not secrets.compare_digest(supplied_password, PILOT_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid pilot access password.")

    return {"status": "ok"}


@app.get("/workspace")
def current_workspace(company_id: str | None = None):
    company = ensure_company(company_id)
    return {"company": company}


@app.get("/companies")
def companies():
    return {"items": list_companies(), "default_company_id": DEFAULT_COMPANY_ID}


@app.post("/companies")
def create_company(payload: WorkspaceCompanyCreate):
    requested_id = payload.id or payload.name.lower().strip().replace(" ", "-")
    company = ensure_company(requested_id, payload.name)
    return {"company": company}


@app.get("/forecast-history")
def forecast_history(limit: int = 20, company_id: str | None = None):
    company = ensure_company(company_id)
    return {
        "company": company,
        "items": list_forecast_history(limit=limit, company_id=company["id"]),
    }


@app.get("/forecast-history/{run_id}")
def forecast_history_detail(run_id: str, company_id: str | None = None):
    item = get_forecast_history_run(run_id, company_id=company_id)

    if item is None:
        raise HTTPException(status_code=404, detail="Forecast run not found for this company.")

    return {"item": item}


@app.get("/forecast-history/{run_id}/result")
def forecast_history_result(run_id: str, company_id: str | None = None):
    item = get_forecast_history_run(run_id, company_id=company_id)

    if item is None:
        raise HTTPException(status_code=404, detail="Forecast run not found for this company.")

    raw_response = item.get("raw_response")

    if not isinstance(raw_response, dict) or not raw_response.get("forecast"):
        raise HTTPException(
            status_code=409,
            detail="This saved run does not include a reloadable forecast result. Run a fresh forecast after the latest update.",
        )

    return {"result": raw_response, "item": item}


@app.delete("/forecast-history")
def clear_forecast_history(company_id: str | None = None):
    """Clear forecast history for one company during MVP testing."""
    if IS_PRODUCTION:
        raise HTTPException(
            status_code=403,
            detail="Deleting forecast history is disabled in production.",
        )

    cleared_company_id = clear_forecast_history_for_company(company_id)
    return {"status": "cleared", "company_id": cleared_company_id}


# ------------------------------------------------------------
# Data quality
# ------------------------------------------------------------

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
        warnings.append(
            f"Missing data: {missing_cells} empty cells ({missing_ratio:.1%} of all fields)."
        )

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
                    warnings.append(
                        f"Missing months in range: {len(missing_months)} {'month' if len(missing_months) == 1 else 'months'}."
                    )
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
        warnings.append(
            f"Missing days in range: {len(missing_days)} {'day' if len(missing_days) == 1 else 'days'}."
        )

    amount_col = "amount" if "amount" in df.columns else None

    if amount_col:
        amounts = parse_amount_series(df[amount_col])
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


# ------------------------------------------------------------
# Forecast upload endpoint
# ------------------------------------------------------------

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
    company_id: str | None = None,
    date_column: str | None = None,
    amount_column: str | None = None,
    description_column: str | None = None,
    counterparty_column: str | None = None,
    category_column: str | None = None,
    customer_column: str | None = None,
):
    print("📥 Received upload request")
    content = await file.read()

    def process():
        df = parse_uploaded_csv_bytes(content)
        df = apply_column_mapping(
            df,
            date_column=date_column,
            amount_column=amount_column,
            description_column=description_column,
            counterparty_column=counterparty_column,
            category_column=category_column,
            customer_column=customer_column,
        )
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
            "pressure_drivers": None,
            "minimum_required_action": None,
            "transaction_intelligence": None,
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

        forecast["scenario"]["pressure_drivers"] = build_pressure_drivers(
            forecast["scenario"]
        )

        forecast["scenario"]["minimum_required_action"] = build_minimum_required_action(
            forecast["scenario"]
        )

    if forecast.get("scenario"):
        summary = build_selected_scenario_summary(forecast, decision_impact)
        summary_target = forecast["scenario"]
    else:
        summary = build_executive_summary(forecast)
        summary_target = forecast

    print("📤 Returning forecast response")

    response_payload = {
        "company": ensure_company(company_id),
        "quality": quality,
        "forecast": forecast,
        "summary": summary,
        "decision_impact": decision_impact,
        "risk": summary_target.get("risk"),
        "risk_drivers": summary_target.get("risk_drivers"),
        "pressure_drivers": summary_target.get("pressure_drivers"),
        "minimum_required_action": summary_target.get("minimum_required_action"),
        "transaction_intelligence": summary_target.get("transaction_intelligence"),
        "scenario_ranking": ranking,
    }

    try:
        history_id = save_forecast_run_to_history(
            company_id=company_id,
            filename=file.filename,
            response_payload=response_payload,
            opening_balance=opening_balance,
            horizon_days=horizon_days,
            baseline_window_days=baseline_window_days,
            warning_threshold=warning_threshold,
            outflow_shift_days=outflow_shift_days,
            inflow_shift_days=inflow_shift_days,
            cash_injection_amount=cash_injection_amount,
        )
        response_payload["history_id"] = history_id
    except Exception as e:
        print(f"⚠️ Could not save forecast history: {e}")
        response_payload["history_id"] = None

    return response_payload


# ------------------------------------------------------------
# Company endpoints / alerts
# ------------------------------------------------------------

COMPANIES = {}
LAST_ALERT_KEY = "last_alert_sent_first_warning_date"


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
        "pressure_drivers": summary_target.get("pressure_drivers"),
        "minimum_required_action": summary_target.get("minimum_required_action"),
        "transaction_intelligence": summary_target.get("transaction_intelligence"),
        "scenario_ranking": ranking,
    }


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
        "pressure_drivers": summary_target.get("pressure_drivers"),
        "minimum_required_action": summary_target.get("minimum_required_action"),
        "transaction_intelligence": summary_target.get("transaction_intelligence"),
        "scenario_ranking": ranking,
    }


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


scheduler = BackgroundScheduler()


@app.on_event("startup")
def _start_scheduler():
    init_history_db()
    print(f"✅ Forecast history database ready at {DB_PATH}")
    print(f"✅ Default workspace ready: {DEFAULT_COMPANY_NAME} ({DEFAULT_COMPANY_ID})")

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
