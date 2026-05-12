from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src.config import LINE_RANGE_BINS, LINE_RANGE_LABELS, PROP_MARKET_MAP, SCORE_BANDS


def to_snake_case(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(value))
    return re.sub(r"_+", "_", cleaned).strip("_").lower()


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [to_snake_case(column) for column in data.columns]
    return data


def standardize_text(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return re.sub(r"\s+", " ", str(value)).strip().title()


def standardize_prop_type(value: object) -> object:
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "pts": "points",
        "point": "points",
        "points": "points",
        "reb": "rebounds",
        "rebound": "rebounds",
        "rebounds": "rebounds",
        "ast": "assists",
        "assist": "assists",
        "assists": "assists",
        "pr": "points_rebounds",
        "pts_reb": "points_rebounds",
        "points_rebounds": "points_rebounds",
        "points_rebound": "points_rebounds",
        "pts_ast": "points_assists",
        "points_assists": "points_assists",
        "points_assist": "points_assists",
        "ra": "rebounds_assists",
        "reb_ast": "rebounds_assists",
        "rebounds_assists": "rebounds_assists",
        "rebounds_assist": "rebounds_assists",
        "pra": "points_rebounds_assists",
        "par": "points_rebounds_assists",
        "points_rebounds_assists": "points_rebounds_assists",
        "points_rebounds_assist": "points_rebounds_assists",
    }
    return aliases.get(cleaned, cleaned)


def format_prop_type(value: object) -> str:
    prop_type = standardize_prop_type(value)
    if pd.isna(prop_type):
        return "Unknown"
    prop_meta = PROP_MARKET_MAP.get(str(prop_type))
    if prop_meta:
        return str(prop_meta["short_label"])
    return str(prop_type).replace("_", " ").title()


def describe_prop_type(value: object) -> str:
    prop_type = standardize_prop_type(value)
    if pd.isna(prop_type):
        return "stat line"
    prop_meta = PROP_MARKET_MAP.get(str(prop_type))
    if prop_meta:
        return str(prop_meta["full_label"]).lower()
    return str(prop_type).replace("_", " ").lower()


def standardize_team_code(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper()


def parse_bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
            "home": True,
            "away": False,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.where(mapped.notna(), numeric.map(lambda x: bool(int(x)) if pd.notna(x) else np.nan))
    return mapped.astype("boolean")


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    data = df.copy()
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def clamp(value: object, lower: float = 0.0, upper: float = 100.0) -> object:
    if isinstance(value, pd.Series):
        return value.clip(lower=lower, upper=upper)
    if isinstance(value, (np.ndarray, list, tuple)):
        return np.clip(value, lower, upper)
    return float(np.clip(value, lower, upper))


def bucketize_line(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=LINE_RANGE_BINS,
        labels=LINE_RANGE_LABELS,
        include_lowest=True,
        right=False,
    )


def bucketize_rest(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(2)
    conditions = [
        numeric <= 0,
        numeric == 1,
        numeric == 2,
        numeric >= 3,
    ]
    labels = ["Back-to-back", "1 day", "2 days", "3+ days"]
    return pd.Series(np.select(conditions, labels, default="Unknown"), index=series.index)


def wilson_interval(successes: float, total: float, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    lower, upper = proportion_confint(count=successes, nobs=total, alpha=alpha, method="wilson")
    return float(lower), float(upper)


def build_game_id(date_value: object, player_name: object, team: object, prop_type: object) -> str:
    if pd.isna(date_value):
        date_part = "unknown_date"
    else:
        date_part = pd.to_datetime(date_value).strftime("%Y%m%d")
    player_part = re.sub(r"[^a-z0-9]+", "", str(player_name).lower())
    team_part = re.sub(r"[^a-z0-9]+", "", str(team).lower())
    prop_part = re.sub(r"[^a-z0-9]+", "", str(prop_type).lower())
    return f"{date_part}_{team_part}_{player_part}_{prop_part}"


def score_to_label(score: float) -> str:
    for band in SCORE_BANDS:
        if band["lower"] <= score <= band["upper"]:
            return band["label"]
    return SCORE_BANDS[-1]["label"]


def sample_quality_label(sample_size: float) -> str:
    if sample_size >= 75:
        return "strong"
    if sample_size >= 35:
        return "decent"
    if sample_size >= 15:
        return "limited"
    return "very limited"


def format_pct(value: float) -> str:
    return f"{value:.1%}"
