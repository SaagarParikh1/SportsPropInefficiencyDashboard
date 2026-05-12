from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import APP_CONFIG
from src.utils import clamp, describe_prop_type, sample_quality_label, score_to_label


def _coalesce(row: pd.Series, columns: list[str], default: float = 0.0) -> float:
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            return float(value)
    return default


def _direction_from_signal(signal: float) -> str:
    if signal >= 0.35:
        return "Over"
    if signal <= -0.35:
        return "Under"
    return "Neutral"


def _aligned_value(value: float, lean: str) -> float:
    if lean == "Over":
        return value
    if lean == "Under":
        return -value
    return 0.0


def _support_word(score: float) -> str:
    if score >= 80:
        return "strong"
    if score >= 65:
        return "solid"
    if score >= 50:
        return "mixed"
    if score >= 35:
        return "cautious"
    return "weak"


def _volatility_phrase(variance: float) -> str:
    if pd.isna(variance):
        return "unknown"
    if variance <= 20:
        return "low"
    if variance <= 45:
        return "moderate"
    return "elevated"


def _build_explanation(row: pd.Series) -> str:
    lean = row["lean"].lower()
    score = float(row["overall_prop_analysis_score"])
    support_phrase = _support_word(score)
    line_gap = abs(float(row.get("line_minus_recent_avg", 0.0)))
    stat_description = describe_prop_type(row.get("prop_type"))
    context_bias = _coalesce(
        row,
        ["context_mean_error", "player_mean_error", "line_bucket_mean_error", "opponent_mean_error"],
        0.0,
    )
    sample_size = _coalesce(
        row,
        ["context_sample_size", "player_sample_size", "line_bucket_sample_size", "opponent_sample_size"],
        0.0,
    )
    sample_phrase = sample_quality_label(sample_size)
    volatility = _volatility_phrase(row.get("recent_variance_10"))
    lean_sign = 1 if row["lean"] == "Over" else -1 if row["lean"] == "Under" else 0
    context_sign = 1 if context_bias > 0.25 else -1 if context_bias < -0.25 else 0

    if row["lean"] == "Neutral":
        return (
            f"This {stat_description} prop looks fairly neutral because the line is close to the player's recent baseline, "
            "the historical context is mixed, and the evidence stack does not point strongly in one direction."
        )

    if context_sign == 0:
        context_phrase = "historical context is mostly mixed"
    elif context_sign == lean_sign:
        context_phrase = f"similar historical spots have leaned {lean}"
    else:
        context_phrase = "historical context is mixed, so recent form is doing more of the work"

    return (
        f"This {lean} grades as a {support_phrase} signal because the line is {line_gap:.1f} "
        f"{'below' if lean == 'over' else 'above'} the player's 10-game average, {context_phrase}, "
        f"and recent {stat_description} volatility is {volatility}. "
        f"The supporting sample is {sample_phrase}, so this should be treated as a signal rather than a guarantee."
    )


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def score_current_props(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["recent_edge"] = (data["rolling_avg_10"] - data["line"]).fillna(0.0)
    data["season_edge"] = (data["season_avg_prior"] - data["line"]).fillna(0.0)
    data["matchup_edge"] = _series_or_default(data, "opponent_actual_minus_line_mean", 0.0)
    data["player_history_edge"] = _series_or_default(data, "player_mean_error", 0.0)
    data["context_edge"] = _series_or_default(data, "context_mean_error", np.nan).fillna(data["player_history_edge"])
    data["trend_edge"] = _series_or_default(data, "trend_direction", 0.0)
    data["line_bucket_edge"] = _series_or_default(data, "line_bucket_mean_error", 0.0)

    data["direction_signal"] = (
        data["recent_edge"] * 0.35
        + data["season_edge"] * 0.20
        + data["context_edge"] * 0.20
        + data["matchup_edge"] * 0.10
        + data["trend_edge"] * 0.15
    )
    data["lean"] = data["direction_signal"].apply(_direction_from_signal)

    aligned_recent = data.apply(lambda row: _aligned_value(row["recent_edge"], row["lean"]), axis=1)
    aligned_season = data.apply(lambda row: _aligned_value(row["season_edge"], row["lean"]), axis=1)
    aligned_context = data.apply(lambda row: _aligned_value(row["context_edge"], row["lean"]), axis=1)
    aligned_matchup = data.apply(lambda row: _aligned_value(row["matchup_edge"], row["lean"]), axis=1)
    aligned_player = data.apply(lambda row: _aligned_value(row["player_history_edge"], row["lean"]), axis=1)
    aligned_line_bucket = data.apply(lambda row: _aligned_value(row["line_bucket_edge"], row["lean"]), axis=1)

    reference_sample = data.apply(
        lambda row: _coalesce(
            row,
            ["context_sample_size", "player_sample_size", "line_bucket_sample_size", "opponent_sample_size"],
            0.0,
        ),
        axis=1,
    )
    sample_quality_score = clamp(25.0 + np.log1p(reference_sample) * 15.0)
    agreement_ratio = (
        (aligned_recent.gt(0).astype(int))
        + (aligned_season.gt(0).astype(int))
        + (aligned_context.gt(0).astype(int))
        + (aligned_matchup.gt(0).astype(int))
        + (aligned_player.gt(0).astype(int))
    ) / 5.0

    data["historical_context_score"] = clamp(50.0 + (aligned_context * 8.0) + (aligned_player * 4.0) + (aligned_line_bucket * 3.0))
    data["recent_form_score"] = clamp(50.0 + aligned_recent * 9.0 + _series_or_default(data, "trend_direction", 0.0) * 4.0)
    data["line_value_score"] = clamp(50.0 + (aligned_recent * 6.0) + (aligned_season * 6.0))
    data["consistency_score_component"] = clamp(0.55 * _series_or_default(data, "consistency_score", 50.0) + 22.5)
    data["matchup_score"] = clamp(
        50.0 + aligned_matchup * 10.0 + (_series_or_default(data, "matchup_difficulty_score", 50.0) - 50.0) * 0.3
    )
    data["stability_score"] = clamp(sample_quality_score * 0.55 + agreement_ratio * 45.0)

    weighted_total = sum(
        data[column] * weight for column, weight in APP_CONFIG.score_weights.items()
    )
    data["overall_prop_analysis_score"] = clamp(
        np.where(data["lean"] == "Neutral", 50.0, weighted_total)
    ).round(1)
    data["support_label"] = data["overall_prop_analysis_score"].apply(score_to_label)
    data["explanation_text"] = data.apply(_build_explanation, axis=1)

    display_columns = [
        "historical_context_score",
        "recent_form_score",
        "line_value_score",
        "consistency_score_component",
        "matchup_score",
        "stability_score",
        "overall_prop_analysis_score",
    ]
    data[display_columns] = data[display_columns].round(1)
    return data
