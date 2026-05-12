from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import APP_CONFIG
from src.utils import bucketize_line, bucketize_rest, clamp


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=max(2, window // 2)).mean()


def _shifted_rolling_median(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=max(2, window // 2)).median()


def _shifted_rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=max(2, window // 2)).std()


def _expanding_prior_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=2).mean()


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    short_window, long_window = APP_CONFIG.rolling_windows
    data = df.sort_values(["player_name", "prop_type", "game_date"]).copy()
    data["line"] = pd.to_numeric(data["line"], errors="coerce")
    data["actual_value"] = pd.to_numeric(data["actual_value"], errors="coerce")
    data["days_rest"] = pd.to_numeric(data.get("days_rest", 2), errors="coerce").fillna(2)
    data["back_to_back"] = (data["days_rest"] <= 0).astype(int)
    data["days_rest_bucket"] = bucketize_rest(data["days_rest"])
    data["line_range_bucket"] = bucketize_line(data["line"]).astype("string")
    data["actual_minus_line"] = data["actual_value"] - data["line"]
    data["absolute_line_error"] = data["actual_minus_line"].abs()
    data["over_hit"] = (data["actual_value"] > data["line"]).astype(int)
    data["under_hit"] = (data["actual_value"] < data["line"]).astype(int)
    data["push"] = (data["actual_value"] == data["line"]).astype(int)
    data["game_sequence"] = data.groupby(["player_name", "prop_type"]).cumcount() + 1

    player_groups = data.groupby(["player_name", "prop_type"], group_keys=False)
    data["rolling_avg_5"] = player_groups["actual_value"].transform(lambda series: _shifted_rolling_mean(series, short_window))
    data["rolling_avg_10"] = player_groups["actual_value"].transform(lambda series: _shifted_rolling_mean(series, long_window))
    data["rolling_median_10"] = player_groups["actual_value"].transform(lambda series: _shifted_rolling_median(series, long_window))
    data["recent_variance_10"] = player_groups["actual_value"].transform(lambda series: _shifted_rolling_std(series, long_window)).pow(2)
    data["recent_std_10"] = player_groups["actual_value"].transform(lambda series: _shifted_rolling_std(series, long_window))
    data["season_avg_prior"] = player_groups["actual_value"].transform(_expanding_prior_mean)
    data["recent_hit_rate_over_5"] = player_groups["over_hit"].transform(lambda series: _shifted_rolling_mean(series, short_window))
    data["recent_hit_rate_under_5"] = player_groups["under_hit"].transform(lambda series: _shifted_rolling_mean(series, short_window))
    data["rolling_minutes_5"] = player_groups["minutes_played"].transform(lambda series: _shifted_rolling_mean(series, short_window))
    data["rolling_usage_5"] = player_groups["usage_rate"].transform(lambda series: _shifted_rolling_mean(series, short_window))

    coefficient_of_variation = data["recent_std_10"] / data["rolling_avg_10"].abs().clip(lower=1.0)
    data["consistency_score"] = clamp((1.0 - coefficient_of_variation.clip(lower=0, upper=1.5) / 1.5) * 100.0)
    data["line_minus_recent_avg"] = data["line"] - data["rolling_avg_10"]
    data["line_minus_recent_median"] = data["line"] - data["rolling_median_10"]
    data["line_minus_season_avg"] = data["line"] - data["season_avg_prior"]
    data["trend_direction"] = data["rolling_avg_5"] - data["rolling_avg_10"]

    opponent_frame = data.sort_values(["prop_type", "game_date", "opponent", "player_name"]).copy()
    opponent_frame["opponent_avg_allowed_prior"] = opponent_frame.groupby(["prop_type", "opponent"])["actual_value"].transform(_expanding_prior_mean)
    opponent_frame["league_avg_allowed_prior"] = opponent_frame.groupby("prop_type")["actual_value"].transform(
        lambda series: series.shift(1).expanding(min_periods=5).mean()
    )
    opponent_frame["matchup_allowance_delta"] = (
        opponent_frame["opponent_avg_allowed_prior"] - opponent_frame["league_avg_allowed_prior"]
    ).fillna(0.0)
    opponent_frame["matchup_difficulty_score"] = clamp(50.0 + opponent_frame["matchup_allowance_delta"] * 8.0)

    matchup_features = opponent_frame[
        ["game_id", "prop_type", "matchup_allowance_delta", "matchup_difficulty_score", "opponent_avg_allowed_prior"]
    ]
    data = data.merge(matchup_features, on=["game_id", "prop_type"], how="left")
    data["matchup_difficulty_score"] = data["matchup_difficulty_score"].fillna(50.0)
    data["matchup_allowance_delta"] = data["matchup_allowance_delta"].fillna(0.0)
    data["opponent_avg_allowed_prior"] = data["opponent_avg_allowed_prior"].fillna(data["season_avg_prior"])
    return data.sort_values(["game_date", "player_name", "prop_type"]).reset_index(drop=True)


def build_player_feature_snapshot(featured_history: pd.DataFrame) -> pd.DataFrame:
    if featured_history.empty:
        return featured_history.copy()

    snapshot_columns = [
        "player_name",
        "team",
        "prop_type",
        "game_date",
        "line",
        "rolling_avg_5",
        "rolling_avg_10",
        "rolling_median_10",
        "season_avg_prior",
        "recent_variance_10",
        "consistency_score",
        "trend_direction",
        "recent_hit_rate_over_5",
        "recent_hit_rate_under_5",
        "rolling_minutes_5",
        "rolling_usage_5",
    ]
    snapshot = (
        featured_history.sort_values(["player_name", "prop_type", "game_date"])
        .groupby(["player_name", "prop_type"], as_index=False)
        .tail(1)[snapshot_columns]
        .rename(columns={"game_date": "last_game_date", "line": "latest_historical_line"})
        .reset_index(drop=True)
    )
    return snapshot


def build_opponent_context_snapshot(featured_history: pd.DataFrame) -> pd.DataFrame:
    if featured_history.empty:
        return featured_history.copy()

    return (
        featured_history.groupby(["prop_type", "opponent"], dropna=False)
        .agg(
            opponent_sample=("actual_value", "size"),
            opponent_actual_minus_line_mean=("actual_minus_line", "mean"),
            opponent_over_hit_rate=("over_hit", "mean"),
            opponent_under_hit_rate=("under_hit", "mean"),
            opponent_matchup_difficulty=("matchup_difficulty_score", "mean"),
        )
        .reset_index()
    )
