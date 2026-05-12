from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import APP_CONFIG
from src.utils import wilson_interval


def ensure_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "actual_minus_line" not in data.columns:
        data["actual_minus_line"] = data["actual_value"] - data["line"]
    if "absolute_line_error" not in data.columns:
        data["absolute_line_error"] = data["actual_minus_line"].abs()
    if "over_hit" not in data.columns:
        data["over_hit"] = (data["actual_value"] > data["line"]).astype(int)
    if "under_hit" not in data.columns:
        data["under_hit"] = (data["actual_value"] < data["line"]).astype(int)
    if "push" not in data.columns:
        data["push"] = (data["actual_value"] == data["line"]).astype(int)
    return data


def overall_market_metrics(df: pd.DataFrame) -> dict[str, float]:
    data = ensure_outcome_columns(df)
    total_props = float(len(data))
    decision_sample = float((1 - data["push"]).sum())
    over_hits = float(data["over_hit"].sum())
    under_hits = float(data["under_hit"].sum())
    over_ci_low, over_ci_high = wilson_interval(over_hits, decision_sample)
    under_ci_low, under_ci_high = wilson_interval(under_hits, decision_sample)

    return {
        "total_props": total_props,
        "decision_sample": decision_sample,
        "over_hit_rate": over_hits / decision_sample if decision_sample else 0.0,
        "under_hit_rate": under_hits / decision_sample if decision_sample else 0.0,
        "push_rate": float(data["push"].mean()) if total_props else 0.0,
        "mean_error": float(data["actual_minus_line"].mean()) if total_props else 0.0,
        "median_error": float(data["actual_minus_line"].median()) if total_props else 0.0,
        "mae": float(data["absolute_line_error"].mean()) if total_props else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(data["actual_minus_line"])))) if total_props else 0.0,
        "over_ci_low": over_ci_low,
        "over_ci_high": over_ci_high,
        "under_ci_low": under_ci_low,
        "under_ci_high": under_ci_high,
        "unique_players": float(data["player_name"].nunique()) if "player_name" in data.columns else 0.0,
    }


def evaluate_segments(
    df: pd.DataFrame,
    group_by: str | list[str],
    min_sample: int | None = None,
) -> pd.DataFrame:
    data = ensure_outcome_columns(df)
    group_columns = [group_by] if isinstance(group_by, str) else group_by
    threshold = min_sample if min_sample is not None else APP_CONFIG.min_segment_samples

    aggregated = (
        data.groupby(group_columns, dropna=False)
        .agg(
            sample_size=("actual_value", "size"),
            pushes=("push", "sum"),
            over_hits=("over_hit", "sum"),
            under_hits=("under_hit", "sum"),
            mean_error=("actual_minus_line", "mean"),
            median_error=("actual_minus_line", "median"),
            mae=("absolute_line_error", "mean"),
            avg_line=("line", "mean"),
            avg_actual=("actual_value", "mean"),
            unique_players=("player_name", "nunique"),
            first_game=("game_date", "min"),
            last_game=("game_date", "max"),
        )
        .reset_index()
    )

    aggregated["decision_sample"] = aggregated["sample_size"] - aggregated["pushes"]
    aggregated["over_hit_rate"] = (
        aggregated["over_hits"] / aggregated["decision_sample"].replace(0, np.nan)
    ).fillna(0.0)
    aggregated["under_hit_rate"] = (
        aggregated["under_hits"] / aggregated["decision_sample"].replace(0, np.nan)
    ).fillna(0.0)
    aggregated["push_rate"] = (aggregated["pushes"] / aggregated["sample_size"].replace(0, np.nan)).fillna(0.0)
    intervals = aggregated.apply(
        lambda row: wilson_interval(row["over_hits"], row["decision_sample"]),
        axis=1,
        result_type="expand",
    )
    aggregated["over_ci_low"] = intervals[0]
    aggregated["over_ci_high"] = intervals[1]
    aggregated["bias_direction"] = np.select(
        [aggregated["mean_error"] > 0.75, aggregated["mean_error"] < -0.75],
        ["Over lean", "Under lean"],
        default="Balanced / unclear",
    )
    aggregated["potential_inefficiency_flag"] = np.select(
        [
            (aggregated["sample_size"] >= threshold) & (aggregated["mean_error"] > 1.0),
            (aggregated["sample_size"] >= threshold) & (aggregated["mean_error"] < -1.0),
        ],
        ["Historically favorable to overs", "Historically favorable to unders"],
        default="Requires caution",
    )
    aggregated = aggregated.loc[aggregated["sample_size"] >= threshold].sort_values(
        ["sample_size", "mean_error"], ascending=[False, False]
    )
    return aggregated.reset_index(drop=True)


def collect_segment_tables(df: pd.DataFrame, min_sample: int | None = None) -> dict[str, pd.DataFrame]:
    return {
        "Prop Type": evaluate_segments(df, "prop_type", min_sample),
        "Player": evaluate_segments(df, "player_name", min_sample),
        "Team": evaluate_segments(df, "team", min_sample),
        "Opponent": evaluate_segments(df, "opponent", min_sample),
        "Home/Away": evaluate_segments(df, "is_home", max(5, (min_sample or APP_CONFIG.min_segment_samples) // 2)),
        "Rest Bucket": evaluate_segments(df, "days_rest_bucket", max(5, (min_sample or APP_CONFIG.min_segment_samples) // 2)),
        "Line Range": evaluate_segments(df, "line_range_bucket", max(5, (min_sample or APP_CONFIG.min_segment_samples) // 2)),
    }


def rolling_edge_summary(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    data = ensure_outcome_columns(df)
    rolling_window = window or APP_CONFIG.rolling_edge_window
    daily = (
        data.groupby("game_date")
        .agg(
            total_props=("actual_value", "size"),
            over_hit_rate=("over_hit", "mean"),
            under_hit_rate=("under_hit", "mean"),
            mean_error=("actual_minus_line", "mean"),
            mae=("absolute_line_error", "mean"),
        )
        .reset_index()
        .sort_values("game_date")
    )
    min_periods = max(3, rolling_window // 3)
    daily["rolling_over_hit_rate"] = daily["over_hit_rate"].rolling(rolling_window, min_periods=min_periods).mean()
    daily["rolling_under_hit_rate"] = daily["under_hit_rate"].rolling(rolling_window, min_periods=min_periods).mean()
    daily["rolling_mean_error"] = daily["mean_error"].rolling(rolling_window, min_periods=min_periods).mean()
    daily["rolling_mae"] = daily["mae"].rolling(rolling_window, min_periods=min_periods).mean()
    return daily


def temporal_stability_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    min_sample: int | None = None,
) -> pd.DataFrame:
    data = ensure_outcome_columns(df).sort_values("game_date")
    threshold = min_sample if min_sample is not None else APP_CONFIG.min_context_samples
    unique_dates = sorted(data["game_date"].dropna().unique())
    if len(unique_dates) < 8:
        return pd.DataFrame()

    cutoff_index = max(1, int(len(unique_dates) * APP_CONFIG.stability_split_ratio) - 1)
    cutoff_date = unique_dates[cutoff_index]
    train = data.loc[data["game_date"] <= cutoff_date]
    test = data.loc[data["game_date"] > cutoff_date]

    train_segments = evaluate_segments(train, segment_col, threshold)
    test_segments = evaluate_segments(test, segment_col, max(5, threshold // 2))
    if train_segments.empty or test_segments.empty:
        return pd.DataFrame()

    merged = train_segments.merge(test_segments, on=segment_col, suffixes=("_train", "_test"))
    merged["over_rate_delta"] = merged["over_hit_rate_test"] - merged["over_hit_rate_train"]
    merged["mean_error_delta"] = merged["mean_error_test"] - merged["mean_error_train"]
    merged["direction_persisted"] = np.where(
        np.sign(merged["mean_error_train"]) == np.sign(merged["mean_error_test"]),
        "Yes",
        "No",
    )
    merged["stability_assessment"] = np.select(
        [
            (merged["direction_persisted"] == "Yes") & (merged["mean_error_delta"].abs() <= 0.75),
            (merged["direction_persisted"] == "Yes") & (merged["mean_error_delta"].abs() > 0.75),
        ],
        ["Relatively stable", "Direction held, strength shifted"],
        default="Did not persist cleanly",
    )
    return merged.sort_values("sample_size_train", ascending=False).reset_index(drop=True)


def bootstrap_hit_rate_bounds(
    outcomes: pd.Series,
    iterations: int | None = None,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clean = outcomes.dropna().astype(float)
    if clean.empty:
        return 0.0, 0.0
    n_iterations = iterations or APP_CONFIG.bootstrap_iterations
    rng = np.random.default_rng(42)
    samples = [rng.choice(clean.to_numpy(), size=len(clean), replace=True).mean() for _ in range(n_iterations)]
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    return float(np.quantile(samples, lower_q)), float(np.quantile(samples, upper_q))
