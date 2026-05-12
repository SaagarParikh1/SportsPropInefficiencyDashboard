from __future__ import annotations

import pandas as pd

from src.config import APP_CONFIG
from src.feature_engineering import build_opponent_context_snapshot, build_player_feature_snapshot
from src.line_evaluation import evaluate_segments
from src.scoring import score_current_props
from src.utils import bucketize_line


def _rename_reference_columns(table: pd.DataFrame, prefix: str, key_columns: list[str]) -> pd.DataFrame:
    rename_map = {column: f"{prefix}_{column}" for column in table.columns if column not in key_columns}
    return table.rename(columns=rename_map)


def build_historical_reference_tables(
    historical_features: pd.DataFrame,
    min_sample: int | None = None,
) -> dict[str, pd.DataFrame]:
    threshold = min_sample if min_sample is not None else APP_CONFIG.min_context_samples

    player_reference = _rename_reference_columns(
        evaluate_segments(historical_features, ["player_name", "prop_type"], max(5, threshold)),
        "player",
        ["player_name", "prop_type"],
    )
    context_reference = _rename_reference_columns(
        evaluate_segments(
            historical_features,
            ["player_name", "prop_type", "line_range_bucket", "is_home"],
            max(4, threshold // 2),
        ),
        "context",
        ["player_name", "prop_type", "line_range_bucket", "is_home"],
    )
    line_reference = _rename_reference_columns(
        evaluate_segments(historical_features, ["prop_type", "line_range_bucket"], max(4, threshold // 2)),
        "line_bucket",
        ["prop_type", "line_range_bucket"],
    )
    opponent_reference = _rename_reference_columns(
        evaluate_segments(historical_features, ["prop_type", "opponent"], max(5, threshold // 2)),
        "opponent",
        ["prop_type", "opponent"],
    )

    return {
        "player": player_reference,
        "context": context_reference,
        "line_bucket": line_reference,
        "opponent": opponent_reference,
    }


def attach_reference_tables(
    current_props: pd.DataFrame,
    historical_features: pd.DataFrame,
) -> pd.DataFrame:
    data = current_props.copy()
    data["line_range_bucket"] = bucketize_line(data["line"]).astype("string")
    reference_tables = build_historical_reference_tables(historical_features)

    enriched = data.merge(reference_tables["player"], on=["player_name", "prop_type"], how="left")
    enriched = enriched.merge(
        reference_tables["context"],
        on=["player_name", "prop_type", "line_range_bucket", "is_home"],
        how="left",
    )
    enriched = enriched.merge(reference_tables["line_bucket"], on=["prop_type", "line_range_bucket"], how="left")
    enriched = enriched.merge(reference_tables["opponent"], on=["prop_type", "opponent"], how="left")
    return enriched


def enrich_current_props(
    current_props: pd.DataFrame,
    historical_features: pd.DataFrame,
) -> pd.DataFrame:
    data = current_props.copy()
    player_snapshot = build_player_feature_snapshot(historical_features)
    opponent_snapshot = build_opponent_context_snapshot(historical_features)

    enriched = data.merge(player_snapshot, on=["player_name", "team", "prop_type"], how="left")
    enriched = enriched.merge(opponent_snapshot, on=["prop_type", "opponent"], how="left")
    enriched = attach_reference_tables(enriched, historical_features)

    enriched["line_minus_recent_avg"] = enriched["line"] - enriched["rolling_avg_10"]
    enriched["line_minus_season_avg"] = enriched["line"] - enriched["season_avg_prior"]
    enriched["line_minus_recent_median"] = enriched["line"] - enriched["rolling_median_10"]
    benchmark_line = (
        enriched["player_avg_line"]
        if "player_avg_line" in enriched.columns
        else enriched["latest_historical_line"]
        if "latest_historical_line" in enriched.columns
        else enriched["line"]
    )
    enriched["market_difficulty_relative"] = enriched["line"] - benchmark_line
    return enriched


def analyze_current_props(current_props: pd.DataFrame, historical_features: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_current_props(current_props, historical_features)
    scored = score_current_props(enriched)
    return scored.sort_values(["overall_prop_analysis_score", "prop_type", "player_name"], ascending=[False, True, True]).reset_index(drop=True)
