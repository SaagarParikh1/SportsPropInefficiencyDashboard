from __future__ import annotations

import pandas as pd

from src.config import APP_CONFIG
from src.data_ingestion import merge_historical_props_and_results
from src.utils import (
    build_game_id,
    coerce_numeric_columns,
    parse_bool_series,
    standardize_prop_type,
    standardize_dataframe_columns,
    standardize_team_code,
    standardize_text,
)


def _standardize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    data = standardize_dataframe_columns(df)
    if "game_date" in data.columns:
        data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce").dt.normalize()

    for column in ["player_name", "sport", "market_type", "season", "bookmaker"]:
        if column in data.columns:
            data[column] = data[column].map(standardize_text)

    if "prop_type" in data.columns:
        data["prop_type"] = data["prop_type"].map(standardize_prop_type)

    for column in ["team", "opponent"]:
        if column in data.columns:
            data[column] = data[column].map(standardize_team_code)

    if "is_home" in data.columns:
        data["is_home"] = parse_bool_series(data["is_home"]).fillna(False).astype(bool)

    return data


def clean_historical_props(df: pd.DataFrame) -> pd.DataFrame:
    data = _standardize_common_fields(df)
    data = coerce_numeric_columns(data, ["opening_line", "closing_line", "line", "over_odds", "under_odds"])
    if "closing_line" not in data.columns and "line" in data.columns:
        data["closing_line"] = data["line"]
    if "opening_line" not in data.columns:
        data["opening_line"] = data["closing_line"]

    data["sport"] = data.get("sport", APP_CONFIG.default_sport).fillna(APP_CONFIG.default_sport)
    data["prop_type"] = data.get("prop_type", APP_CONFIG.default_prop_type).fillna(APP_CONFIG.default_prop_type)
    data["market_type"] = data.get("market_type", APP_CONFIG.default_market_type).fillna(APP_CONFIG.default_market_type)
    if "game_id" not in data.columns:
        data["game_id"] = data.apply(
            lambda row: build_game_id(row.get("game_date"), row.get("player_name"), row.get("team"), row.get("prop_type")),
            axis=1,
        )

    data = data.dropna(subset=["game_date", "player_name", "team", "opponent", "closing_line"])
    data = data.drop_duplicates(subset=["game_id", "player_name", "prop_type"], keep="last")
    return data.sort_values(["game_date", "player_name"]).reset_index(drop=True)


def clean_game_results(df: pd.DataFrame) -> pd.DataFrame:
    data = _standardize_common_fields(df)
    data = coerce_numeric_columns(data, ["actual_value", "minutes_played", "usage_rate", "days_rest"])
    data["sport"] = data.get("sport", APP_CONFIG.default_sport).fillna(APP_CONFIG.default_sport)
    data["prop_type"] = data.get("prop_type", APP_CONFIG.default_prop_type).fillna(APP_CONFIG.default_prop_type)
    if "game_id" not in data.columns:
        data["game_id"] = data.apply(
            lambda row: build_game_id(row.get("game_date"), row.get("player_name"), row.get("team"), row.get("prop_type")),
            axis=1,
        )

    data["days_rest"] = data.get("days_rest", 2).fillna(2)
    data = data.dropna(subset=["game_date", "player_name", "team", "opponent", "actual_value"])
    data = data.drop_duplicates(subset=["game_id", "player_name", "prop_type"], keep="last")
    return data.sort_values(["game_date", "player_name"]).reset_index(drop=True)


def clean_current_props(df: pd.DataFrame) -> pd.DataFrame:
    data = _standardize_common_fields(df)
    data = coerce_numeric_columns(data, ["line", "closing_line", "over_odds", "under_odds"])
    if "line" not in data.columns and "closing_line" in data.columns:
        data["line"] = data["closing_line"]

    data["sport"] = data.get("sport", APP_CONFIG.default_sport).fillna(APP_CONFIG.default_sport)
    data["prop_type"] = data.get("prop_type", APP_CONFIG.default_prop_type).fillna(APP_CONFIG.default_prop_type)
    data["market_type"] = data.get("market_type", APP_CONFIG.default_market_type).fillna(APP_CONFIG.default_market_type)
    if "game_id" not in data.columns:
        data["game_id"] = data.apply(
            lambda row: build_game_id(row.get("game_date"), row.get("player_name"), row.get("team"), row.get("prop_type")),
            axis=1,
        )

    data = data.dropna(subset=["game_date", "player_name", "team", "opponent", "line"])
    return data.sort_values(["game_date", "player_name"]).reset_index(drop=True)


def prepare_historical_dataset(historical_props: pd.DataFrame, game_results: pd.DataFrame) -> pd.DataFrame:
    merged = merge_historical_props_and_results(historical_props, game_results)
    for base_column in ["season", "sport", "days_rest", "minutes_played", "usage_rate"]:
        result_column = f"{base_column}_result"
        if base_column not in merged.columns and result_column in merged.columns:
            merged[base_column] = merged[result_column]
        elif base_column in merged.columns and result_column in merged.columns:
            merged[base_column] = merged[base_column].fillna(merged[result_column])

    merged["line"] = merged.get("closing_line", merged.get("line"))
    merged["days_rest"] = pd.to_numeric(merged.get("days_rest", 2), errors="coerce").fillna(2)
    merged["actual_value"] = pd.to_numeric(merged["actual_value"], errors="coerce")
    merged = merged.dropna(subset=["actual_value", "line"])
    return merged.sort_values(["game_date", "player_name"]).reset_index(drop=True)
