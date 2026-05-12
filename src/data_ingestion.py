from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import APP_CONFIG, PROP_MARKET_MAP, REQUIRED_COLUMNS
from src.utils import standardize_dataframe_columns


DataSource = Any


def _read_tabular(source: DataSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if source is None:
        raise ValueError("A data source is required.")

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    filename = getattr(source, "name", "upload.csv")
    suffix = Path(filename).suffix.lower()
    if hasattr(source, "seek"):
        source.seek(0)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def validate_dataset_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    data = standardize_dataframe_columns(df)
    required = REQUIRED_COLUMNS[dataset_name]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")
    return data


def load_historical_props(source: DataSource) -> pd.DataFrame:
    return validate_dataset_columns(_read_tabular(source), "historical_props")


def load_game_results(source: DataSource) -> pd.DataFrame:
    return validate_dataset_columns(_read_tabular(source), "game_results")


def load_current_props(source: DataSource) -> pd.DataFrame:
    return validate_dataset_columns(_read_tabular(source), "current_props")


def merge_historical_props_and_results(
    historical_props: pd.DataFrame, game_results: pd.DataFrame
) -> pd.DataFrame:
    merge_candidates = [
        "game_id",
        "game_date",
        "player_name",
        "team",
        "opponent",
        "is_home",
        "prop_type",
    ]
    merge_keys = [column for column in merge_candidates if column in historical_props.columns and column in game_results.columns]
    if len(merge_keys) < 4:
        raise ValueError("Historical prop lines and game results do not share enough merge keys.")

    return historical_props.merge(
        game_results,
        on=merge_keys,
        how="left",
        suffixes=("", "_result"),
        validate="many_to_one",
    )


def _round_half(value: float) -> float:
    return float(np.round(value * 2.0) / 2.0)


def load_demo_data(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    player_profiles = [
        {"player_name": "Jalen Brunson", "team": "NYK", "base_points": 27.2, "base_rebounds": 3.8, "base_assists": 6.7, "vol_points": 5.8, "vol_rebounds": 2.1, "vol_assists": 2.4, "usage": 31.2, "market_bias": 0.3},
        {"player_name": "Jayson Tatum", "team": "BOS", "base_points": 28.4, "base_rebounds": 8.5, "base_assists": 5.9, "vol_points": 6.6, "vol_rebounds": 2.8, "vol_assists": 2.3, "usage": 30.5, "market_bias": 1.1},
        {"player_name": "Stephen Curry", "team": "GSW", "base_points": 27.6, "base_rebounds": 4.6, "base_assists": 5.8, "vol_points": 7.4, "vol_rebounds": 2.0, "vol_assists": 2.2, "usage": 29.8, "market_bias": 1.2},
        {"player_name": "Anthony Edwards", "team": "MIN", "base_points": 26.8, "base_rebounds": 5.6, "base_assists": 5.1, "vol_points": 6.9, "vol_rebounds": 2.2, "vol_assists": 2.0, "usage": 31.8, "market_bias": 0.5},
        {"player_name": "Devin Booker", "team": "PHX", "base_points": 27.1, "base_rebounds": 4.8, "base_assists": 6.4, "vol_points": 6.5, "vol_rebounds": 1.9, "vol_assists": 2.5, "usage": 30.4, "market_bias": 0.8},
        {"player_name": "Donovan Mitchell", "team": "CLE", "base_points": 26.7, "base_rebounds": 5.1, "base_assists": 5.4, "vol_points": 6.2, "vol_rebounds": 2.0, "vol_assists": 2.1, "usage": 31.1, "market_bias": 0.7},
        {"player_name": "Luka Doncic", "team": "DAL", "base_points": 31.5, "base_rebounds": 8.9, "base_assists": 9.4, "vol_points": 8.1, "vol_rebounds": 2.9, "vol_assists": 3.0, "usage": 35.8, "market_bias": 1.8},
        {"player_name": "Trae Young", "team": "ATL", "base_points": 25.9, "base_rebounds": 3.2, "base_assists": 10.2, "vol_points": 7.6, "vol_rebounds": 1.7, "vol_assists": 3.2, "usage": 33.4, "market_bias": 0.9},
    ]
    teams = [profile["team"] for profile in player_profiles]
    opponent_environment = {
        "ATL": 1.6,
        "BOS": -1.3,
        "CLE": -0.7,
        "DAL": 1.0,
        "GSW": 0.8,
        "MIN": -1.8,
        "NYK": -0.9,
        "PHX": 0.6,
    }

    historical_props_rows: list[dict[str, Any]] = []
    game_results_rows: list[dict[str, Any]] = []
    combo_bias_scale = {
        "points": 1.0,
        "rebounds": 0.45,
        "assists": 0.42,
        "points_rebounds": 0.70,
        "points_assists": 0.68,
        "rebounds_assists": 0.52,
        "points_rebounds_assists": 0.74,
    }

    for index, profile in enumerate(player_profiles):
        last_date = pd.Timestamp("2025-01-02") + pd.Timedelta(days=index)
        for game_number in range(34):
            days_rest = int(rng.choice([0, 1, 2, 3], p=[0.12, 0.42, 0.30, 0.16]))
            game_date = last_date + pd.Timedelta(days=days_rest + 1)
            last_date = game_date

            opponents = [team for team in teams if team != profile["team"]]
            opponent = str(rng.choice(opponents))
            is_home = bool(rng.integers(0, 2))
            home_effect = 1.1 if is_home else -0.6
            rest_effect = {0: -1.9, 1: -0.5, 2: 0.4, 3: 0.9}[days_rest]
            form_wave = 2.3 * np.sin((game_number + index) / 4.0)
            points_noise = rng.normal(0, profile["vol_points"])
            rebounds_noise = rng.normal(0, profile["vol_rebounds"])
            assists_noise = rng.normal(0, profile["vol_assists"])
            actual_points = np.clip(
                profile["base_points"] + form_wave + opponent_environment[opponent] + home_effect + rest_effect + points_noise,
                8,
                55,
            )
            actual_rebounds = np.clip(
                profile["base_rebounds"] + 0.18 * form_wave + 0.35 * opponent_environment[opponent] + 0.22 * home_effect + 0.25 * rest_effect + rebounds_noise,
                1,
                20,
            )
            actual_assists = np.clip(
                profile["base_assists"] + 0.22 * form_wave + 0.3 * opponent_environment[opponent] + 0.18 * home_effect + 0.28 * rest_effect + assists_noise,
                1,
                18,
            )
            minutes_played = np.clip(
                33.0 + profile["usage"] / 10.0 + rng.normal(0, 2.4) - (0.8 if days_rest == 0 else 0.0),
                24,
                41,
            )
            usage_rate = np.clip(profile["usage"] + rng.normal(0, 1.3), 24, 39)

            stat_values = {
                "points": actual_points,
                "rebounds": actual_rebounds,
                "assists": actual_assists,
                "points_rebounds": actual_points + actual_rebounds,
                "points_assists": actual_points + actual_assists,
                "rebounds_assists": actual_rebounds + actual_assists,
                "points_rebounds_assists": actual_points + actual_rebounds + actual_assists,
            }
            base_line_levels = {
                "points": profile["base_points"],
                "rebounds": profile["base_rebounds"],
                "assists": profile["base_assists"],
                "points_rebounds": profile["base_points"] + profile["base_rebounds"],
                "points_assists": profile["base_points"] + profile["base_assists"],
                "rebounds_assists": profile["base_rebounds"] + profile["base_assists"],
                "points_rebounds_assists": profile["base_points"] + profile["base_rebounds"] + profile["base_assists"],
            }

            for prop_type in PROP_MARKET_MAP:
                opening_line = _round_half(
                    base_line_levels[prop_type]
                    + 0.45 * form_wave
                    + 0.35 * opponent_environment[opponent]
                    + 0.25 * home_effect
                    + combo_bias_scale[prop_type] * profile["market_bias"]
                    + rng.normal(0, 1.1 if PROP_MARKET_MAP[prop_type]["kind"] == "single" else 1.5)
                )
                closing_line = _round_half(opening_line + rng.normal(0, 0.7 if PROP_MARKET_MAP[prop_type]["kind"] == "single" else 1.0))
                game_id = f"{game_date:%Y%m%d}_{profile['team']}_{profile['player_name'].lower().replace(' ', '')}_{prop_type}"
                historical_props_rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date,
                        "season": "2025-2026",
                        "sport": APP_CONFIG.default_sport,
                        "player_name": profile["player_name"],
                        "team": profile["team"],
                        "opponent": opponent,
                        "is_home": is_home,
                        "prop_type": prop_type,
                        "market_type": APP_CONFIG.default_market_type,
                        "opening_line": opening_line,
                        "closing_line": closing_line,
                        "over_odds": int(-110 + rng.integers(-12, 10)),
                        "under_odds": int(-110 + rng.integers(-10, 12)),
                        "bookmaker": "consensus_demo",
                    }
                )
                game_results_rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date,
                        "season": "2025-2026",
                        "sport": APP_CONFIG.default_sport,
                        "player_name": profile["player_name"],
                        "team": profile["team"],
                        "opponent": opponent,
                        "is_home": is_home,
                        "prop_type": prop_type,
                        "actual_value": round(float(stat_values[prop_type]), 1),
                        "minutes_played": round(float(minutes_played), 1),
                        "usage_rate": round(float(usage_rate), 1),
                        "days_rest": days_rest,
                    }
                )

    historical_props = pd.DataFrame(historical_props_rows).sort_values(["game_date", "player_name"]).reset_index(drop=True)
    game_results = pd.DataFrame(game_results_rows).sort_values(["game_date", "player_name"]).reset_index(drop=True)

    current_rows: list[dict[str, Any]] = []
    for profile in player_profiles:
        player_games = game_results.loc[game_results["player_name"] == profile["player_name"]].sort_values(["prop_type", "game_date"])
        last_game = player_games.iloc[-1]
        upcoming_date = pd.to_datetime(last_game["game_date"]) + pd.Timedelta(days=2)
        opponents = [team for team in teams if team != profile["team"]]
        opponent = str(rng.choice(opponents))
        is_home = bool(rng.integers(0, 2))
        for prop_type in PROP_MARKET_MAP:
            latest = player_games.loc[player_games["prop_type"] == prop_type].tail(10)
            line = _round_half(
                latest["actual_value"].mean()
                + combo_bias_scale[prop_type] * profile["market_bias"]
                + 0.35 * opponent_environment[opponent]
                + (0.4 if is_home else -0.2)
                + rng.normal(0, 0.9 if PROP_MARKET_MAP[prop_type]["kind"] == "single" else 1.2)
            )
            current_rows.append(
                {
                    "game_date": upcoming_date,
                    "sport": APP_CONFIG.default_sport,
                    "player_name": profile["player_name"],
                    "team": profile["team"],
                    "opponent": opponent,
                    "is_home": is_home,
                    "prop_type": prop_type,
                    "market_type": APP_CONFIG.default_market_type,
                    "line": line,
                    "over_odds": int(-112 + rng.integers(-10, 9)),
                    "under_odds": int(-108 + rng.integers(-8, 11)),
                    "bookmaker": "consensus_demo",
                }
            )

    return {
        "historical_props": historical_props,
        "game_results": game_results,
        "current_props": pd.DataFrame(current_rows).sort_values(["game_date", "player_name"]).reset_index(drop=True),
    }
