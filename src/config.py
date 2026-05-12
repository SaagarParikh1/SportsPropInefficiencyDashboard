from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = ROOT_DIR / "outputs"

PROP_MARKET_MAP = {
    "points": {
        "market_key": "player_points",
        "short_label": "PTS",
        "full_label": "Points",
        "stat_columns": ("PTS",),
        "kind": "single",
    },
    "rebounds": {
        "market_key": "player_rebounds",
        "short_label": "REB",
        "full_label": "Rebounds",
        "stat_columns": ("REB",),
        "kind": "single",
    },
    "assists": {
        "market_key": "player_assists",
        "short_label": "AST",
        "full_label": "Assists",
        "stat_columns": ("AST",),
        "kind": "single",
    },
    "points_rebounds": {
        "market_key": "player_points_rebounds",
        "short_label": "PTS+REB",
        "full_label": "Points + Rebounds",
        "stat_columns": ("PTS", "REB"),
        "kind": "combo",
    },
    "points_assists": {
        "market_key": "player_points_assists",
        "short_label": "PTS+AST",
        "full_label": "Points + Assists",
        "stat_columns": ("PTS", "AST"),
        "kind": "combo",
    },
    "rebounds_assists": {
        "market_key": "player_rebounds_assists",
        "short_label": "REB+AST",
        "full_label": "Rebounds + Assists",
        "stat_columns": ("REB", "AST"),
        "kind": "combo",
    },
    "points_rebounds_assists": {
        "market_key": "player_points_rebounds_assists",
        "short_label": "PRA",
        "full_label": "Points + Rebounds + Assists",
        "stat_columns": ("PTS", "REB", "AST"),
        "kind": "combo",
    },
}

MARKET_KEY_TO_PROP_TYPE = {
    value["market_key"]: prop_type for prop_type, value in PROP_MARKET_MAP.items()
}

LINE_RANGE_BINS = [-10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 100.0]
LINE_RANGE_LABELS = [
    "Under 15",
    "15-19.5",
    "20-24.5",
    "25-29.5",
    "30-34.5",
    "35-39.5",
    "40+",
]

REQUIRED_COLUMNS = {
    "historical_props": [
        "game_id",
        "game_date",
        "player_name",
        "team",
        "opponent",
        "is_home",
        "prop_type",
        "closing_line",
    ],
    "game_results": [
        "game_id",
        "game_date",
        "player_name",
        "team",
        "opponent",
        "is_home",
        "prop_type",
        "actual_value",
    ],
    "current_props": [
        "game_date",
        "player_name",
        "team",
        "opponent",
        "is_home",
        "prop_type",
        "line",
    ],
}

SCORE_BANDS = [
    {"label": "Strong historical support", "lower": 80, "upper": 100},
    {"label": "Moderate support", "lower": 65, "upper": 79},
    {"label": "Neutral", "lower": 50, "upper": 64},
    {"label": "Caution", "lower": 35, "upper": 49},
    {"label": "Historically unfavorable", "lower": 0, "upper": 34},
]


@dataclass(frozen=True)
class AppConfig:
    default_sport: str = "NBA"
    default_prop_type: str = "points"
    default_market_type: str = "over_under"
    min_segment_samples: int = 15
    min_context_samples: int = 8
    rolling_windows: tuple[int, int] = (5, 10)
    rolling_edge_window: int = 12
    bootstrap_iterations: int = 250
    stability_split_ratio: float = 0.7
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "historical_context_score": 0.24,
            "recent_form_score": 0.18,
            "line_value_score": 0.18,
            "consistency_score_component": 0.12,
            "matchup_score": 0.12,
            "stability_score": 0.16,
        }
    )


APP_CONFIG = AppConfig()
