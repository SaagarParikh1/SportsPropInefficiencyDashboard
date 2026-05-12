from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelogs, scoreboardv3
from nba_api.stats.static import teams

from src.config import APP_CONFIG, CACHE_DIR, MARKET_KEY_TO_PROP_TYPE, PROP_MARKET_MAP
from src.prop_analysis import attach_reference_tables
from src.scoring import score_current_props
from src.utils import build_game_id, clamp


THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CACHED_HISTORY_PROPS = CACHE_DIR / "historical_props_latest.parquet"
CACHED_HISTORY_RESULTS = CACHE_DIR / "game_results_latest.parquet"
PENDING_PROP_ARCHIVE = CACHE_DIR / "pending_live_props.parquet"


def current_nba_season(reference_date: datetime | None = None) -> str:
    now = reference_date or datetime.now(timezone.utc)
    if now.month >= 10:
        start_year = now.year
    else:
        start_year = now.year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"


def _normalize_calendar_date(values: Any) -> Any:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_localize(None).dt.normalize()
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed.tz_localize(None).normalize()
    if pd.isna(parsed):
        return pd.NaT
    return parsed.tz_localize(None).normalize()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def fetch_daily_nba_games(game_date: str | datetime | pd.Timestamp | None = None, timezone_name: str = "America/Chicago") -> pd.DataFrame:
    reference_date = (
        pd.Timestamp.now(tz=timezone_name).date()
        if game_date is None
        else pd.to_datetime(game_date, errors="coerce").date()
    )
    empty = pd.DataFrame(
        columns=[
            "game_id",
            "game_time",
            "status",
            "away_team",
            "home_team",
            "away_team_id",
            "home_team_id",
            "away_team_name",
            "home_team_name",
            "away_logo_url",
            "home_logo_url",
            "away_score",
            "home_score",
            "away_record",
            "home_record",
            "series_text",
            "game_label",
        ]
    )
    if pd.isna(reference_date):
        return empty

    board = scoreboardv3.ScoreboardV3(game_date=reference_date.isoformat(), timeout=20)
    frames = board.get_data_frames()
    if len(frames) < 3:
        return empty

    games = frames[1].copy()
    team_lines = frames[2].copy()
    if games.empty or team_lines.empty:
        return empty

    team_lookup: dict[tuple[str, str], pd.Series] = {
        (str(row["gameId"]), str(row["teamTricode"])): row
        for _, row in team_lines.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for _, game in games.iterrows():
        game_id = str(game.get("gameId", ""))
        code = str(game.get("gameCode", "")).split("/")[-1]
        away_code = code[:3] if len(code) >= 6 else ""
        home_code = code[-3:] if len(code) >= 6 else ""
        away = team_lookup.get((game_id, away_code))
        home = team_lookup.get((game_id, home_code))
        if away is None or home is None:
            grouped = team_lines.loc[team_lines["gameId"].astype(str) == game_id]
            if len(grouped) >= 2:
                away = grouped.iloc[1]
                home = grouped.iloc[0]
            else:
                continue

        game_time = pd.to_datetime(game.get("gameTimeUTC"), utc=True, errors="coerce")
        local_time = ""
        if pd.notna(game_time):
            local_time = game_time.tz_convert(timezone_name).strftime("%I:%M %p").lstrip("0")

        rows.append(
            {
                "game_id": game_id,
                "game_time": local_time or str(game.get("gameStatusText", "")),
                "status": str(game.get("gameStatusText", "")),
                "away_team": str(away.get("teamTricode", "")),
                "home_team": str(home.get("teamTricode", "")),
                "away_team_id": str(away.get("teamId", "")),
                "home_team_id": str(home.get("teamId", "")),
                "away_team_name": f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip(),
                "home_team_name": f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip(),
                "away_logo_url": f"https://cdn.nba.com/logos/nba/{away.get('teamId', '')}/primary/L/logo.svg",
                "home_logo_url": f"https://cdn.nba.com/logos/nba/{home.get('teamId', '')}/primary/L/logo.svg",
                "away_score": _safe_int(away.get("score", 0)),
                "home_score": _safe_int(home.get("score", 0)),
                "away_record": f"{_safe_int(away.get('wins', 0))}-{_safe_int(away.get('losses', 0))}",
                "home_record": f"{_safe_int(home.get('wins', 0))}-{_safe_int(home.get('losses', 0))}",
                "series_text": str(game.get("seriesText", "") or ""),
                "game_label": str(game.get("gameLabel", "") or game.get("gameSubLabel", "") or ""),
            }
        )

    return pd.DataFrame(rows)


def _team_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for team in teams.get_teams():
        full_name = team["full_name"]
        abbreviation = team["abbreviation"]
        mapping[full_name.lower()] = abbreviation
        mapping[team["nickname"].lower()] = abbreviation
        mapping[f"{team['city'].lower()} {team['nickname'].lower()}"] = abbreviation
    mapping["la clippers"] = "LAC"
    mapping["los angeles clippers"] = "LAC"
    mapping["la lakers"] = "LAL"
    mapping["los angeles lakers"] = "LAL"
    return mapping


TEAM_NAME_TO_ABBR = _team_name_map()


def supported_prop_types() -> list[str]:
    return list(PROP_MARKET_MAP.keys())


def supported_market_keys(prop_types: list[str] | None = None) -> list[str]:
    selected_prop_types = prop_types or supported_prop_types()
    return [PROP_MARKET_MAP[prop_type]["market_key"] for prop_type in selected_prop_types if prop_type in PROP_MARKET_MAP]


def _requests_get(url: str, params: dict[str, Any], max_attempts: int = 3) -> Any:
    response: requests.Response | None = None
    for attempt in range(1, max_attempts + 1):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429 and attempt < max_attempts:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else 1.5 * attempt
            except (TypeError, ValueError):
                wait_seconds = 1.5 * attempt
            time.sleep(min(max(wait_seconds, 1.0), 8.0))
            continue
        break

    assert response is not None
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("error") or "").strip()
        except ValueError:
            detail = response.text.strip()[:240]
        message = f"The Odds API request failed with status {response.status_code}."
        if detail:
            message = f"{message} {detail}"
        if response.status_code == 429:
            message = (
                f"{message} Wait a moment and refresh again. "
                "This scanner now uses cached/manual refresh behavior, but the API can still throttle burst requests."
            )
        raise RuntimeError(message) from exc
    return response.json()


def fetch_upcoming_nba_events(
    api_key: str,
    days_ahead: int = 2,
) -> list[dict[str, Any]]:
    url = f"{THE_ODDS_API_BASE}/sports/basketball_nba/events"
    events = _requests_get(url, {"apiKey": api_key, "dateFormat": "iso"})
    cutoff = datetime.now(timezone.utc) + timedelta(days=max(1, days_ahead))
    filtered_events: list[dict[str, Any]] = []
    for event in events:
        commence_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(commence_time):
            continue
        if commence_time.to_pydatetime() <= cutoff:
            filtered_events.append(event)
    return filtered_events


def fetch_event_player_props(
    api_key: str,
    event_id: str,
    markets: list[str],
    bookmakers: list[str] | None = None,
    regions: str = "us,us2",
) -> dict[str, Any]:
    url = f"{THE_ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    return _requests_get(url, params)


def _flatten_outcomes_to_rows(event_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_id = event_payload.get("id")
    commence_time = event_payload.get("commence_time")
    home_team_name = event_payload.get("home_team")
    away_team_name = event_payload.get("away_team")

    for bookmaker in event_payload.get("bookmakers", []):
        bookmaker_key = bookmaker.get("key")
        bookmaker_title = bookmaker.get("title")
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            prop_type = MARKET_KEY_TO_PROP_TYPE.get(str(market_key))
            if prop_type is None:
                continue
            grouped: dict[tuple[str, float], dict[str, Any]] = {}
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description")
                line = outcome.get("point")
                side = outcome.get("name")
                if player_name is None or line is None or side not in {"Over", "Under"}:
                    continue
                key = (str(player_name), float(line), prop_type)
                grouped.setdefault(
                    key,
                    {
                        "event_id": event_id,
                        "game_date": commence_time,
                        "player_name": str(player_name),
                        "prop_type": prop_type,
                        "market_key": market_key,
                        "line": float(line),
                        "home_team_name": home_team_name,
                        "away_team_name": away_team_name,
                        "bookmaker": bookmaker_title or bookmaker_key,
                        "bookmaker_key": bookmaker_key,
                        "market_last_update": market.get("last_update") or bookmaker.get("last_update"),
                    },
                )
                grouped[key][f"{side.lower()}_odds"] = outcome.get("price")
            rows.extend(grouped.values())
    return rows


def fetch_live_nba_player_points_props(
    api_key: str,
    bookmakers: list[str] | None = None,
    days_ahead: int = 2,
    regions: str = "us,us2",
    prop_types: list[str] | None = None,
) -> pd.DataFrame:
    events = fetch_upcoming_nba_events(api_key=api_key, days_ahead=days_ahead)
    markets = supported_market_keys(prop_types)
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if index > 0:
            time.sleep(0.35)
        payload = fetch_event_player_props(
            api_key=api_key,
            event_id=event["id"],
            markets=markets,
            bookmakers=bookmakers,
            regions=regions,
        )
        rows.extend(_flatten_outcomes_to_rows(payload))

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "game_date",
                "player_name",
                "prop_type",
                "line",
                "home_team_name",
                "away_team_name",
                "bookmaker",
                "bookmaker_key",
                "over_odds",
                "under_odds",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["event_id", "player_name", "prop_type", "line", "bookmaker_key"], keep="last"
    ).reset_index(drop=True)


@lru_cache(maxsize=4)
def _fetch_season_player_logs_cached(season_value: str) -> pd.DataFrame:
    logs = playergamelogs.PlayerGameLogs(season_nullable=season_value).get_data_frames()[0].copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
    for stat_column in ["PTS", "REB", "AST", "MIN"]:
        if stat_column in logs.columns:
            logs[stat_column] = pd.to_numeric(logs[stat_column], errors="coerce")
    logs = logs.dropna(subset=["GAME_DATE", "PTS", "REB", "AST", "PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP"]).copy()
    logs["OPPONENT"] = logs["MATCHUP"].astype(str).str.extract(r"(?:vs\\.|@)\\s+([A-Z]{2,3})")[0]
    logs["IS_HOME"] = logs["MATCHUP"].astype(str).str.contains("vs\\.", regex=True)
    return logs.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)


def fetch_season_player_logs(season: str | None = None) -> pd.DataFrame:
    season_value = season or current_nba_season()
    return _fetch_season_player_logs_cached(season_value).copy()


def _expand_logs_by_prop_type(player_logs: pd.DataFrame) -> pd.DataFrame:
    prop_frames: list[pd.DataFrame] = []
    base_columns = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "OPPONENT", "IS_HOME", "MIN"]
    for prop_type, meta in PROP_MARKET_MAP.items():
        stat_columns = list(meta["stat_columns"])
        if any(column not in player_logs.columns for column in stat_columns):
            continue
        frame = player_logs[base_columns + stat_columns].copy()
        frame["prop_type"] = prop_type
        frame["actual_value"] = frame[stat_columns].sum(axis=1)
        prop_frames.append(frame)

    if not prop_frames:
        return pd.DataFrame()
    return pd.concat(prop_frames, ignore_index=True).sort_values(["PLAYER_NAME", "prop_type", "GAME_DATE"]).reset_index(drop=True)


def build_auto_game_results_from_logs(player_logs: pd.DataFrame) -> pd.DataFrame:
    if player_logs.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "sport",
                "player_name",
                "team",
                "opponent",
                "is_home",
                "prop_type",
                "actual_value",
                "minutes_played",
                "usage_rate",
                "days_rest",
            ]
        )

    base_logs = player_logs.sort_values(["PLAYER_NAME", "GAME_DATE"]).copy()
    base_logs["days_rest"] = (
        base_logs.groupby("PLAYER_NAME")["GAME_DATE"].diff().dt.days.sub(1).clip(lower=0).fillna(2)
    )
    expanded = _expand_logs_by_prop_type(base_logs)
    if expanded.empty:
        return pd.DataFrame()

    expanded = expanded.merge(
        base_logs[["PLAYER_NAME", "GAME_DATE", "days_rest"]],
        on=["PLAYER_NAME", "GAME_DATE"],
        how="left",
    )
    expanded = expanded.rename(
        columns={
            "PLAYER_NAME": "player_name",
            "TEAM_ABBREVIATION": "team",
            "GAME_DATE": "game_date",
            "OPPONENT": "opponent",
            "IS_HOME": "is_home",
            "MIN": "minutes_played",
        }
    )
    expanded["sport"] = APP_CONFIG.default_sport
    expanded["usage_rate"] = np.nan
    expanded["game_date"] = _normalize_calendar_date(expanded["game_date"])
    expanded["game_id"] = expanded.apply(
        lambda row: build_game_id(row.get("game_date"), row.get("player_name"), row.get("team"), row.get("prop_type")),
        axis=1,
    )
    return expanded[
        [
            "game_id",
            "game_date",
            "sport",
            "player_name",
            "team",
            "opponent",
            "is_home",
            "prop_type",
            "actual_value",
            "minutes_played",
            "usage_rate",
            "days_rest",
        ]
    ].drop_duplicates(subset=["game_id", "player_name", "prop_type"], keep="last").reset_index(drop=True)


def build_live_player_feature_snapshot(player_logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if player_logs.empty:
        return pd.DataFrame(), pd.DataFrame()

    prop_logs = _expand_logs_by_prop_type(player_logs)
    if prop_logs.empty:
        return pd.DataFrame(), pd.DataFrame()

    league_avg = (
        prop_logs.groupby("prop_type", dropna=False)["actual_value"]
        .mean()
        .rename("league_avg")
        .reset_index()
    )
    opponent_snapshot = (
        prop_logs.groupby(["prop_type", "OPPONENT"], dropna=False)
        .agg(
            opponent_sample=("actual_value", "size"),
            opponent_avg_allowed=("actual_value", "mean"),
        )
        .reset_index()
    )
    opponent_snapshot = opponent_snapshot.merge(league_avg, on="prop_type", how="left")
    opponent_snapshot["matchup_allowance_delta"] = opponent_snapshot["opponent_avg_allowed"] - opponent_snapshot["league_avg"]
    opponent_snapshot["matchup_difficulty_score"] = clamp(50.0 + opponent_snapshot["matchup_allowance_delta"] * 8.0)
    opponent_snapshot = opponent_snapshot.rename(columns={"OPPONENT": "opponent"})
    opponent_snapshot = opponent_snapshot.drop(columns=["league_avg"])

    feature_rows: list[dict[str, Any]] = []
    for (player_name, prop_type), group in prop_logs.groupby(["PLAYER_NAME", "prop_type"], dropna=False):
        ordered = group.sort_values("GAME_DATE").copy()
        last_5 = ordered.tail(5)
        last_10 = ordered.tail(10)
        rolling_avg_5 = last_5["actual_value"].mean()
        rolling_avg_10 = last_10["actual_value"].mean()
        recent_std_10 = last_10["actual_value"].std(ddof=0)
        recent_variance_10 = recent_std_10**2 if pd.notna(recent_std_10) else np.nan
        coefficient = (recent_std_10 / max(abs(rolling_avg_10), 1.0)) if pd.notna(recent_std_10) and pd.notna(rolling_avg_10) else np.nan
        consistency_score = clamp((1.0 - min(max(coefficient, 0.0), 1.5) / 1.5) * 100.0) if pd.notna(coefficient) else 50.0

        feature_rows.append(
            {
                "player_name": player_name,
                "prop_type": prop_type,
                "player_id": ordered["PLAYER_ID"].iloc[-1],
                "team": ordered["TEAM_ABBREVIATION"].iloc[-1],
                "last_game_date": ordered["GAME_DATE"].iloc[-1],
                "games_in_sample": int(len(ordered)),
                "rolling_avg_5": rolling_avg_5,
                "rolling_avg_10": rolling_avg_10,
                "rolling_median_10": last_10["actual_value"].median(),
                "season_avg_prior": ordered["actual_value"].mean(),
                "recent_std_10": recent_std_10,
                "recent_variance_10": recent_variance_10,
                "consistency_score": consistency_score,
                "trend_direction": rolling_avg_5 - rolling_avg_10 if pd.notna(rolling_avg_5) and pd.notna(rolling_avg_10) else 0.0,
            }
        )

    player_snapshot = pd.DataFrame(feature_rows)
    return player_snapshot, opponent_snapshot


def enrich_live_current_props(
    current_props: pd.DataFrame,
    player_snapshot: pd.DataFrame,
    opponent_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if current_props.empty:
        return current_props.copy()

    data = current_props.copy()
    data["home_team"] = data["home_team_name"].astype(str).str.lower().map(TEAM_NAME_TO_ABBR)
    data["away_team"] = data["away_team_name"].astype(str).str.lower().map(TEAM_NAME_TO_ABBR)
    enriched = data.merge(player_snapshot, on=["player_name", "prop_type"], how="left")
    enriched["team"] = enriched["team"].fillna("")
    enriched["is_home"] = enriched["team"] == enriched["home_team"]
    enriched["opponent"] = np.where(enriched["is_home"], enriched["away_team"], enriched["home_team"])
    enriched["game_date"] = _normalize_calendar_date(enriched["game_date"])
    enriched["sport"] = APP_CONFIG.default_sport
    enriched["market_type"] = APP_CONFIG.default_market_type
    enriched = enriched.merge(opponent_snapshot, on=["prop_type", "opponent"], how="left")
    enriched["line_minus_recent_avg"] = enriched["line"] - enriched["rolling_avg_10"]
    enriched["line_minus_season_avg"] = enriched["line"] - enriched["season_avg_prior"]
    enriched["line_minus_recent_median"] = enriched["line"] - enriched["rolling_median_10"]
    enriched["player_sample_size"] = enriched["games_in_sample"].fillna(0)
    enriched["opponent_sample_size"] = enriched["opponent_sample"].fillna(0)
    enriched["opponent_actual_minus_line_mean"] = enriched["matchup_allowance_delta"].fillna(0.0)
    enriched["matchup_difficulty_score"] = enriched["matchup_difficulty_score"].fillna(50.0)
    return enriched


def score_live_current_props(
    current_props: pd.DataFrame,
    season: str | None = None,
    historical_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if current_props.empty:
        return current_props.copy()

    logs = fetch_season_player_logs(season=season)
    player_snapshot, opponent_snapshot = build_live_player_feature_snapshot(logs)
    enriched = enrich_live_current_props(current_props, player_snapshot, opponent_snapshot)
    if historical_features is not None and not historical_features.empty:
        enriched = attach_reference_tables(enriched, historical_features)
    scored = score_current_props(enriched)

    consensus = (
        scored.groupby(["event_id", "player_name", "prop_type"], dropna=False)
        .agg(
            consensus_line=("line", "median"),
            book_count=("bookmaker", "nunique"),
        )
        .reset_index()
    )
    scored = scored.merge(consensus, on=["event_id", "player_name", "prop_type"], how="left")
    scored["shopping_edge_points"] = np.select(
        [
            scored["lean"] == "Over",
            scored["lean"] == "Under",
        ],
        [
            scored["consensus_line"] - scored["line"],
            scored["line"] - scored["consensus_line"],
        ],
        default=0.0,
    )
    scored["shopping_bonus"] = clamp(50.0 + scored["shopping_edge_points"].fillna(0.0) * 20.0) - 50.0
    scored["scanner_score"] = clamp(scored["overall_prop_analysis_score"] + scored["shopping_bonus"]).round(1)
    return scored.sort_values(["scanner_score", "overall_prop_analysis_score", "shopping_edge_points", "prop_type"], ascending=[False, False, False, True]).reset_index(drop=True)


def _load_pending_prop_archive() -> pd.DataFrame:
    if PENDING_PROP_ARCHIVE.exists():
        return pd.read_parquet(PENDING_PROP_ARCHIVE)
    return pd.DataFrame()


def _save_pending_prop_archive(df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PENDING_PROP_ARCHIVE, index=False)


def build_pending_prop_archive_rows(live_scored: pd.DataFrame) -> pd.DataFrame:
    if live_scored.empty:
        return pd.DataFrame()
    required_columns = {"event_id", "player_name", "team", "opponent", "is_home", "prop_type", "line"}
    if not required_columns.issubset(live_scored.columns):
        return pd.DataFrame()

    archive = (
        live_scored.groupby(
            ["event_id", "player_name", "team", "opponent", "is_home", "prop_type"],
            dropna=False,
        )
        .agg(
            game_date=("game_date", "first"),
            closing_line=("line", "median"),
            opening_line=("line", "first"),
            consensus_line=("consensus_line", "median"),
            over_odds=("over_odds", "median"),
            under_odds=("under_odds", "median"),
            book_count=("bookmaker", "nunique"),
            market_last_update=("market_last_update", "max"),
        )
        .reset_index()
    )
    archive["game_date"] = _normalize_calendar_date(archive["game_date"])
    archive["sport"] = APP_CONFIG.default_sport
    archive["market_type"] = APP_CONFIG.default_market_type
    archive["bookmaker"] = "auto_consensus"
    archive["captured_at"] = pd.Timestamp.utcnow()
    return archive


def sync_auto_history_from_live(
    live_scored: pd.DataFrame,
    season: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, int]]:
    if live_scored.empty:
        return None, None, {"resolved_rows": 0, "pending_rows": 0}

    pending_existing = _load_pending_prop_archive()
    if not pending_existing.empty and "game_date" in pending_existing.columns:
        pending_existing["game_date"] = _normalize_calendar_date(pending_existing["game_date"])
    pending_new = build_pending_prop_archive_rows(live_scored)
    if pending_new.empty:
        return None, None, {"resolved_rows": 0, "pending_rows": 0}
    pending_keys = ["event_id", "player_name", "prop_type"]
    pending_archive = (
        pd.concat([pending_existing, pending_new], ignore_index=True)
        .sort_values("captured_at")
        .drop_duplicates(subset=pending_keys, keep="last")
        .reset_index(drop=True)
    )

    player_logs = fetch_season_player_logs(season=season)
    auto_results = build_auto_game_results_from_logs(player_logs)
    if auto_results.empty:
        _save_pending_prop_archive(pending_archive)
        return None, None, {"resolved_rows": 0, "pending_rows": len(pending_archive)}

    pending_archive = pending_archive.reset_index(drop=True)
    pending_archive["archive_row_id"] = pending_archive.index

    candidates = pending_archive.merge(
        auto_results,
        on=["player_name", "team", "opponent", "is_home", "prop_type"],
        how="left",
        suffixes=("_pending", "_result"),
    )
    if candidates.empty:
        _save_pending_prop_archive(pending_archive.drop(columns=["archive_row_id"], errors="ignore"))
        return None, None, {"resolved_rows": 0, "pending_rows": len(pending_archive)}

    pending_dates = _normalize_calendar_date(candidates["game_date_pending"])
    result_dates = _normalize_calendar_date(candidates["game_date_result"])
    candidates["date_gap_days"] = pending_dates.sub(result_dates).dt.days.abs()
    resolved_matches = (
        candidates.loc[candidates["date_gap_days"].fillna(99) <= 1]
        .sort_values(["archive_row_id", "date_gap_days", "game_date_result"])
        .drop_duplicates(subset=["archive_row_id"], keep="first")
        .reset_index(drop=True)
    )
    if resolved_matches.empty:
        _save_pending_prop_archive(pending_archive.drop(columns=["archive_row_id"], errors="ignore"))
        return None, None, {"resolved_rows": 0, "pending_rows": len(pending_archive)}

    resolved_props = resolved_matches.rename(
        columns={
            "game_date_result": "game_date",
            "sport_result": "sport",
        }
    )
    resolved_props["game_id"] = resolved_props["game_id"]
    historical_props = resolved_props[
        [
            "game_id",
            "game_date",
            "sport",
            "player_name",
            "team",
            "opponent",
            "is_home",
            "prop_type",
            "market_type",
            "opening_line",
            "closing_line",
            "over_odds",
            "under_odds",
            "bookmaker",
        ]
    ].copy()

    game_results = resolved_matches.rename(columns={"game_date_result": "game_date"})[
        [
            "game_id",
            "game_date",
            "sport_result",
            "player_name",
            "team",
            "opponent",
            "is_home",
            "prop_type",
            "actual_value",
            "minutes_played",
            "usage_rate",
            "days_rest",
        ]
    ].copy()
    game_results = game_results.rename(columns={"sport_result": "sport"})

    existing_props, existing_results = load_cached_history()
    existing_props = existing_props if existing_props is not None else pd.DataFrame(columns=historical_props.columns)
    existing_results = existing_results if existing_results is not None else pd.DataFrame(columns=game_results.columns)

    combined_props = (
        pd.concat([existing_props, historical_props], ignore_index=True)
        .drop_duplicates(subset=["game_id", "player_name", "prop_type"], keep="last")
        .sort_values(["game_date", "player_name", "prop_type"])
        .reset_index(drop=True)
    )
    combined_results = (
        pd.concat([existing_results, game_results], ignore_index=True)
        .drop_duplicates(subset=["game_id", "player_name", "prop_type"], keep="last")
        .sort_values(["game_date", "player_name", "prop_type"])
        .reset_index(drop=True)
    )
    save_cached_history(combined_props, combined_results)

    unresolved_archive = pending_archive.loc[
        ~pending_archive["archive_row_id"].isin(resolved_matches["archive_row_id"])
    ].drop(columns=["archive_row_id"], errors="ignore")
    _save_pending_prop_archive(unresolved_archive.reset_index(drop=True))
    return combined_props, combined_results, {"resolved_rows": len(historical_props), "pending_rows": len(unresolved_archive)}


def save_cached_history(historical_props: pd.DataFrame, game_results: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    historical_props.to_parquet(CACHED_HISTORY_PROPS, index=False)
    game_results.to_parquet(CACHED_HISTORY_RESULTS, index=False)


def load_cached_history() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if CACHED_HISTORY_PROPS.exists() and CACHED_HISTORY_RESULTS.exists():
        historical_props = pd.read_parquet(CACHED_HISTORY_PROPS)
        game_results = pd.read_parquet(CACHED_HISTORY_RESULTS)
        if "game_date" in historical_props.columns:
            historical_props["game_date"] = _normalize_calendar_date(historical_props["game_date"])
        if "game_date" in game_results.columns:
            game_results["game_date"] = _normalize_calendar_date(game_results["game_date"])
        return historical_props, game_results
    return None, None
