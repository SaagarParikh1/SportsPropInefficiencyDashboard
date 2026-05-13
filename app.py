from __future__ import annotations

import html
import os
from typing import Any

import pandas as pd
import streamlit as st

from src.config import APP_CONFIG, PROP_MARKET_MAP, REQUIRED_COLUMNS
from src.data_cleaning import clean_game_results, clean_historical_props, prepare_historical_dataset
from src.data_ingestion import load_demo_data
from src.feature_engineering import build_historical_features
from src.line_evaluation import collect_segment_tables, overall_market_metrics, rolling_edge_summary, temporal_stability_by_segment
from src.live_data import (
    current_basketball_season,
    fetch_daily_basketball_games,
    fetch_live_basketball_player_props,
    load_cached_history,
    normalize_sport,
    score_live_current_props,
    sync_auto_history_from_live,
)
from src.settings_store import load_user_settings, save_user_settings
from src.utils import describe_prop_type, format_pct, format_prop_type
from src.visualization import (
    build_component_score_chart,
    build_line_vs_actual_scatter,
    build_player_history_chart,
    build_residual_distribution,
    build_rolling_edge_chart,
    build_score_ranking_chart,
    build_segment_bar_chart,
)


st.set_page_config(
    page_title="Prop Edge Board",
    page_icon=":bar_chart:",
    layout="wide",
)


SEGMENT_OPTIONS = {
    "Prop market": {"segment_key": "prop_type", "table_key": "Prop Type"},
    "Line range": {"segment_key": "line_range_bucket", "table_key": "Line Range"},
    "Player": {"segment_key": "player_name", "table_key": "Player"},
    "Team": {"segment_key": "team", "table_key": "Team"},
    "Opponent": {"segment_key": "opponent", "table_key": "Opponent"},
    "Venue": {"segment_key": "is_home", "table_key": "Home/Away"},
    "Rest": {"segment_key": "days_rest_bucket", "table_key": "Rest Bucket"},
}

BOOKMAKER_OPTIONS = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
    "espnbet": "ESPN BET",
    "fliff": "Fliff",
    "williamhill_us": "Caesars",
}

LEAGUE_OPTIONS = ["NBA", "WNBA"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Rajdhani:wght@600;700&display=swap');

            :root {
                --bg: #080807;
                --panel: #11130f;
                --panel-2: #181b14;
                --line: rgba(242, 238, 218, 0.12);
                --text: #f7f2df;
                --muted: #a8a38f;
                --green: #85ff4c;
                --teal: #3ff5d5;
                --amber: #ffbf38;
                --red: #ff5f57;
                --violet: #b891ff;
                --pink: #ff6dc7;
                --orange: #ff8a2a;
            }

            .stApp {
                background:
                    linear-gradient(180deg, rgba(255,191,56,0.07), rgba(8,8,7,0) 260px),
                    repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0, rgba(255,255,255,0.018) 1px, transparent 1px, transparent 62px),
                    var(--bg);
                color: var(--text);
                font-family: "IBM Plex Sans", sans-serif;
            }

            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            div[data-testid="stStatusWidget"],
            #MainMenu,
            footer {
                display: none !important;
                visibility: hidden !important;
            }

            .block-container {
                max-width: 1440px;
                padding-top: 4rem;
                padding-bottom: 2.4rem;
            }

            section[data-testid="stSidebar"] {
                background: #0c0e0b;
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] span {
                color: rgba(243,245,237,0.78);
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            .app-header {
                display: flex;
                justify-content: space-between;
                gap: 1.5rem;
                align-items: flex-end;
                padding: 1.05rem 1.1rem 1rem 1.1rem;
                margin-bottom: 0.85rem;
                border: 1px solid var(--line);
                border-radius: 10px;
                background:
                    linear-gradient(135deg, rgba(255,191,56,0.12), transparent 38%),
                    linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012)),
                    #10110e;
            }

            .app-title {
                font-family: "Rajdhani", sans-serif;
                font-size: 3.35rem;
                line-height: 0.95;
                color: var(--text);
                text-transform: uppercase;
                margin: 0;
            }

            .app-subtitle {
                color: var(--muted);
                font-size: 0.95rem;
                margin-top: 0.3rem;
            }

            .status-strip {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.42rem;
                max-width: 620px;
            }

            .status-chip {
                display: inline-flex;
                align-items: center;
                min-height: 28px;
                border: 1px solid var(--line);
                background: rgba(8,8,7,0.36);
                color: rgba(243,245,237,0.78);
                border-radius: 6px;
                padding: 0.28rem 0.55rem;
                font-size: 0.78rem;
                white-space: nowrap;
            }

            .games-band {
                border: 1px solid var(--line);
                border-radius: 10px;
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01)),
                    #10120f;
                padding: 0.85rem;
                margin-bottom: 1.05rem;
            }

            .games-head {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: baseline;
                margin-bottom: 0.6rem;
            }

            .games-title {
                color: var(--text);
                font-weight: 700;
                font-size: 1rem;
            }

            .games-date {
                color: var(--muted);
                text-transform: uppercase;
                font-size: 0.72rem;
            }

            .games-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 0.65rem;
            }

            .game-tile {
                border: 1px solid rgba(242,238,218,0.12);
                border-radius: 9px;
                background:
                    linear-gradient(135deg, rgba(133,255,76,0.08), transparent 44%),
                    #0b0d0a;
                padding: 0.72rem;
                overflow: hidden;
            }

            .game-matchup {
                display: grid;
                grid-template-columns: 1fr auto 1fr;
                align-items: center;
                gap: 0.6rem;
                color: var(--text);
                font-family: "Rajdhani", sans-serif;
                font-size: 1.45rem;
                line-height: 1;
            }

            .team-side {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                min-width: 0;
            }

            .team-side.home-team {
                justify-content: flex-end;
                text-align: right;
            }

            .team-logo {
                width: 42px;
                height: 42px;
                object-fit: contain;
                flex: 0 0 auto;
                filter: drop-shadow(0 5px 10px rgba(0,0,0,0.45));
            }

            .team-code {
                color: var(--text);
                white-space: nowrap;
            }

            .team-name {
                color: var(--muted);
                font-family: "IBM Plex Sans", sans-serif;
                font-size: 0.68rem;
                line-height: 1.05;
                margin-top: 0.16rem;
                max-width: 96px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .versus-mark {
                color: var(--amber);
                border: 1px solid rgba(255,191,56,0.28);
                border-radius: 6px;
                padding: 0.22rem 0.36rem;
                font-size: 0.82rem;
                font-family: "IBM Plex Sans", sans-serif;
                font-weight: 700;
            }

            .game-meta {
                display: flex;
                justify-content: space-between;
                gap: 0.8rem;
                color: var(--muted);
                font-size: 0.78rem;
                margin-top: 0.62rem;
            }

            .empty-strip {
                border: 1px dashed rgba(232,238,222,0.16);
                border-radius: 7px;
                color: var(--muted);
                padding: 0.75rem;
                font-size: 0.9rem;
            }

            .board-head {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 1rem;
                margin: 0.4rem 0 0.7rem 0;
            }

            .board-title {
                font-family: "Rajdhani", sans-serif;
                color: var(--text);
                font-size: 2.15rem;
                line-height: 1;
                text-transform: uppercase;
            }

            .board-note {
                color: var(--muted);
                font-size: 0.86rem;
                margin-top: 0.2rem;
            }

            .board-count {
                color: var(--green);
                font-family: "Rajdhani", sans-serif;
                font-size: 1.2rem;
                white-space: nowrap;
            }

            .detail-panel {
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01)),
                    var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.95rem;
                margin-bottom: 0.65rem;
            }

            .detail-kicker {
                color: var(--muted);
                text-transform: uppercase;
                font-size: 0.68rem;
                margin-bottom: 0.32rem;
            }

            .detail-title {
                color: var(--text);
                font-family: "Rajdhani", sans-serif;
                font-size: 1.9rem;
                line-height: 1;
                text-transform: uppercase;
            }

            .detail-copy {
                color: rgba(243,245,237,0.78);
                line-height: 1.5;
                font-size: 0.9rem;
            }

            .inspector-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.48rem;
                margin-top: 0.75rem;
            }

            .mini-stat {
                border: 1px solid var(--line);
                border-radius: 6px;
                padding: 0.52rem 0.55rem;
                background: rgba(255,255,255,0.025);
            }

            .mini-label {
                color: var(--muted);
                font-size: 0.7rem;
                text-transform: uppercase;
            }

            .mini-value {
                color: var(--text);
                font-weight: 700;
                margin-top: 0.16rem;
            }

            .section-title {
                color: var(--text);
                font-family: "Rajdhani", sans-serif;
                text-transform: uppercase;
                font-size: 1.8rem;
                margin: 0.3rem 0 0.18rem 0;
            }

            .section-copy {
                color: var(--muted);
                line-height: 1.5;
                margin-bottom: 0.7rem;
                max-width: 62rem;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin: 0.55rem 0 0.7rem 0;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                border-radius: 6px;
                padding: 0.28rem 0.54rem;
                font-size: 0.76rem;
                border: 1px solid transparent;
            }

            .badge-lime {
                background: rgba(133,255,76,0.1);
                color: var(--green);
                border-color: rgba(133,255,76,0.22);
            }

            .badge-pink {
                background: rgba(255,109,199,0.1);
                color: var(--pink);
                border-color: rgba(255,109,199,0.22);
            }

            .badge-gold {
                background: rgba(255,191,56,0.1);
                color: var(--amber);
                border-color: rgba(255,191,56,0.22);
            }

            .badge-red {
                background: rgba(255,95,87,0.1);
                color: var(--red);
                border-color: rgba(255,95,87,0.22);
            }

            .badge-cyan {
                background: rgba(63,245,213,0.1);
                color: var(--teal);
                border-color: rgba(63,245,213,0.22);
            }

            .badge-muted {
                background: rgba(255,255,255,0.05);
                color: var(--muted);
                border-color: rgba(255,255,255,0.08);
            }

            .notice {
                border-radius: 8px;
                padding: 0.72rem 0.8rem;
                margin: 0.35rem 0 0.75rem 0;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.03);
            }

            .notice.notice-gold {
                border-color: rgba(255,191,56,0.18);
                background: rgba(255,191,56,0.07);
            }

            .notice.notice-red {
                border-color: rgba(255,95,87,0.2);
                background: rgba(255,95,87,0.08);
            }

            .notice.notice-cyan {
                border-color: rgba(63,245,213,0.16);
                background: rgba(63,245,213,0.06);
            }

            .notice-title {
                color: var(--text);
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .notice-copy {
                color: rgba(243,245,237,0.76);
                line-height: 1.45;
                font-size: 0.9rem;
            }

            .method-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.65rem;
                margin-bottom: 1rem;
            }

            .method-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem 0.9rem;
            }

            .method-step {
                color: var(--green);
                font-family: "Rajdhani", sans-serif;
                text-transform: uppercase;
                font-size: 1.05rem;
            }

            .method-label {
                color: var(--text);
                font-weight: 700;
                margin: 0.2rem 0 0.35rem 0;
            }

            .method-copy {
                color: rgba(243,245,237,0.72);
                line-height: 1.45;
                font-size: 0.92rem;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }

            button[kind="primary"] {
                border-radius: 6px !important;
                font-weight: 700 !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                border-bottom: 1px solid var(--line);
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 6px 6px 0 0;
                padding: 0.55rem 0.9rem;
            }

            @media (max-width: 1100px) {
                .app-header {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .status-strip {
                    justify-content: flex-start;
                }

                .method-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badges(badges: list[tuple[str, str]]) -> None:
    if not badges:
        return
    markup = "".join(
        f'<span class="badge badge-{html.escape(tone)}">{html.escape(label)}</span>'
        for label, tone in badges
    )
    st.markdown(f'<div class="badge-row">{markup}</div>', unsafe_allow_html=True)


def render_metric_row(cards: list[dict[str, str]]) -> None:
    if not cards:
        return
    columns = st.columns(len(cards), gap="small")
    for column, card in zip(columns, cards):
        with column:
            st.metric(card["label"], card["value"])
            st.caption(card["note"])


def render_notice(title: str, body: str, tone: str = "gold") -> None:
    st.markdown(
        f"""
        <div class="notice notice-{html.escape(tone)}">
            <div class="notice-title">{html.escape(title)}</div>
            <div class="notice-copy">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">{html.escape(title)}</div>
        <div class="section-copy">{html.escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(status_items: list[str], league: str) -> None:
    chips = "".join(f'<span class="status-chip">{html.escape(item)}</span>' for item in status_items if item)
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-title">Prop Edge Board</div>
                <div class="app-subtitle">Live {html.escape(league)} prop market board for points, rebounds, assists, and combo lines.</div>
            </div>
            <div class="status-strip">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_games_strip(games: pd.DataFrame, game_date_label: str, league: str, error: str = "") -> None:
    if error:
        body = f'<div class="empty-strip">Today\'s games could not be loaded: {html.escape(error)}</div>'
    elif games.empty:
        body = f'<div class="empty-strip">No {html.escape(league)} games are scheduled for this date.</div>'
    else:
        tiles = []
        for _, row in games.iterrows():
            away = html.escape(str(row.get("away_team", "")))
            home = html.escape(str(row.get("home_team", "")))
            away_name = html.escape(str(row.get("away_team_name", "")))
            home_name = html.escape(str(row.get("home_team_name", "")))
            away_logo = html.escape(str(row.get("away_logo_url", "")))
            home_logo = html.escape(str(row.get("home_logo_url", "")))
            time_text = html.escape(str(row.get("game_time", "")))
            series = html.escape(str(row.get("series_text", "")) or str(row.get("game_label", "")))
            score_text = ""
            if int(row.get("away_score", 0) or 0) or int(row.get("home_score", 0) or 0):
                score_text = f'{int(row.get("away_score", 0) or 0)} - {int(row.get("home_score", 0) or 0)}'
            meta_right = html.escape(score_text or series or str(row.get("status", "")))
            tiles.append(
                f"""
                <div class="game-tile">
                    <div class="game-matchup">
                        <div class="team-side">
                            <img class="team-logo" src="{away_logo}" alt="{away} logo">
                            <div>
                                <div class="team-code">{away}</div>
                                <div class="team-name">{away_name}</div>
                            </div>
                        </div>
                        <div class="versus-mark">@</div>
                        <div class="team-side home-team">
                            <div>
                                <div class="team-code">{home}</div>
                                <div class="team-name">{home_name}</div>
                            </div>
                            <img class="team-logo" src="{home_logo}" alt="{home} logo">
                        </div>
                    </div>
                    <div class="game-meta"><span>{time_text}</span><span>{meta_right}</span></div>
                </div>
                """
            )
        body = f'<div class="games-grid">{"".join(tiles)}</div>'
    st.markdown(
        f"""
        <div class="games-band">
            <div class="games-head">
                <div class="games-title">Today's {html.escape(league)} games</div>
                <div class="games-date">{html.escape(game_date_label)}</div>
            </div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_board_header(row_count: int) -> None:
    st.markdown(
        f"""
        <div class="board-head">
            <div>
                <div class="board-title">Top 12 Live Props</div>
                <div class="board-note">Ranked by support score, line value, current form, matchup context, and book comparison.</div>
            </div>
            <div class="board-count">{row_count} shown</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_prop_card(row: pd.Series) -> None:
    venue_text = f"vs {row['opponent']}" if bool(row.get("is_home")) else f"at {row['opponent']}"
    prop_label = format_prop_type(row.get("prop_type"))
    team_text = str(row.get("team", ""))
    support_text = str(row.get("support_label", "Unrated"))
    rolling_text = format_number(row.get("rolling_avg_10"))
    consensus_text = format_number(row.get("consensus_line", row.get("line")))
    score_text = format_number(row.get("scanner_score", row.get("overall_prop_analysis_score", 0.0)))
    st.markdown(
        f"""
        <div class="detail-panel">
            <div class="detail-kicker">Selected prop</div>
            <div class="detail-title">{html.escape(str(row['lean']))} {float(row['line']):.1f} {html.escape(prop_label)}</div>
            <div class="detail-copy">
                <strong>{html.escape(str(row['player_name']))}</strong> · {html.escape(team_text)} · {html.escape(venue_text)}
                <br>Best book: <strong>{html.escape(str(row.get('bookmaker', 'Unknown')))}</strong> · Support: <strong>{html.escape(support_text)}</strong>
            </div>
            <div class="inspector-grid">
                <div class="mini-stat"><div class="mini-label">Score</div><div class="mini-value">{html.escape(score_text)}</div></div>
                <div class="mini-stat"><div class="mini-label">10G avg</div><div class="mini-value">{html.escape(rolling_text)}</div></div>
                <div class="mini-stat"><div class="mini-label">Consensus</div><div class="mini-value">{html.escape(consensus_text)}</div></div>
                <div class="mini-stat"><div class="mini-label">Book count</div><div class="mini-value">{int(row.get('book_count', 0) or 0)}</div></div>
            </div>
        </div>
        <div class="detail-panel">
            <div class="detail-kicker">Read</div>
            <div class="detail-copy">{html.escape(str(row.get('explanation_text', 'No explanation available.')))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_games_cached(game_date: str, league: str) -> pd.DataFrame:
    return fetch_daily_basketball_games(game_date=game_date, timezone_name="America/Chicago", sport=league)


def lean_text_style(value: Any) -> str:
    if value == "Over":
        return "color: #85ff4c; font-weight: 700;"
    if value == "Under":
        return "color: #ff5f57; font-weight: 700;"
    return "color: #ffbf38; font-weight: 600;"


def prop_text_style(value: Any) -> str:
    palette = {
        "PTS": "color: #3ff5d5; font-weight: 700;",
        "REB": "color: #85ff4c; font-weight: 700;",
        "AST": "color: #ffbf38; font-weight: 700;",
        "PTS+REB": "color: #b891ff; font-weight: 700;",
        "PTS+AST": "color: #ff6dc7; font-weight: 700;",
        "REB+AST": "color: #34d399; font-weight: 700;",
        "PRA": "color: #ff8a2a; font-weight: 700;",
    }
    return palette.get(str(value), "color: #d3d8e1; font-weight: 600;")


def style_live_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styler = df.style
    if "lean" in df.columns:
        styler = styler.map(lean_text_style, subset=["lean"])
    if "prop_label" in df.columns:
        styler = styler.map(prop_text_style, subset=["prop_label"])
    if "prop_type" in df.columns:
        styler = styler.map(lambda value: prop_text_style(format_prop_type(value)), subset=["prop_type"])
    return styler


def ensure_live_display_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    prepared = df.copy()
    if "prop_label" not in prepared.columns and "prop_type" in prepared.columns:
        prepared["prop_label"] = prepared["prop_type"].map(format_prop_type)
    if "player_display" not in prepared.columns and {"player_name", "prop_label"}.issubset(prepared.columns):
        prepared["player_display"] = prepared.apply(
            lambda row: f"{row['player_name']} {row['prop_label']}",
            axis=1,
        )
    if (
        "display_label" not in prepared.columns
        and {"player_name", "prop_label", "lean", "line", "bookmaker"}.issubset(prepared.columns)
    ):
        prepared["display_label"] = prepared.apply(
            lambda row: f"{row['player_name']} | {row['prop_label']} | {row['lean']} {float(row['line']):.1f} | {row['bookmaker']}",
            axis=1,
        )
    return prepared


def build_history_pipeline(historical_props: pd.DataFrame, game_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cleaned_props = clean_historical_props(historical_props)
    cleaned_results = clean_game_results(game_results)
    historical = prepare_historical_dataset(cleaned_props, cleaned_results)
    historical_features = build_historical_features(historical)
    return {
        "historical_props": cleaned_props,
        "game_results": cleaned_results,
        "historical": historical,
        "historical_features": historical_features,
    }


@st.cache_data(show_spinner=False)
def load_demo_history_pipeline() -> dict[str, pd.DataFrame]:
    demo = load_demo_data()
    return build_history_pipeline(demo["historical_props"], demo["game_results"])


def build_empty_history_pipeline() -> dict[str, pd.DataFrame]:
    empty_props = pd.DataFrame(columns=REQUIRED_COLUMNS["historical_props"] + ["sport", "market_type", "opening_line", "over_odds", "under_odds", "bookmaker"])
    empty_results = pd.DataFrame(columns=REQUIRED_COLUMNS["game_results"] + ["sport", "minutes_played", "usage_rate", "days_rest"])
    return build_history_pipeline(empty_props, empty_results)


def filter_history_for_league(history_features: pd.DataFrame, league: str) -> pd.DataFrame:
    selected_league = normalize_sport(league)
    if history_features.empty:
        return history_features.copy()
    if "sport" not in history_features.columns:
        return history_features.copy() if selected_league == "NBA" else history_features.iloc[0:0].copy()
    league_mask = history_features["sport"].fillna("NBA").astype(str).str.upper() == selected_league
    return history_features.loc[league_mask].reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_props_cached(api_key: str, bookmakers: tuple[str, ...], days_ahead: int, prop_types: tuple[str, ...], league: str) -> pd.DataFrame:
    keys = list(bookmakers) if bookmakers else None
    return fetch_live_basketball_player_props(
        api_key=api_key,
        bookmakers=keys,
        days_ahead=days_ahead,
        prop_types=list(prop_types),
        sport=league,
    )


@st.cache_data(ttl=900, show_spinner=False)
def score_live_current_props_cached(
    current_props: pd.DataFrame,
    season: str,
    historical_features: pd.DataFrame | None,
    league: str,
) -> pd.DataFrame:
    history_input = historical_features if historical_features is not None and not historical_features.empty else None
    return score_live_current_props(
        current_props,
        season=season,
        historical_features=history_input,
        sport=league,
    )


def build_schema_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset_name, columns in REQUIRED_COLUMNS.items():
        for column in columns:
            rows.append({"dataset": dataset_name, "required_column": column})
    return pd.DataFrame(rows)


def apply_team_venue_filters(df: pd.DataFrame, teams: list[str], venue: str, prop_types: list[str] | None = None) -> pd.DataFrame:
    filtered = df.copy()
    if teams and "team" in filtered.columns:
        filtered = filtered.loc[filtered["team"].isin(teams)]
    if prop_types and "prop_type" in filtered.columns:
        filtered = filtered.loc[filtered["prop_type"].isin(prop_types)]
    if venue == "Home" and "is_home" in filtered.columns:
        filtered = filtered.loc[filtered["is_home"]]
    if venue == "Away" and "is_home" in filtered.columns:
        filtered = filtered.loc[~filtered["is_home"]]
    return filtered.reset_index(drop=True)


def first_non_null(row: pd.Series, columns: list[str], default: float = 0.0) -> float:
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            return float(value)
    return default


def format_number(value: Any, digits: int = 1, default: str = "N/A") -> str:
    if pd.isna(value):
        return default
    return f"{float(value):.{digits}f}"


def historical_reference_size(row: pd.Series) -> float:
    return first_non_null(
        row,
        ["context_sample_size", "player_sample_size", "line_bucket_sample_size", "opponent_sample_size"],
        0.0,
    )


def live_warning_badges(row: pd.Series) -> list[tuple[str, str]]:
    reference_sample = historical_reference_size(row)
    badges: list[tuple[str, str]] = []
    if reference_sample < 15:
        badges.append(("Small history sample", "red"))
    elif reference_sample < 30:
        badges.append(("Limited history sample", "gold"))
    if float(row.get("stability_score", 50.0)) < 55:
        badges.append(("Weak stability", "red"))
    elif float(row.get("stability_score", 50.0)) < 65:
        badges.append(("Needs caution", "gold"))
    if float(row.get("book_count", 0.0)) < 3:
        badges.append(("Thin book coverage", "gold"))
    if row.get("lean") == "Neutral":
        badges.append(("Neutral lean", "muted"))
    if not badges:
        badges.append(("Cleanest board read", "lime"))
    return badges


def segment_warning_text(row: pd.Series) -> str:
    warnings: list[str] = []
    ci_width = float(row.get("over_ci_high", 0.0) - row.get("over_ci_low", 0.0))
    if float(row.get("sample_size", 0.0)) < 20:
        warnings.append("Small sample")
    if ci_width > 0.18:
        warnings.append("Wide interval")
    if float(row.get("unique_players", 0.0)) <= 2 and float(row.get("sample_size", 0.0)) < 35:
        warnings.append("Concentrated on few players")
    return " | ".join(warnings) if warnings else "No major caution"


def stability_warning_text(row: pd.Series) -> str:
    sample_train = float(row.get("sample_size_train", 0.0))
    if sample_train < 20:
        return "Thin train sample"
    assessment = str(row.get("stability_assessment", ""))
    if assessment == "Relatively stable":
        return "Held up later"
    if assessment == "Direction held, strength shifted":
        return "Direction held, strength moved"
    if assessment == "Did not persist cleanly":
        return "Did not hold later"
    return "Needs review"


def prepare_segment_table(segment_df: pd.DataFrame, segment_key: str) -> pd.DataFrame:
    if segment_df.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "sample_size",
                "over_hit_rate_pct",
                "under_hit_rate_pct",
                "mean_error",
                "potential_inefficiency_flag",
                "warning",
            ]
        )

    data = segment_df.copy()
    if segment_key == "is_home":
        data["segment"] = data["is_home"].map({True: "Home", False: "Away"}).fillna("Unknown")
    elif segment_key == "prop_type":
        data["segment"] = data["prop_type"].map(format_prop_type)
    else:
        data["segment"] = data[segment_key].astype(str)
    data["over_hit_rate_pct"] = (data["over_hit_rate"] * 100.0).round(1)
    data["under_hit_rate_pct"] = (data["under_hit_rate"] * 100.0).round(1)
    data["warning"] = data.apply(segment_warning_text, axis=1)
    data = data.sort_values(["mean_error", "sample_size"], key=lambda series: series.abs() if series.name == "mean_error" else series, ascending=[False, False])
    return data[
        [
            "segment",
            "sample_size",
            "over_hit_rate_pct",
            "under_hit_rate_pct",
            "mean_error",
            "potential_inefficiency_flag",
            "warning",
        ]
    ].reset_index(drop=True)


def prepare_segment_chart_source(segment_df: pd.DataFrame, segment_key: str) -> tuple[pd.DataFrame, str]:
    if segment_df.empty:
        return segment_df.copy(), segment_key
    chart_df = segment_df.copy()
    if segment_key == "is_home":
        chart_df["segment"] = chart_df["is_home"].map({True: "Home", False: "Away"}).fillna("Unknown")
        return chart_df, "segment"
    if segment_key == "prop_type":
        chart_df["segment"] = chart_df["prop_type"].map(format_prop_type)
        return chart_df, "segment"
    return chart_df, segment_key


def prepare_stability_table(stability_df: pd.DataFrame, segment_key: str) -> pd.DataFrame:
    if stability_df.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "sample_size_train",
                "sample_size_test",
                "over_hit_rate_train_pct",
                "over_hit_rate_test_pct",
                "mean_error_train",
                "mean_error_test",
                "stability_assessment",
                "warning",
            ]
        )

    data = stability_df.copy()
    if segment_key == "is_home":
        data["segment"] = data["is_home"].map({True: "Home", False: "Away"}).fillna("Unknown")
    elif segment_key == "prop_type":
        data["segment"] = data["prop_type"].map(format_prop_type)
    else:
        data["segment"] = data[segment_key].astype(str)
    data["over_hit_rate_train_pct"] = (data["over_hit_rate_train"] * 100.0).round(1)
    data["over_hit_rate_test_pct"] = (data["over_hit_rate_test"] * 100.0).round(1)
    data["warning"] = data.apply(stability_warning_text, axis=1)
    data = data.sort_values(["sample_size_train", "mean_error_delta"], ascending=[False, True])
    return data[
        [
            "segment",
            "sample_size_train",
            "sample_size_test",
            "over_hit_rate_train_pct",
            "over_hit_rate_test_pct",
            "mean_error_train",
            "mean_error_test",
            "stability_assessment",
            "warning",
        ]
    ].reset_index(drop=True)


def build_live_board(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df.copy()

    ordered = ensure_live_display_fields(scored_df).sort_values(
        ["scanner_score", "shopping_edge_points", "overall_prop_analysis_score", "book_count"],
        ascending=[False, False, False, False],
    ).copy()
    board = (
        ordered.groupby(["event_id", "player_name", "prop_type"], dropna=False, as_index=False)
        .first()
        .sort_values(["scanner_score", "shopping_edge_points"], ascending=[False, False])
        .reset_index(drop=True)
    )
    board["rank"] = range(1, len(board) + 1)
    board["prop_label"] = board["prop_type"].map(format_prop_type)
    board["reference_sample"] = board.apply(historical_reference_size, axis=1)
    board["warning"] = board.apply(lambda row: " | ".join(label for label, _tone in live_warning_badges(row)), axis=1)
    board["matchup"] = board.apply(
        lambda row: f"vs {row['opponent']}" if bool(row.get("is_home")) else f"at {row['opponent']}",
        axis=1,
    )
    board["player_display"] = board.apply(
        lambda row: f"{row['player_name']} {row['prop_label']}",
        axis=1,
    )
    board["display_label"] = board.apply(
        lambda row: f"{row['player_name']} | {row['prop_label']} | {row['lean']} {float(row['line']):.1f} | {row['bookmaker']}",
        axis=1,
    )
    return board


def player_summary_cards(player_history: pd.DataFrame) -> list[dict[str, str]]:
    games = len(player_history)
    over_rate = player_history["over_hit"].mean() if games else 0.0
    avg_line = player_history["line"].mean() if games else 0.0
    avg_actual = player_history["actual_value"].mean() if games else 0.0
    mean_error = player_history["actual_minus_line"].mean() if games else 0.0
    prop_description = describe_prop_type(player_history["prop_type"].iloc[0] if games else "prop")
    return [
        {"label": "Games tracked", "value": str(games), "note": f"Historical player {prop_description} props in the selected data.", "tone": "cyan"},
        {"label": "Average line", "value": format_number(avg_line), "note": f"Average closing {prop_description} line.", "tone": "gold"},
        {"label": "Average actual", "value": format_number(avg_actual), "note": f"Average actual {prop_description}.", "tone": "lime"},
        {"label": "Mean line error", "value": format_number(mean_error), "note": "Positive means the player beat the line on average.", "tone": "pink"},
        {"label": "Over hit rate", "value": format_pct(over_rate), "note": "Pushes excluded by design of the over flag.", "tone": "cyan"},
    ]


def player_split_table(player_history: pd.DataFrame, split_col: str) -> pd.DataFrame:
    if player_history.empty:
        return pd.DataFrame()

    split = (
        player_history.groupby(split_col, dropna=False)
        .agg(
            games=("actual_value", "size"),
            avg_line=("line", "mean"),
            avg_actual=("actual_value", "mean"),
            mean_error=("actual_minus_line", "mean"),
            over_hit_rate=("over_hit", "mean"),
        )
        .reset_index()
    )
    if split_col == "is_home":
        split["segment"] = split["is_home"].map({True: "Home", False: "Away"}).fillna("Unknown")
    else:
        split["segment"] = split[split_col].astype(str)
    split["over_hit_rate_pct"] = (split["over_hit_rate"] * 100.0).round(1)
    return split[["segment", "games", "avg_line", "avg_actual", "mean_error", "over_hit_rate_pct"]].sort_values("games", ascending=False)


inject_styles()

saved_settings = load_user_settings()
saved_api_key = str(saved_settings.get("api_key") or os.getenv("THE_ODDS_API_KEY", "")).strip()
saved_bookmakers = [key for key in saved_settings.get("bookmakers", []) if key in BOOKMAKER_OPTIONS] or list(BOOKMAKER_OPTIONS.keys())[:5]
saved_fetch_prop_types = [prop for prop in saved_settings.get("prop_types", []) if prop in PROP_MARKET_MAP] or list(PROP_MARKET_MAP.keys())
saved_days_ahead = int(saved_settings.get("days_ahead", 1))
saved_league = normalize_sport(saved_settings.get("league", "NBA"))
auto_refresh_enabled = bool(saved_settings.get("auto_refresh", True))
auto_refresh_minutes = int(saved_settings.get("auto_refresh_minutes", 20))

if "last_history_sync_signature" not in st.session_state:
    st.session_state["last_history_sync_signature"] = None
if "last_live_sync_display" not in st.session_state:
    st.session_state["last_live_sync_display"] = ""

with st.sidebar:
    active_view = st.radio("View", options=["Board", "History", "Player", "Method"], index=0)
    league_choice = st.radio(
        "League",
        options=LEAGUE_OPTIONS,
        index=LEAGUE_OPTIONS.index(saved_league),
        horizontal=True,
    )
    selected_league = normalize_sport(league_choice)
    st.markdown("---")
    st.markdown("## Feed")
    refresh_live = st.button("Run live scan now", width="stretch", type="primary")
    if st.session_state.get("last_live_sync_display"):
        st.caption(f"Last successful sync: {st.session_state['last_live_sync_display']}")
    with st.form("feed_settings_form"):
        api_key_input = st.text_input(
            "API key",
            value=saved_api_key,
            type="password",
            help="Saved locally so the app can auto-load live props on future runs.",
        )
        bookmaker_input = st.multiselect(
            "Sportsbooks",
            options=list(BOOKMAKER_OPTIONS.keys()),
            default=saved_bookmakers,
            format_func=lambda key: BOOKMAKER_OPTIONS.get(key, key),
        )
        fetch_prop_input = st.multiselect(
            "Fetched prop markets",
            options=list(PROP_MARKET_MAP.keys()),
            default=saved_fetch_prop_types,
            format_func=format_prop_type,
        )
        days_ahead_input = st.slider("Upcoming window (days)", min_value=1, max_value=3, value=saved_days_ahead)
        auto_refresh_input = st.toggle("Auto refresh live board", value=auto_refresh_enabled)
        save_settings_clicked = st.form_submit_button("Save settings", width="stretch")
    st.caption("Saved locally. Use Run for an immediate refresh.")
    st.markdown("---")
    st.markdown("## Board Filters")
    venue_filter = st.selectbox("Venue", options=["All", "Home", "Away"])
    min_sample = st.slider("History sample min", min_value=5, max_value=50, value=APP_CONFIG.min_segment_samples, step=5)
    player_search = st.text_input("Player search", placeholder="Type part of a player name")

if save_settings_clicked:
    save_user_settings(
        {
            "api_key": api_key_input.strip(),
            "bookmakers": bookmaker_input,
            "prop_types": fetch_prop_input or list(PROP_MARKET_MAP.keys()),
            "days_ahead": days_ahead_input,
            "league": selected_league,
            "auto_refresh": auto_refresh_input,
            "auto_refresh_minutes": auto_refresh_minutes,
        }
    )
    st.rerun()

api_key = saved_api_key
bookmaker_keys = saved_bookmakers
fetch_prop_types = saved_fetch_prop_types

cached_history_props, cached_game_results = load_cached_history()
has_saved_history = cached_history_props is not None and cached_game_results is not None
history_source_label = "Auto history cache" if has_saved_history else "Auto history warming up"
history_source_note = (
    "Historical training data is loading from the local cache and will keep updating automatically."
    if has_saved_history
    else "The local history cache is still warming up. Captured live lines will be added automatically after those games finish."
)
history_is_real = has_saved_history
history_pipeline = build_history_pipeline(cached_history_props, cached_game_results) if has_saved_history else build_empty_history_pipeline()
history_features = filter_history_for_league(history_pipeline["historical_features"], selected_league)

live_raw = pd.DataFrame()
live_scored = pd.DataFrame()
live_board = pd.DataFrame()
live_error = ""
live_status_note = "Connect a saved API key to turn on the automatic live feed."
history_sync_note = ""

active_prop_filters = list(PROP_MARKET_MAP.keys())
should_load_live = bool(
    api_key
    and fetch_prop_types
    and (refresh_live or (auto_refresh_enabled and active_view in {"Board", "Player"}))
)
if should_load_live:
    if refresh_live:
        fetch_live_props_cached.clear()
        score_live_current_props_cached.clear()
    try:
        live_raw = fetch_live_props_cached(api_key, tuple(bookmaker_keys), saved_days_ahead, tuple(fetch_prop_types), selected_league)
        st.session_state["last_live_sync_display"] = pd.Timestamp.now(tz="America/Chicago").strftime("%b %d, %Y %I:%M %p %Z")
        sync_signature = (
            selected_league,
            tuple(sorted(fetch_prop_types)),
            tuple(sorted(bookmaker_keys)),
            saved_days_ahead,
            str(live_raw["market_last_update"].max()) if "market_last_update" in live_raw.columns and not live_raw.empty else "empty",
            len(live_raw),
        )
        needs_final_score = True
        if sync_signature != st.session_state["last_history_sync_signature"]:
            baseline_history = history_features if history_is_real else pd.DataFrame()
            live_scored = score_live_current_props_cached(
                live_raw,
                current_basketball_season(selected_league),
                baseline_history if not baseline_history.empty else None,
                selected_league,
            )
            needs_final_score = False
            combined_props, combined_results, sync_info = sync_auto_history_from_live(
                live_scored,
                season=current_basketball_season(selected_league),
                sport=selected_league,
            )
            st.session_state["last_history_sync_signature"] = sync_signature
            if combined_props is not None and combined_results is not None and not combined_props.empty:
                history_pipeline = build_history_pipeline(combined_props, combined_results)
                history_features = filter_history_for_league(history_pipeline["historical_features"], selected_league)
                history_is_real = True
                history_source_label = "Auto history cache"
                history_source_note = f"Historical training cache is updating automatically from captured lines and finished {selected_league} results."
                if sync_info.get("resolved_rows", 0) > 0:
                    history_sync_note = f"Auto-added {sync_info['resolved_rows']} finished props into the training cache."
                needs_final_score = True

        if needs_final_score:
            live_scored = score_live_current_props_cached(
                live_raw,
                current_basketball_season(selected_league),
                history_features if history_is_real and not history_features.empty else None,
                selected_league,
            )
        live_scored = ensure_live_display_fields(live_scored)
        live_board = build_live_board(live_scored)
        live_status_note = (
            f"Auto-updated live feed across {len(bookmaker_keys)} books. "
            f"Last sync: {st.session_state['last_live_sync_display']}. "
            f"Auto refresh runs on app load with a {auto_refresh_minutes}-minute cache window."
        )
    except Exception as exc:
        live_error = str(exc)
        live_status_note = "Live feed is connected, but the last automatic refresh did not complete."
elif api_key and not auto_refresh_enabled:
    live_status_note = "Live feed is saved locally, but auto refresh is currently off."
elif api_key:
    live_status_note = "Live feed is connected. Open the Board view or press Run live scan now to refresh props."

team_options = sorted(
    set(history_features["team"].dropna().unique().tolist()).union(live_board["team"].dropna().unique().tolist() if "team" in live_board.columns else [])
)

with st.sidebar:
    prop_type_filter = st.multiselect(
        "Displayed prop markets",
        options=list(PROP_MARKET_MAP.keys()),
        default=fetch_prop_types or list(PROP_MARKET_MAP.keys()),
        format_func=format_prop_type,
    )
    team_filter = st.multiselect("Teams", options=team_options)
    if history_source_note:
        st.caption(history_source_note)
    if history_sync_note:
        st.caption(history_sync_note)

active_prop_filters = prop_type_filter or fetch_prop_types or list(PROP_MARKET_MAP.keys())

history_filtered = apply_team_venue_filters(history_features, team_filter, venue_filter, active_prop_filters)
live_board_filtered = apply_team_venue_filters(live_board, team_filter, venue_filter, active_prop_filters)
if player_search and "player_name" in live_board_filtered.columns:
    live_board_filtered = live_board_filtered.loc[
        live_board_filtered["player_name"].str.contains(player_search, case=False, na=False)
    ].reset_index(drop=True)

live_book_count = int(live_scored["bookmaker"].nunique()) if not live_scored.empty else len(bookmaker_keys)
live_prop_count = len(live_scored)
live_players_count = int(live_board["player_name"].nunique()) if not live_board.empty else 0
history_total = len(history_features)

today_local = pd.Timestamp.now(tz="America/Chicago")
today_games = pd.DataFrame()
today_games_error = ""
try:
    today_games = fetch_daily_games_cached(today_local.date().isoformat(), selected_league)
except Exception as exc:
    today_games_error = str(exc)

status_parts = [
    selected_league,
    f"{history_source_label}",
    f"{live_book_count} books",
    f"{live_prop_count} live lines",
    f"{live_players_count} player-prop spots",
    f"{history_total} training rows",
    f"{len(today_games)} games today",
]
if st.session_state.get("last_live_sync_display"):
    status_parts.append(f"Last sync {st.session_state['last_live_sync_display']}")
render_app_header(status_parts, selected_league)
render_games_strip(today_games, today_local.strftime("%A, %b %d"), selected_league, today_games_error)

if active_view == "Board":
    if not api_key:
        render_notice(
            "Live scan is off",
            "Add a The Odds API key in the sidebar to pull live sportsbook props directly into the app. If you set THE_ODDS_API_KEY in your environment, the board will populate automatically on load.",
            tone="gold",
        )
    elif live_error:
        render_notice(
            "Live scan failed",
            live_error,
            tone="red",
        )
    elif live_board_filtered.empty:
        render_notice(
            "No live props matched the current filters",
            "Try widening the sportsbook list, switching venue back to All, or clearing the team and player filters.",
            tone="gold",
        )
    else:
        if not history_is_real:
            render_notice(
                "Historical cache is still warming up",
                f"The live board is active. Historical validation will get stronger as captured lines resolve into finished {selected_league} results.",
                tone="gold",
            )

        top_board = live_board_filtered.head(12).copy()
        render_board_header(len(top_board))
        left_col, right_col = st.columns([1.35, 0.95], gap="large")

        with left_col:
            st.dataframe(
                style_live_table(
                    top_board[
                        [
                            "rank",
                            "player_name",
                            "team",
                            "lean",
                            "line",
                            "prop_label",
                            "rolling_avg_10",
                            "consensus_line",
                            "matchup",
                            "bookmaker",
                            "scanner_score",
                            "support_label",
                        ]
                    ]
                ),
                hide_index=True,
                width="stretch",
                height=520,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                    "player_name": st.column_config.TextColumn("Player", width="medium"),
                    "team": st.column_config.TextColumn("Team", width="small"),
                    "lean": st.column_config.TextColumn("Lean", width="small"),
                    "line": st.column_config.NumberColumn("Line", format="%.1f", width="small"),
                    "prop_label": st.column_config.TextColumn("Prop", width="small"),
                    "rolling_avg_10": st.column_config.NumberColumn("10G avg", format="%.1f", width="small"),
                    "consensus_line": st.column_config.NumberColumn("Consensus", format="%.1f", width="small"),
                    "matchup": st.column_config.TextColumn("Matchup", width="small"),
                    "bookmaker": st.column_config.TextColumn("Best book", width="small"),
                    "scanner_score": st.column_config.NumberColumn("Score", format="%.1f", width="small"),
                    "support_label": st.column_config.TextColumn("Support", width="medium"),
                },
            )

        selected_options = top_board["display_label"].tolist() if not top_board.empty else live_board_filtered["display_label"].tolist()
        with right_col:
            selected_label = st.selectbox("Inspect line", options=selected_options)
        selected_prop = live_board_filtered.loc[live_board_filtered["display_label"] == selected_label].iloc[0]

        with right_col:
            render_selected_prop_card(selected_prop)
            render_badges(live_warning_badges(selected_prop))
            st.plotly_chart(build_component_score_chart(selected_prop), width="stretch")

        selected_shop = live_scored.loc[
            (live_scored["event_id"] == selected_prop["event_id"])
            & (live_scored["player_name"] == selected_prop["player_name"])
            & (live_scored["prop_type"] == selected_prop["prop_type"])
        ].sort_values(["scanner_score", "shopping_edge_points"], ascending=[False, False])

        with st.expander("Book comparison for selected prop", expanded=False):
            st.dataframe(
                style_live_table(
                    selected_shop[
                        [
                            "bookmaker",
                            "lean",
                            "line",
                            "prop_label",
                            "rolling_avg_10",
                            "over_odds",
                            "under_odds",
                            "consensus_line",
                            "shopping_edge_points",
                            "scanner_score",
                        ]
                    ]
                ),
                hide_index=True,
                width="stretch",
                height=260,
                column_config={
                    "bookmaker": st.column_config.TextColumn("Book", width="medium"),
                    "lean": st.column_config.TextColumn("Lean", width="small"),
                    "line": st.column_config.NumberColumn("Line", format="%.1f", width="small"),
                    "prop_label": st.column_config.TextColumn("Prop", width="small"),
                    "rolling_avg_10": st.column_config.NumberColumn("10G avg", format="%.1f", width="small"),
                    "over_odds": st.column_config.NumberColumn("Over odds", format="%d", width="small"),
                    "under_odds": st.column_config.NumberColumn("Under odds", format="%d", width="small"),
                    "consensus_line": st.column_config.NumberColumn("Consensus", format="%.1f", width="small"),
                    "shopping_edge_points": st.column_config.NumberColumn("Line edge", format="%.1f", width="small"),
                    "scanner_score": st.column_config.NumberColumn("Scanner score", format="%.1f", width="small"),
                },
            )

        with st.expander("See all scanned lines"):
            st.dataframe(
                style_live_table(
                    live_scored[
                        [
                            "player_name",
                            "team",
                            "bookmaker",
                            "lean",
                            "line",
                            "prop_label",
                            "opponent",
                            "overall_prop_analysis_score",
                            "scanner_score",
                            "support_label",
                            "explanation_text",
                        ]
                    ].sort_values(["scanner_score", "player_name"], ascending=[False, True])
                ),
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "player_name": st.column_config.TextColumn("Player", width="medium"),
                    "team": st.column_config.TextColumn("Team", width="small"),
                    "bookmaker": st.column_config.TextColumn("Book", width="small"),
                    "lean": st.column_config.TextColumn("Lean", width="small"),
                    "line": st.column_config.NumberColumn("Line", format="%.1f"),
                    "prop_label": st.column_config.TextColumn("Prop", width="small"),
                    "opponent": st.column_config.TextColumn("Opp", width="small"),
                    "overall_prop_analysis_score": st.column_config.NumberColumn("Base score", format="%.1f"),
                    "scanner_score": st.column_config.NumberColumn("Scanner score", format="%.1f"),
                    "explanation_text": st.column_config.TextColumn("Explanation", width="large"),
                },
            )

if active_view == "History":
    render_section_header(
        "Historical market view",
        "This page checks whether the selected prop markets have been running high, low, or fairly efficient in the historical sample. Focus on line error, hit-rate splits, and whether any segment actually holds later.",
    )

    if history_filtered.empty:
        render_notice(
            "No historical rows matched the current filters",
            "Clear the team filter or switch the venue filter back to All to bring the historical sample back.",
            tone="red",
        )
    else:
        history_prop_options = sorted(history_filtered["prop_type"].dropna().unique().tolist())
        if len(history_prop_options) > 1:
            history_prop_focus = st.selectbox(
                "Historical prop focus",
                options=["All selected"] + history_prop_options,
                format_func=lambda value: value if value == "All selected" else format_prop_type(value),
            )
        else:
            history_prop_focus = history_prop_options[0]
        history_view = history_filtered if history_prop_focus == "All selected" else history_filtered.loc[history_filtered["prop_type"] == history_prop_focus].reset_index(drop=True)

        metrics = overall_market_metrics(history_view)
        render_metric_row(
            [
                {
                    "label": "Props tracked",
                    "value": f"{int(metrics['total_props'])}",
                    "note": f"Historical {selected_league} player props in the active filter set.",
                    "tone": "cyan",
                },
                {
                    "label": "Over hit rate",
                    "value": format_pct(metrics["over_hit_rate"]),
                    "note": "Observed over rate after excluding pushes.",
                    "tone": "lime",
                },
                {
                    "label": "Mean line error",
                    "value": f"{metrics['mean_error']:+.2f}",
                    "note": "Positive means players beat the line on average.",
                    "tone": "pink",
                },
                {
                    "label": "Mean abs error",
                    "value": f"{metrics['mae']:.2f}",
                    "note": "Average absolute miss between line and actual outcome.",
                    "tone": "gold",
                },
            ]
        )

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.plotly_chart(build_line_vs_actual_scatter(history_view), width="stretch")
        with chart_right:
            st.plotly_chart(build_residual_distribution(history_view), width="stretch")

        st.plotly_chart(
            build_rolling_edge_chart(rolling_edge_summary(history_view)),
            width="stretch",
        )

        segment_label = st.selectbox("Segment explorer", options=list(SEGMENT_OPTIONS.keys()))
        segment_key = SEGMENT_OPTIONS[segment_label]["segment_key"]
        segment_table_key = SEGMENT_OPTIONS[segment_label]["table_key"]
        segment_tables = collect_segment_tables(history_view, min_sample=min_sample)
        segment_raw = segment_tables[segment_table_key]
        segment_display = prepare_segment_table(segment_raw, segment_key)
        segment_chart_source, segment_chart_key = prepare_segment_chart_source(segment_raw, segment_key)

        segment_left, segment_right = st.columns([0.9, 1.1], gap="large")
        with segment_left:
            st.plotly_chart(
                build_segment_bar_chart(segment_chart_source, segment_chart_key),
                width="stretch",
            )
        with segment_right:
            st.dataframe(
                segment_display,
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "sample_size": st.column_config.NumberColumn("Sample", format="%d"),
                    "over_hit_rate_pct": st.column_config.NumberColumn("Over %", format="%.1f"),
                    "under_hit_rate_pct": st.column_config.NumberColumn("Under %", format="%.1f"),
                    "mean_error": st.column_config.NumberColumn("Mean error", format="%.2f"),
                    "potential_inefficiency_flag": st.column_config.TextColumn("Signal", width="medium"),
                    "warning": st.column_config.TextColumn("Warning", width="large"),
                },
            )

        render_section_header(
            "Stability check",
            "A hot-looking segment is only interesting if it still leans the same way later in the sample. This table compares the earlier and later portions of the dataset.",
        )
        stability_raw = temporal_stability_by_segment(history_view, segment_key, min_sample=min_sample)
        stability_display = prepare_stability_table(stability_raw, segment_key)
        st.dataframe(
            stability_display,
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "sample_size_train": st.column_config.NumberColumn("Train sample", format="%d"),
                "sample_size_test": st.column_config.NumberColumn("Later sample", format="%d"),
                "over_hit_rate_train_pct": st.column_config.NumberColumn("Train over %", format="%.1f"),
                "over_hit_rate_test_pct": st.column_config.NumberColumn("Later over %", format="%.1f"),
                "mean_error_train": st.column_config.NumberColumn("Train error", format="%.2f"),
                "mean_error_test": st.column_config.NumberColumn("Later error", format="%.2f"),
                "stability_assessment": st.column_config.TextColumn("Assessment", width="medium"),
                "warning": st.column_config.TextColumn("Warning", width="medium"),
            },
        )

if active_view == "Player":
    render_section_header(
        "Player view",
        "Use this page to understand one player and one prop market at a time: where the line has been set, how often it has been beaten, and how the live board is pricing that market across books.",
    )

    history_player_pool = history_features["player_name"].dropna().unique().tolist()
    live_player_pool = live_scored["player_name"].dropna().unique().tolist() if "player_name" in live_scored.columns else []
    player_pool = sorted(set(history_player_pool).union(live_player_pool))
    if player_search:
        matches = [player for player in player_pool if player_search.lower() in player.lower()]
        player_options = matches if matches else player_pool
    else:
        player_options = player_pool

    selected_player = st.selectbox("Player", options=player_options)
    history_player_props = history_features.loc[history_features["player_name"] == selected_player, "prop_type"].dropna().unique().tolist()
    live_player_props = live_scored.loc[live_scored["player_name"] == selected_player, "prop_type"].dropna().unique().tolist() if "player_name" in live_scored.columns else []
    player_prop_options = sorted(set(history_player_props).union(live_player_props))
    selected_player_prop = st.selectbox("Prop market", options=player_prop_options, format_func=format_prop_type)
    player_history = history_features.loc[
        (history_features["player_name"] == selected_player) & (history_features["prop_type"] == selected_player_prop)
    ].sort_values("game_date")

    if player_history.empty:
        render_notice(
            "No historical sample for this player market",
            "The selected saved history does not include this player and prop market combination yet. Live pricing can still be inspected below if a current line exists.",
            tone="gold",
        )

    render_metric_row(player_summary_cards(player_history))
    st.plotly_chart(build_player_history_chart(player_history), width="stretch")

    split_left, split_right = st.columns(2, gap="large")
    with split_left:
        st.dataframe(
            player_split_table(player_history, "is_home"),
            hide_index=True,
            width="stretch",
            height=240,
            column_config={
                "games": st.column_config.NumberColumn("Games", format="%d"),
                "avg_line": st.column_config.NumberColumn("Avg line", format="%.1f"),
                "avg_actual": st.column_config.NumberColumn("Avg actual", format="%.1f"),
                "mean_error": st.column_config.NumberColumn("Mean error", format="%.2f"),
                "over_hit_rate_pct": st.column_config.NumberColumn("Over %", format="%.1f"),
            },
        )
    with split_right:
        st.dataframe(
            player_split_table(player_history, "line_range_bucket"),
            hide_index=True,
            width="stretch",
            height=240,
            column_config={
                "games": st.column_config.NumberColumn("Games", format="%d"),
                "avg_line": st.column_config.NumberColumn("Avg line", format="%.1f"),
                "avg_actual": st.column_config.NumberColumn("Avg actual", format="%.1f"),
                "mean_error": st.column_config.NumberColumn("Mean error", format="%.2f"),
                "over_hit_rate_pct": st.column_config.NumberColumn("Over %", format="%.1f"),
            },
        )

    if "player_name" in live_scored.columns:
        player_live = ensure_live_display_fields(
            live_scored.loc[live_scored["player_name"] == selected_player]
        ).sort_values(
            ["scanner_score", "shopping_edge_points"],
            ascending=[False, False],
        )
    else:
        player_live = pd.DataFrame()
    if not player_live.empty:
        player_live = player_live.loc[player_live["prop_type"] == selected_player_prop].sort_values(
            ["scanner_score", "shopping_edge_points"],
            ascending=[False, False],
        )
    if not player_live.empty:
        best_live = player_live.iloc[0]
        render_notice(
            "Current live read",
            f"{selected_player} currently grades best at {best_live['bookmaker']} in {format_prop_type(best_live['prop_type'])} with a {best_live['lean']} lean on {float(best_live['line']):.1f}. Scanner score: {float(best_live['scanner_score']):.1f}.",
            tone="cyan",
        )
        render_badges(live_warning_badges(best_live))
        st.dataframe(
            style_live_table(
                player_live[
                    [
                        "bookmaker",
                        "lean",
                        "line",
                        "prop_label",
                        "over_odds",
                        "under_odds",
                        "consensus_line",
                        "shopping_edge_points",
                        "scanner_score",
                        "explanation_text",
                    ]
                ]
            ),
            hide_index=True,
            width="stretch",
            height=300,
            column_config={
                "lean": st.column_config.TextColumn("Lean", width="small"),
                "line": st.column_config.NumberColumn("Line", format="%.1f"),
                "prop_label": st.column_config.TextColumn("Prop", width="small"),
                "over_odds": st.column_config.NumberColumn("Over odds", format="%d"),
                "under_odds": st.column_config.NumberColumn("Under odds", format="%d"),
                "consensus_line": st.column_config.NumberColumn("Consensus", format="%.1f"),
                "shopping_edge_points": st.column_config.NumberColumn("Shop edge", format="%.1f"),
                "scanner_score": st.column_config.NumberColumn("Scanner score", format="%.1f"),
                "explanation_text": st.column_config.TextColumn("Explanation", width="large"),
            },
        )
    elif api_key and not live_error:
        render_notice(
            "No live line found for this player",
            "That usually means the sportsbooks in the current filter set do not have a listed prop for this player and market in the selected upcoming window.",
            tone="gold",
        )

if active_view == "Method":
    render_section_header(
        "How this works",
        "The app is designed to be easier to operate than to explain. This page keeps the logic plain: where the data comes from, what the score is, and where the limits still are.",
    )

    st.markdown(
        """
        <div class="method-grid">
            <div class="method-card">
                <div class="method-step">Step 1</div>
                <div class="method-label">Pull live books</div>
                <div class="method-copy">The scanner calls a live sportsbook API for NBA and WNBA player prop markets, including points, rebounds, assists, and common combos, then builds a book-by-book board for the next few days.</div>
            </div>
            <div class="method-card">
                <div class="method-step">Step 2</div>
                <div class="method-label">Score the setup</div>
                <div class="method-copy">Each prop gets a support score from recent player form, line value versus current averages, consistency, matchup context, and optional historical market reference tables if real history is loaded.</div>
            </div>
            <div class="method-card">
                <div class="method-step">Step 3</div>
                <div class="method-label">Validate the market</div>
                <div class="method-copy">The History and Player pages show whether certain lines or contexts have actually looked beatable before, and whether those trends stayed stable later in the sample.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_row(
        [
            {
                "label": "Current prop checks",
                "value": "API",
                "note": "Once your feed settings are saved locally, the live scanner can load automatically on future runs.",
                "tone": "lime",
            },
            {
                "label": "Historical layer",
                "value": "Auto cache",
                "note": "Captured live lines are archived locally and matched to finished NBA or WNBA results to grow the training set over time.",
                "tone": "cyan",
            },
            {
                "label": "Score meaning",
                "value": "Support",
                "note": "This is a structured ranking score, not a model win probability.",
                "tone": "gold",
            },
            {
                "label": "Scope",
                "value": "NBA + WNBA",
                "note": "The live board supports basketball prop markets while keeping each league filtered separately.",
                "tone": "pink",
            },
        ]
    )

    data_source_frame = pd.DataFrame(
        [
            {
                "layer": "Live current props",
                "status": "Automatic",
                "details": "Pulled from The Odds API across selected sportsbooks for points, rebounds, assists, and combo markets.",
            },
            {
                "layer": "Historical market validation",
                "status": "Auto-updating",
                "details": "The app stores captured live lines locally, then resolves them into historical training rows after those games finish. Optional manual imports can still supplement that cache.",
            },
            {
                "layer": "Current player form",
                "status": "Automatic",
                "details": "Pulled from nba_api season player logs to build rolling player context for every supported prop market.",
            },
        ]
    )
    st.dataframe(
        data_source_frame,
        hide_index=True,
        width="stretch",
        height=180,
        column_config={"details": st.column_config.TextColumn("Details", width="large")},
    )

    render_notice(
        "Important limitation",
        "The historical model now updates automatically going forward, but it still cannot magically backfill a complete free historical prop archive for every sportsbook from seasons you never captured. For a deeper back-history, you would still need a paid historical feed or your own imported archive.",
        tone="gold",
    )

    st.markdown(
        """
        **Version 1 limitations**

        - NBA and WNBA player props are supported, focused on points, rebounds, assists, and common combo markets.
        - The scanner ranks setups; it does not output true bet probabilities or expected value after vig.
        - Historical conclusions are only as good as the local prop-line history the app has captured or imported.
        - Injury news, teammate absences, and pace/game-total context are not fully modeled yet.
        - Historical stability checks are simple train-later split checks, not a full walk-forward production validation stack.
        """
    )

    with st.expander("Optional manual import schema"):
        st.dataframe(
            build_schema_frame(),
            hide_index=True,
            width="stretch",
            height=280,
        )
