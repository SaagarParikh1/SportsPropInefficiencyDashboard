from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORWAY = ["#39E7FF", "#8DFF45", "#FF5DC8", "#FFC857", "#7E8BFF"]
CARD_BG = "rgba(8, 12, 28, 0.0)"
GRID_COLOR = "rgba(142, 162, 255, 0.14)"
TEXT_COLOR = "#F5F7FF"
NEUTRAL_LINE = "rgba(245, 247, 255, 0.28)"
RESIDUAL_SCALE = ["#FF5DC8", "#1B2342", "#8DFF45"]


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=TEXT_COLOR, size=14),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return figure


def _style_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        colorway=COLORWAY,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Space Grotesk, Avenir Next, Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        title_font=dict(color=TEXT_COLOR, size=20),
    )
    figure.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    figure.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    return figure


def build_line_vs_actual_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No historical observations available for this view.")

    hover_fields = ["game_date", "player_name", "team", "opponent"]
    if "prop_type" in df.columns:
        hover_fields.append("prop_type")
    figure = px.scatter(
        df,
        x="line",
        y="actual_value",
        color="actual_minus_line",
        color_continuous_scale=RESIDUAL_SCALE,
        hover_data=hover_fields,
        labels={"line": "Closing line", "actual_value": "Actual outcome", "actual_minus_line": "Residual"},
    )
    figure.update_traces(marker=dict(size=9, line=dict(width=0)), opacity=0.8)
    min_bound = min(df["line"].min(), df["actual_value"].min()) - 1
    max_bound = max(df["line"].max(), df["actual_value"].max()) + 1
    figure.add_shape(
        type="line",
        x0=min_bound,
        y0=min_bound,
        x1=max_bound,
        y1=max_bound,
        line=dict(color=NEUTRAL_LINE, dash="dash"),
    )
    figure.update_layout(title="Line vs Actual")
    return _style_figure(figure)


def build_residual_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No residual data available.")

    figure = px.histogram(
        df,
        x="actual_minus_line",
        nbins=24,
        labels={"actual_minus_line": "Actual minus closing line"},
        color_discrete_sequence=["#39E7FF"],
    )
    figure.update_traces(marker_line_width=0, opacity=0.92)
    figure.add_vline(x=0, line_dash="dash", line_color=NEUTRAL_LINE)
    figure.update_layout(title="Residual Distribution")
    return _style_figure(figure)


def build_rolling_edge_chart(rolling_df: pd.DataFrame) -> go.Figure:
    if rolling_df.empty:
        return _empty_figure("Not enough data to build rolling edge trends.")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=rolling_df["game_date"],
            y=rolling_df["rolling_over_hit_rate"],
            name="Rolling over hit rate",
            mode="lines",
            line=dict(color="#8DFF45", width=3),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=rolling_df["game_date"],
            y=rolling_df["rolling_under_hit_rate"],
            name="Rolling under hit rate",
            mode="lines",
            line=dict(color="#FF5DC8", width=2.6),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=rolling_df["game_date"],
            y=rolling_df["rolling_mean_error"],
            name="Rolling mean error",
            mode="lines",
            line=dict(color="#39E7FF", width=2.2, dash="dot"),
        ),
        secondary_y=True,
    )
    figure.add_hline(y=0.5, line_dash="dash", line_color=NEUTRAL_LINE, secondary_y=False)
    figure.update_layout(title="Rolling Market Bias")
    figure.update_yaxes(title_text="Hit rate", secondary_y=False)
    figure.update_yaxes(title_text="Mean error", secondary_y=True)
    return _style_figure(figure)


def build_segment_bar_chart(segment_df: pd.DataFrame, segment_col: str, metric_col: str = "mean_error") -> go.Figure:
    if segment_df.empty or segment_col not in segment_df.columns:
        return _empty_figure("No segment results available.")

    top_segments = segment_df.copy().sort_values(metric_col, key=lambda series: series.abs(), ascending=False).head(12)
    figure = px.bar(
        top_segments.sort_values(metric_col),
        x=metric_col,
        y=segment_col,
        orientation="h",
        color=metric_col,
        color_continuous_scale=RESIDUAL_SCALE,
        labels={metric_col: "Mean residual", segment_col: "Segment"},
    )
    figure.add_vline(x=0, line_dash="dash", line_color=NEUTRAL_LINE)
    figure.update_layout(title=f"Top {segment_col.replace('_', ' ').title()} Bias Spots")
    return _style_figure(figure)


def build_player_history_chart(player_df: pd.DataFrame) -> go.Figure:
    if player_df.empty:
        return _empty_figure("No player history available.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=player_df["game_date"],
            y=player_df["actual_value"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#39E7FF", width=2.7),
            marker=dict(size=7),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=player_df["game_date"],
            y=player_df["line"],
            mode="lines",
            name="Closing line",
            line=dict(color="#FF5DC8", width=2.2),
        )
    )
    if "rolling_avg_10" in player_df.columns:
        figure.add_trace(
            go.Scatter(
                x=player_df["game_date"],
                y=player_df["rolling_avg_10"],
                mode="lines",
                name="Rolling avg (10)",
                line=dict(color="#8DFF45", width=2.0, dash="dot"),
            )
        )
    figure.update_layout(title="Player Prop History")
    return _style_figure(figure)


def build_score_ranking_chart(
    df: pd.DataFrame,
    score_column: str = "overall_prop_analysis_score",
    title: str = "Top Current Prop Signals",
) -> go.Figure:
    if df.empty:
        return _empty_figure("No current props available.")

    if score_column not in df.columns:
        return _empty_figure("Ranking scores are not available.")

    ranking = df.head(12).copy().sort_values(score_column)
    y_column = "player_display" if "player_display" in ranking.columns else "player_name"
    figure = px.bar(
        ranking,
        x=score_column,
        y=y_column,
        orientation="h",
        color="lean",
        color_discrete_map={"Over": "#8DFF45", "Under": "#FF5DC8", "Neutral": "#FFC857"},
        hover_data=["support_label", "line", "opponent"],
    )
    figure.update_layout(title=title, showlegend=True)
    return _style_figure(figure)


def build_component_score_chart(prop_row: pd.Series) -> go.Figure:
    score_columns = [
        "historical_context_score",
        "recent_form_score",
        "line_value_score",
        "consistency_score_component",
        "matchup_score",
        "stability_score",
    ]
    if any(column not in prop_row.index for column in score_columns):
        return _empty_figure("Component scores are not available.")

    component_frame = pd.DataFrame(
        {
            "component": [
                "Historical context",
                "Recent form",
                "Line value",
                "Consistency",
                "Matchup",
                "Stability",
            ],
            "score": [prop_row[column] for column in score_columns],
        }
    ).sort_values("score")
    figure = px.bar(
        component_frame,
        x="score",
        y="component",
        orientation="h",
        color="score",
        color_continuous_scale=["#1B2342", "#39E7FF", "#8DFF45"],
    )
    figure.update_layout(title="Why This Prop Scores Here")
    return _style_figure(figure)
