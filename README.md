# Sports Prop Market Inefficiency Dashboard

This project analyzes whether sports prop markets show persistent inefficiencies and translates those insights into a decision-support dashboard for structured prop evaluation.

The MVP is intentionally narrow:

- Sport: `NBA`
- Prop type: `Player points`
- Market type: `Over/under`
- Delivery: `Local-first Streamlit dashboard`

This is not a gambling bot, an auto-betting system, or a guaranteed prediction engine. The focus is historical market evaluation, contextual feature engineering, and careful evidence-based scoring.

## Product Plan

### Phase 1: Architecture

- Modular `src/` package for ingestion, cleaning, feature engineering, evaluation, scoring, visualization, and optional modeling
- Streamlit app shell with production-style information architecture
- Local-first storage using CSV/Parquet inputs with clean upgrade paths for adapters later

### Phase 2: MVP Scope

- Start with NBA player points props only
- Evaluate whether closing lines have historically leaned too high or too low
- Support current prop review only after historical validation features exist

### Phase 3: Historical Evaluation Engine

- Over hit rate and under hit rate
- Closing line accuracy and residual analysis
- Splits by player, team, opponent, home/away, rest, rolling form, and line range
- Rolling edge tracking to test whether patterns look stable or noisy

### Phase 4: Feature Engineering

- Rest and back-to-back context
- Rolling averages, medians, variance, hit rates, and trend direction
- Line relative to recent and season form
- Opponent difficulty proxies based on prior historical allowance

### Phase 5: Evidence-Based Beatability Framework

- Confidence intervals
- Sample-size filters
- Segment analysis
- Train/test style temporal stability checks
- Language that stays cautious: "historically favorable", "potential inefficiency", "requires caution"

### Phase 6: Current Prop Analysis

- Enrich a current slate with player snapshots and historical segment performance
- Score setups from `0-100` as a decision-support signal
- Generate plain-language explanations instead of hard prediction claims

## Repository Structure

```text
sports-prop-dashboard/
  app.py
  requirements.txt
  README.md
  data/
    raw/
    processed/
    cache/
  src/
    config.py
    data_ingestion.py
    data_cleaning.py
    feature_engineering.py
    line_evaluation.py
    prop_analysis.py
    modeling.py
    scoring.py
    visualization.py
    utils.py
  notebooks/
  outputs/
    charts/
    summaries/
```

## Core Data Schema

### Historical Prop Lines

| column | type | purpose |
| --- | --- | --- |
| `game_id` | string | unique game-player key |
| `game_date` | date | game date |
| `season` | string | season label |
| `sport` | string | sport name, default `NBA` |
| `player_name` | string | player identity |
| `team` | string | player team |
| `opponent` | string | opposing team |
| `is_home` | bool | home/away flag |
| `prop_type` | string | prop market, default `points` |
| `market_type` | string | market type, default `over_under` |
| `opening_line` | float | opening line if available |
| `closing_line` | float | closing line used for evaluation |
| `over_odds` | float | optional over price |
| `under_odds` | float | optional under price |
| `bookmaker` | string | source/book label |

### Historical Game Results

| column | type | purpose |
| --- | --- | --- |
| `game_id` | string | merge key |
| `game_date` | date | game date |
| `season` | string | season label |
| `sport` | string | sport name |
| `player_name` | string | player identity |
| `team` | string | player team |
| `opponent` | string | opposing team |
| `is_home` | bool | home/away flag |
| `prop_type` | string | stat family represented by `actual_value` |
| `actual_value` | float | realized stat outcome |
| `minutes_played` | float | optional context signal |
| `usage_rate` | float | optional context signal |

### Current Props

| column | type | purpose |
| --- | --- | --- |
| `game_date` | date | slate date |
| `sport` | string | sport name |
| `player_name` | string | player identity |
| `team` | string | player team |
| `opponent` | string | opposing team |
| `is_home` | bool | home/away flag |
| `prop_type` | string | prop market |
| `market_type` | string | over/under |
| `line` | float | current line to analyze |
| `over_odds` | float | optional over price |
| `under_odds` | float | optional under price |
| `bookmaker` | string | source/book label |

Template CSV headers are included in [data/raw](/Users/saagar/Desktop/Data Analytics Projects/Sports Prop Inefficiency Dashboard/data/raw/README.md).

## MVP Functionality

Implemented in this first pass:

- Project scaffolding and modular architecture
- CSV/Parquet ingestion with schema validation
- Cleaning and standardization of names, dates, booleans, and numeric fields
- Historical prop/result merge pipeline
- Leak-aware feature engineering from prior games only
- Historical market evaluation with:
  - hit rates
  - confidence intervals
  - residual analysis
  - rolling edge analysis
  - context splits
  - temporal stability checks
- Current prop analysis layer with:
  - player snapshots
  - historical reference tables
  - evidence-based scoring
  - explanation text
- Streamlit dashboard with:
  - overview
  - historical evaluation
  - player explorer
  - current props analyzer
  - methodology
- Bundled deterministic demo data so the app runs immediately

## Scoring Interpretation

The `Overall Prop Analysis Score` is not a win probability. It is a structured evidence score based on historical context, recent form, line value, consistency, matchup difficulty, and stability/sample quality.

Default score bands:

- `80-100`: Strong historical support
- `65-79`: Moderate support
- `50-64`: Neutral
- `35-49`: Caution
- `0-34`: Historically unfavorable

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Sections

1. `Overview`
   - Scope, KPIs, current scoring snapshot, uncertainty reminders
2. `Historical Market Evaluation`
   - Line accuracy, residuals, rolling performance, and segment tables
3. `Player Prop Explorer`
   - Individual player history, line versus actual trends, and contextual splits
4. `Current Props Analyzer`
   - Ranked slate scoring, lean labels, explanation text, and download option
5. `Methodology`
   - Data assumptions, validation approach, limitations, and extension roadmap

## Positioning

This project is framed as a market analytics and decision-support product. It is designed to showcase:

- analytical product thinking
- modular Python engineering
- statistical caution
- dashboard UX
- extensible architecture for later data-source and modeling upgrades

## Limitations

- MVP scope is intentionally narrow to avoid noisy multi-market claims
- Demo mode uses synthetic data for app functionality and portfolio presentation
- Opponent context and usage/minutes signals are simple local proxies in version one
- No claim of guaranteed edge or profitability is made anywhere in the product

## Future Improvements

- Add real historical datasets and external source adapters
- Expand from player points to rebounds, assists, and cross-sport modules
- Add richer injury and teammate-absence context
- Add walk-forward predictive benchmarks with calibration checks
- Persist curated outputs to `sqlite` or parquet-based marts
