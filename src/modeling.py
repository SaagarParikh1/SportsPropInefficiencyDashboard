from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_FEATURES = [
    "line",
    "rolling_avg_5",
    "rolling_avg_10",
    "season_avg_prior",
    "line_minus_recent_avg",
    "line_minus_season_avg",
    "trend_direction",
    "consistency_score",
    "matchup_difficulty_score",
    "days_rest",
    "back_to_back",
]


def temporal_train_test_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("game_date").reset_index(drop=True)
    cutoff = max(1, int(len(ordered) * (1.0 - test_fraction)))
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def _build_numeric_pipeline(model) -> Pipeline:
    numeric_features = DEFAULT_MODEL_FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_logistic_direction_model(df: pd.DataFrame) -> dict[str, object]:
    model_frame = df.dropna(subset=["game_date"]).copy()
    model_frame["target_over"] = (model_frame["actual_value"] > model_frame["line"]).astype(int)
    model_frame = model_frame.dropna(subset=["target_over"])
    if len(model_frame) < 50:
        return {"status": "insufficient_data", "message": "Need at least 50 rows for a basic benchmark."}

    train, test = temporal_train_test_split(model_frame)
    if train.empty or test.empty:
        return {"status": "insufficient_data", "message": "Temporal split did not produce train and test samples."}

    pipeline = _build_numeric_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(train[DEFAULT_MODEL_FEATURES], train["target_over"])
    predicted_prob = pipeline.predict_proba(test[DEFAULT_MODEL_FEATURES])[:, 1]
    predicted_label = (predicted_prob >= 0.5).astype(int)

    logistic_model = pipeline.named_steps["model"]
    feature_importance = pd.DataFrame(
        {
            "feature": DEFAULT_MODEL_FEATURES,
            "coefficient": logistic_model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)

    return {
        "status": "ok",
        "model_name": "Logistic Regression",
        "accuracy": accuracy_score(test["target_over"], predicted_label),
        "roc_auc": roc_auc_score(test["target_over"], predicted_prob),
        "brier_score": brier_score_loss(test["target_over"], predicted_prob),
        "feature_importance": feature_importance.reset_index(drop=True),
    }


def train_random_forest_benchmark(df: pd.DataFrame) -> dict[str, object]:
    model_frame = df.dropna(subset=["game_date"]).copy()
    model_frame["target_over"] = (model_frame["actual_value"] > model_frame["line"]).astype(int)
    if len(model_frame) < 75:
        return {"status": "insufficient_data", "message": "Need at least 75 rows for the benchmark forest."}

    train, test = temporal_train_test_split(model_frame)
    pipeline = _build_numeric_pipeline(RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5))
    pipeline.fit(train[DEFAULT_MODEL_FEATURES], train["target_over"])
    predicted_prob = pipeline.predict_proba(test[DEFAULT_MODEL_FEATURES])[:, 1]
    predicted_label = (predicted_prob >= 0.5).astype(int)

    return {
        "status": "ok",
        "model_name": "Random Forest",
        "accuracy": accuracy_score(test["target_over"], predicted_label),
        "roc_auc": roc_auc_score(test["target_over"], predicted_prob),
        "brier_score": brier_score_loss(test["target_over"], predicted_prob),
    }
