import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "accident_severity"
DROP_COLUMNS = ["accident_id", "date", "time"]
MODEL_FILE = "road_severity_model.pkl"

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "hour",
    "lanes",
    "traffic_signal",
    "temperature",
    "vehicles_involved",
    "casualties",
    "is_weekend",
    "is_peak_hour",
    "risk_score",
    "month",
]

CATEGORICAL_FEATURES = [
    "city",
    "state",
    "road_type",
    "weather",
    "visibility",
    "traffic_density",
    "cause",
    "festival",
    "day_of_week",
    "season",
    "daypart",
]


def load_data(csv_path: str = "indian_roads_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        0: "unknown",
    }
    df["season"] = df["month"].map(season_map)
    df["daypart"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 21, 24],
        labels=["late_night", "morning", "afternoon", "evening", "night"],
        ordered=False,
    ).astype(str)
    df["festival"] = df["festival"].fillna("None").astype(str)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_model(csv_path: str = "indian_roads_dataset.csv", save_path: Optional[str] = None):
    df = load_data(csv_path)
    df = engineer_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    label_encoder = {label: idx for idx, label in enumerate(sorted(y.unique()))}
    inverse_label_encoder = {idx: label for label, idx in label_encoder.items()}
    y_encoded = y.map(label_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[inverse_label_encoder[i] for i in sorted(inverse_label_encoder.keys())],
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "label_encoder": label_encoder,
        "inverse_label_encoder": inverse_label_encoder,
    }
    if save_path is not None:
        save_model(pipeline, metrics, save_path)
    return pipeline, metrics


def save_model(pipeline: Pipeline, metrics: Dict[str, Any], model_path: str = MODEL_FILE) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"pipeline": pipeline, "metrics": metrics}
    with path.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model_path: str = MODEL_FILE) -> Tuple[Pipeline, Dict[str, Any]]:
    path = Path(model_path)
    with path.open("rb") as f:
        artifact = pickle.load(f)
    return artifact["pipeline"], artifact["metrics"]


def get_or_train_model(csv_path: str = "indian_roads_dataset.csv", model_path: str = MODEL_FILE) -> Tuple[Pipeline, Dict[str, Any]]:
    try:
        return load_model(model_path)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        try:
            Path(model_path).unlink()
        except OSError:
            pass
        return train_model(csv_path, save_path=model_path)


def predict_severity(pipeline: Pipeline, sample: dict, csv_path: str = "indian_roads_dataset.csv", model_path: str = MODEL_FILE) -> tuple[str, dict]:
    sample_df = pd.DataFrame([sample])
    sample_df = engineer_features(sample_df)
    try:
        proba = pipeline.predict_proba(sample_df)[0]
        prediction_idx = int(pipeline.predict(sample_df)[0])
    except (AttributeError, ValueError):
        pipeline, _ = train_model(csv_path, save_path=model_path)
        proba = pipeline.predict_proba(sample_df)[0]
        prediction_idx = int(pipeline.predict(sample_df)[0])
    return prediction_idx, proba
