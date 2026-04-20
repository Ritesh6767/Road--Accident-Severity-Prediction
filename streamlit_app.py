import streamlit as st
import pandas as pd
from model_pipeline import get_or_train_model, load_data, predict_severity

st.set_page_config(page_title="Road Accident Severity Prediction", layout="wide")

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

def set_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #071926;
            color: #e9f1ff;
        }
        [data-testid="stSidebar"] > div {
            background-color: #081f2f;
            color: #f4f7ff;
        }
        .css-1aumxhk, .css-1d391kg, .css-1lcbmhc, .css-18e3th9 {
            background-color: #071926;
            color: #e9f1ff;
        }
        div.stButton > button {
            background-color: #ff6b4a;
            color: white;
            font-weight: 700;
            border: 1px solid #ff8c66;
            box-shadow: 0 4px 12px rgba(255, 107, 74, 0.35);
        }
        div.stButton > button:hover {
            background-color: #ff8c66;
        }
        .st-bc {
            color: #e9f1ff;
        }
        .stMarkdown p, .stText {
            color: #e9f1ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_data("indian_roads_dataset.csv")

try:
    cache_model = st.cache_resource
except AttributeError:
    def cache_model(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@cache_model(show_spinner=False)
def get_model_and_metrics():
    return get_or_train_model("indian_roads_dataset.csv")


def main():
    st.title("Road Accident Severity Prediction")
    st.write(
        "Use feature engineering, preprocessing, and a Scikit-learn pipeline to predict whether an accident is likely to be `minor`, `major`, or `fatal`."
    )

    set_theme()
    df = get_dataset()
    pipeline, metrics = get_model_and_metrics()

    sidebar = st.sidebar
    sidebar.header("Input features")

    city = sidebar.selectbox("City", sorted(df["city"].unique()))
    state = sidebar.selectbox("State", sorted(df["state"].unique()))
    day_of_week_options = [d for d in WEEKDAY_ORDER if d in df["day_of_week"].unique()]
    day_of_week = sidebar.selectbox("Day of week", day_of_week_options)
    road_type = sidebar.selectbox("Road type", sorted(df["road_type"].unique()))
    weather = sidebar.selectbox("Weather", sorted(df["weather"].unique()))
    visibility = sidebar.selectbox("Visibility", sorted(df["visibility"].unique()))
    traffic_density = sidebar.selectbox("Traffic density", sorted(df["traffic_density"].unique()))
    cause = sidebar.selectbox("Accident cause", sorted(df["cause"].unique()))
    festival = sidebar.selectbox(
        "Festival", sorted(df["festival"].fillna("None").unique())
    )
    hour = sidebar.slider("Hour of day", min_value=0, max_value=23, value=12)
    temperature = sidebar.slider(
        "Temperature (°C)", min_value=int(df["temperature"].min()), max_value=int(df["temperature"].max()), value=int(df["temperature"].median())
    )
    lanes = sidebar.slider("Number of lanes", min_value=int(df["lanes"].min()), max_value=int(df["lanes"].max()), value=3)
    traffic_signal = sidebar.selectbox("Traffic signal present", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    vehicles_involved = sidebar.slider("Vehicles involved", min_value=int(df["vehicles_involved"].min()), max_value=int(df["vehicles_involved"].max()), value=2)
    casualties = sidebar.slider("Casualties", min_value=int(df["casualties"].min()), max_value=int(df["casualties"].max()), value=1)
    is_weekend = sidebar.selectbox("Weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_peak_hour = sidebar.selectbox("Peak hour", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    risk_score = sidebar.slider("Risk score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    month = sidebar.selectbox("Month", list(range(1, 13)), index=8)

    input_data = {
        "city": city,
        "state": state,
        "road_type": road_type,
        "weather": weather,
        "visibility": visibility,
        "traffic_density": traffic_density,
        "cause": cause,
        "festival": festival,
        "day_of_week": day_of_week,
        "hour": hour,
        "temperature": temperature,
        "lanes": lanes,
        "traffic_signal": traffic_signal,
        "vehicles_involved": vehicles_involved,
        "casualties": casualties,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "risk_score": risk_score,
        "date": f"2024-{month:02d}-15",
        "time": f"{hour}:00",
        "latitude": float(df["latitude"].median()),
        "longitude": float(df["longitude"].median()),
    }

    st.write("## Live prediction")
    st.write("Adjust the inputs in the sidebar and click Predict to evaluate accident severity.")

    if st.button("Predict Accident Severity"):
        prediction_idx, probabilities = predict_severity(pipeline, input_data)
        inverse_label_encoder = metrics["inverse_label_encoder"]
        labels = [inverse_label_encoder[i] for i in sorted(inverse_label_encoder.keys())]
        prediction = labels[prediction_idx]
        st.success(f"Predicted severity: {prediction.title()}")
        proba_df = pd.DataFrame(
            {"Severity": labels, "Probability": probabilities}
        ).sort_values("Probability", ascending=False)
        st.table(proba_df)
        st.write("### Target distribution")
        st.bar_chart(df["accident_severity"].value_counts())

    st.write("## Model performance")
    st.write(f"Accuracy : {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
