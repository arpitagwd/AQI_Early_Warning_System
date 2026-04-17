"""
app.py
AQI Early Warning System — Streamlit Dashboard

Features:
  - City selector (26 Indian cities)
  - Live data from OpenWeatherMap OR manual input
  - 7-day AQI forecast chart (LSTM)
  - AQI gauge dial
  - Pollutant breakdown bar chart
  - Historical 30-day trend
  - Auto health advisory
  - Population-specific alerts (children, elderly, asthma, healthy)
  - Activity recommender
  - Spike anomaly alert
  - Festival season warning
  - SHAP feature importance chart
  - City vs city comparison
  - Prediction confidence score

Run:
    streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from advisory  import generate_advisory, get_bucket, ACTIVITY_LABELS
from live_api  import fetch_current_aqi, fetch_historical_aqi, get_city_list

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# PAGE CONFIG
st.set_page_config(
    page_title="AQI Early Warning System — India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #e0e0e0;
    margin-bottom: 10px;
  }
  .aqi-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    color: white;
    margin-bottom: 8px;
  }
  .advisory-box {
    padding: 14px 18px;
    border-radius: 10px;
    border-left: 5px solid;
    margin-bottom: 12px;
  }
  .activity-yes { color: #2e7d32; font-weight: 600; }
  .activity-no  { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# LOAD MODELS
@st.cache_resource #Streamlit decorator that runs this function only once per session.
def load_models():
    clf       = joblib.load("models/classifier.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    le_target = joblib.load("models/label_encoder.pkl")
    le_city   = joblib.load("models/city_encoder.pkl")
    le_season = joblib.load("models/season_encoder.pkl")
    features  = joblib.load("models/feature_names.pkl")
    aqi_scaler= joblib.load("models/aqi_scaler.pkl")

    import tensorflow as tf
    lstm = tf.keras.models.load_model("models/lstm_model.keras")

    return clf, scaler, le_target, le_city, le_season, features, aqi_scaler, lstm


@st.cache_data
def load_historical_csv():
    df = pd.read_csv("data/city_day.csv", parse_dates=["Date"])
    df.dropna(subset=["AQI", "AQI_Bucket"], inplace=True)
    return df


# HELPERS
# SEASON_MAP = {
#     12: "Winter", 1: "Winter",  2: "Winter",
#     3:  "Spring", 4: "Spring",  5: "Spring",
#     6:  "Summer", 7: "Summer",  8: "Summer",
#     9:  "Autumn", 10:"Autumn",  11:"Autumn"
# }

SEASON_MAP = {
    12: "Winter",      1: "Winter",      2: "Winter",
    3:  "PreSummer",   4: "PreSummer",
    5:  "Summer",      6: "Summer",
    7:  "Monsoon",     8: "Monsoon",     9: "Monsoon",
    10: "PostMonsoon", 11: "PostMonsoon"
}

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
]

AQI_BAND_COLORS = {
    "Good":         "#00C853",
    "Satisfactory": "#AEEA00",
    "Moderate":     "#FFD600",
    "Poor":         "#FF6D00",
    "Very Poor":    "#DD2C00",
    "Severe":       "#6A1B9A",
}

LOOKBACK = 7


def build_feature_vector(city, pollutants_dict, month, le_city, le_season, hist_aqi):
    """Build one row of features for the classifier."""
    season     = SEASON_MAP[month]
    # city_enc   = int(le_city.transform([city])[0])
    # Safe encoding — if city not in training data, find nearest known city
    CITY_FALLBACK = {
    "Mumbai":    "Hyderabad",   # both coastal, similar industrial profile
    "Pune":      "Hyderabad",
    "Surat":     "Ahmedabad",
}
# If city not trained → use fallback..
    if city in le_city.classes_:
        city_enc = int(le_city.transform([city])[0])
    else:
        fallback = CITY_FALLBACK.get(city, "Delhi")  # default to Delhi
        city_enc = int(le_city.transform([fallback])[0])
        print(f"[INFO] '{city}' not in training data, using '{fallback}' encoding")


    season_enc = int(le_season.transform([season])[0])

    pm25  = pollutants_dict.get("PM2.5", 0)
    pm10  = pollutants_dict.get("PM10",  0)
    no    = pollutants_dict.get("NO",    0)
    no2   = pollutants_dict.get("NO2",   0)
    nox   = pollutants_dict.get("NOx",   0)

    pm_ratio  = pm25 / (pm10  + 1e-5)
    nox_ratio = no2  / (nox   + 1e-5)

    # Use last 7 days of AQI for lag features
    aqi_vals = list(hist_aqi)[-7:]
    while len(aqi_vals) < 7:
        aqi_vals.insert(0, aqi_vals[0] if aqi_vals else 100)

    aqi_lag1  = aqi_vals[-1]
    aqi_lag3  = aqi_vals[-3]
    aqi_lag7  = aqi_vals[-7]
    roll_mean = float(np.mean(aqi_vals))
    roll_std  = float(np.std(aqi_vals))
    delta     = aqi_vals[-1] - aqi_vals[-2] if len(aqi_vals) >= 2 else 0

    is_festival = 1 if month in [10, 11] else 0

    row = (
        POLLUTANTS
        + ["Month", "DayOfWeek", "IsFestivalSeason",
           "PM_ratio", "NOx_ratio",
           "AQI_lag1", "AQI_lag3", "AQI_lag7",
           "AQI_roll7_mean", "AQI_roll7_std",
           "AQI_delta", "City_enc", "Season_enc"]
    )

    values = (
        [pollutants_dict.get(p, 0) for p in POLLUTANTS]
        + [month, 0, is_festival,
           pm_ratio, nox_ratio,
           aqi_lag1, aqi_lag3, aqi_lag7,
           roll_mean, roll_std,
           delta, city_enc, season_enc]
    )

    return np.array([values])


def predict_aqi_bucket(feature_vec, clf, scaler, le_target):
    """Classify AQI bucket + return confidence probabilities."""
    scaled   = scaler.transform(feature_vec)
    pred_idx = clf.predict(scaled)[0]
    probs    = clf.predict_proba(scaled)[0]
    label    = le_target.inverse_transform([pred_idx])[0]
    conf     = float(probs[pred_idx])
    return label, conf, probs, le_target.classes_, scaled


def forecast_7_days(lstm, scaler, aqi_scaler, feature_matrix):
    """
    Use LSTM to predict AQI for next 7 days.
    feature_matrix: shape (lookback, n_features) — last 7 days of scaled features
    """
    seq = feature_matrix[-LOOKBACK:].reshape(1, LOOKBACK, -1)
    preds = []
    current_seq = seq.copy()

    for _ in range(7):
        pred_sc = lstm.predict(current_seq, verbose=0)[0][0]
        pred    = float(aqi_scaler.inverse_transform([[pred_sc]])[0][0])
        preds.append(max(0, pred))
        # Roll the window forward (repeat last row with updated AQI lag)
        next_row = current_seq[0, -1, :].copy()
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = next_row

    return preds


# PLOTLY CHARTS
#speedometer like chart
def gauge_chart(aqi_value: float, bucket_label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = aqi_value,
        title = {"text": "Predicted AQI", "font": {"size": 20}},
        gauge = {
            "axis":  {"range": [0, 500], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,   50],  "color": "#E8F5E9"},
                {"range": [51,  100], "color": "#F9FBE7"},
                {"range": [101, 200], "color": "#FFFDE7"},
                {"range": [201, 300], "color": "#FFF3E0"},
                {"range": [301, 400], "color": "#FFEBEE"},
                {"range": [401, 500], "color": "#F3E5F5"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.75,
                "value": aqi_value
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def forecast_chart(days_ahead: list, forecast_vals: list,
                   hist_dates: list, hist_vals: list) -> go.Figure:
    fig = go.Figure()

    # AQI band background zones
    bands = [(0,50,"Good","#E8F5E9"),(51,100,"Satisfactory","#F9FBE7"),
             (101,200,"Moderate","#FFFDE7"),(201,300,"Poor","#FFF3E0"),
             (301,400,"Very Poor","#FFEBEE"),(401,500,"Severe","#F3E5F5")]
    for lo, hi, label, col in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.4,
                      line_width=0, annotation_text=label,
                      annotation_position="right",
                      annotation_font_size=10)

    # Historical line
    if hist_dates and hist_vals:
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_vals,
            mode="lines", name="Historical AQI",
            line=dict(color="#78909C", width=1.5, dash="dot")
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=days_ahead, y=forecast_vals,
        mode="lines+markers", name="7-day forecast",
        line=dict(color="#7F77DD", width=2.5),
        marker=dict(size=8, color="#7F77DD"),
        text=[f"AQI: {v:.0f}" for v in forecast_vals],
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title="7-day AQI forecast",
        xaxis_title="Date", yaxis_title="AQI",
        yaxis_range=[0, 500],
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=120, t=60, b=40)
    )
    return fig


def pollutant_bar(pollutants_dict: dict) -> go.Figure:
    keys = [k for k in POLLUTANTS if k in pollutants_dict]
    vals = [pollutants_dict[k] for k in keys]

    colors = ["#E24B4A" if v > 100 else "#7F77DD" for v in vals]

    fig = go.Figure(go.Bar(
        x=keys, y=vals,
        marker_color=colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(
        title="Pollutant breakdown",
        yaxis_title="Concentration (µg/m³)",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


# def historical_trend_chart(df_city: pd.DataFrame) -> go.Figure:
#     recent = df_city.sort_values("Date").tail(30)
#     fig = px.line(
#         recent, x="Date", y="AQI",
#         title="Historical AQI — last 30 days",
#         color_discrete_sequence=["#1D9E75"]
#     )
#     fig.update_layout(height=300, margin=dict(l=40, r=20, t=50, b=40))
#     return fig


def historical_trend_chart(df_city: pd.DataFrame, live_hist: pd.DataFrame = None) -> go.Figure:
    
    # If live historical data is available, use it
    if live_hist is not None and len(live_hist) > 0:
        recent = live_hist.sort_values("Date").tail(30)
        title  = "Historical AQI — last 30 days (live)"
    else:
        # Fall back to Kaggle CSV data
        recent = df_city.sort_values("Date").tail(30)
        date_range = f"{recent['Date'].min().strftime('%b %Y')} to {recent['Date'].max().strftime('%b %Y')}"
        title  = f"Historical AQI — {date_range} (training data)"

    fig = px.line(
        recent, x="Date", y="AQI",
        title=title,
        color_discrete_sequence=["#1D9E75"]
    )
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=50, b=40))
    return fig

def city_vs_city_chart(city1_name, city1_vals, city2_name, city2_vals, days) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=city1_vals, mode="lines+markers",
        name=city1_name, line=dict(color="#7F77DD", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=days, y=city2_vals, mode="lines+markers",
        name=city2_name, line=dict(color="#1D9E75", width=2)
    ))
    fig.update_layout(
        title="City vs city — 7-day AQI forecast",
        yaxis_title="AQI", height=340,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig


# SHAP PLOT (matplotlib → image in Streamlit)
def show_shap_plot(clf, scaled_input, feature_names):
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(scaled_input)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(7, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = sv[0],
            base_values   = explainer.expected_value[0] if isinstance(
                explainer.expected_value, list) else explainer.expected_value,
            data          = scaled_input[0],
            feature_names = feature_names
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# MAIN APP
def main():
    st.title("🌫️  AQI Early Warning System — India")
    st.caption("Predicts Air Quality Index up to 7 days ahead with health advisories")

    # Load models
    try:
        clf, scaler, le_target, le_city, le_season, features, aqi_scaler, lstm = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Models not found. Run `python train.py` first.\n\nError: {e}")
        models_loaded = False
        return

    hist_csv = load_historical_csv()

    # ── SIDEBAR
    with st.sidebar:
        st.header("Settings")

        city = st.selectbox("Select city", get_city_list(),
                            index=get_city_list().index("Delhi"))

        data_mode = st.radio(
            "Data source",
            ["Live (OpenWeatherMap API)", "Manual input"]
        )

        if data_mode == "Live (OpenWeatherMap API)":
            api_key = st.text_input(
                "OpenWeatherMap API key",
                type="password",
                help="Free key at openweathermap.org/api/air-pollution"
            )
        else:
            api_key = None

        st.divider()
        st.subheader("Comparison")
        compare_mode = st.checkbox("Enable city vs city comparison")
        city2 = None
        if compare_mode:
            city2 = st.selectbox(
                "Compare with", get_city_list(),
                index=get_city_list().index("Bengaluru")
            )

        st.divider()
        st.subheader("Your profile")
        user_group = st.selectbox(
            "I am...",
            ["healthy", "children", "elderly", "asthma"]
        )

        predict_btn = st.button("Run prediction", type="primary", use_container_width=True)

    # ── GET POLLUTANT READINGS 
    live_data    = None
    poll_values  = {}
    month        = pd.Timestamp.now().month

    if data_mode == "Live (OpenWeatherMap API)" and api_key:
        with st.spinner(f"Fetching live data for {city}..."):
            live_data = fetch_current_aqi(city, api_key)

        if live_data:
            poll_values = {p: live_data.get(p, 0.0) for p in POLLUTANTS}
            st.success(f"Live data loaded — {live_data['timestamp']}")
        else:
            st.warning("Live data unavailable. Switching to manual input.")

    if not poll_values:
        with st.expander("Enter pollutant readings (µg/m³)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                poll_values["PM2.5"]   = st.number_input("PM2.5",   0.0, 1000.0, 85.0)
                poll_values["NO"]      = st.number_input("NO",      0.0, 500.0,  12.0)
                poll_values["CO"]      = st.number_input("CO",      0.0, 100.0,   1.8)
                poll_values["Benzene"] = st.number_input("Benzene", 0.0, 100.0,   2.1)
            with c2:
                poll_values["PM10"]    = st.number_input("PM10",    0.0, 1000.0, 140.0)
                poll_values["NO2"]     = st.number_input("NO2",     0.0, 500.0,  38.0)
                poll_values["SO2"]     = st.number_input("SO2",     0.0, 500.0,  18.0)
                poll_values["Toluene"] = st.number_input("Toluene", 0.0, 200.0,   5.5)
            with c3:
                poll_values["NOx"]     = st.number_input("NOx",     0.0, 500.0,  50.0)
                poll_values["NH3"]     = st.number_input("NH3",     0.0, 200.0,  14.0)
                poll_values["O3"]      = st.number_input("O3",      0.0, 500.0,  40.0)
                month = st.slider("Month", 1, 12, pd.Timestamp.now().month)

    if not predict_btn:
        st.info("Configure inputs in the sidebar, then click **Run prediction**.")
        return

    # ── PREDICTION 
    # Get historical AQI for this city (last 30 days from CSV)
    # city_hist = hist_csv[hist_csv["City"] == city].sort_values("Date")
    # hist_aqi  = list(city_hist["AQI"].tail(30))

    city_hist = hist_csv[hist_csv["City"] == city].sort_values("Date")

# If city has no historical data (e.g. Mumbai), use a nearby city's history
    if len(city_hist) == 0:
        HIST_FALLBACK = {
        "Mumbai": "Hyderabad",
    }
        fallback_city = HIST_FALLBACK.get(city, "Delhi")
        city_hist = hist_csv[hist_csv["City"] == fallback_city].sort_values("Date")
        print(f"[INFO] No history for '{city}', using '{fallback_city}' history for lag features")

    hist_aqi = list(city_hist["AQI"].tail(30))


    feature_vec = build_feature_vector(
        city, poll_values, month,
        le_city, le_season, hist_aqi
    )

    bucket_label, confidence, probs, classes, scaled_vec = predict_aqi_bucket(
        feature_vec, clf, scaler, le_target
    )
    bucket     = get_bucket(0)   # placeholder — use label
    bucket_col = {
        "Good": "#00C853", "Satisfactory": "#AEEA00",
        "Moderate": "#FFD600", "Poor": "#FF6D00",
        "Very Poor": "#DD2C00", "Severe": "#6A1B9A"
    }.get(bucket_label, "#888")

    # Approx numeric AQI from bucket midpoints (for gauge)
    bucket_midpoints = {
        "Good": 25, "Satisfactory": 75,
        "Moderate": 150, "Poor": 250,
        "Very Poor": 350, "Severe": 450
    }
    approx_aqi = bucket_midpoints.get(bucket_label, 150)

    # LSTM 7-day forecast
    scaled_hist = scaler.transform(feature_vec)
    lstm_seq    = np.repeat(scaled_hist, LOOKBACK, axis=0)
    forecast    = forecast_7_days(lstm, scaler, aqi_scaler, lstm_seq)

    today     = pd.Timestamp.now().normalize()
    fore_days = [(today + pd.Timedelta(days=i+1)).strftime("%b %d") for i in range(7)]

    # Advisory
    prev_aqi  = hist_aqi[-1] if hist_aqi else None
    advisory  = generate_advisory(approx_aqi, prev_aqi, month)

    # ── LAYOUT: TOP METRICS 
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

    with col1:
        st.markdown(
            f"<div style='background:{bucket_col}22; border:1px solid {bucket_col};"
            f"border-radius:12px; padding:16px 20px;'>"
            f"<div style='font-size:12px; color:#666; margin-bottom:4px'>AQI Category</div>"
            f"<div style='font-size:28px; font-weight:700; color:{bucket_col}'>{bucket_label}</div>"
            f"<div style='font-size:13px; color:#888; margin-top:4px'>"
            f"~{approx_aqi} AQI  •  {confidence*100:.0f}% confidence</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.metric("7-day avg forecast", f"{np.mean(forecast):.0f} AQI")
    with col3:
        peak_day = fore_days[np.argmax(forecast)]
        st.metric("Peak day", peak_day, f"{max(forecast):.0f} AQI")
    with col4:
        trend = "improving" if forecast[-1] < forecast[0] else "worsening"
        delta = forecast[-1] - forecast[0]
        st.metric("7-day trend", trend, f"{delta:+.0f}")

    # ── ROW 1: GAUGE + FORECAST CHART 
    st.markdown("---")
    g_col, f_col = st.columns([1, 2.5])

    with g_col:
        st.plotly_chart(
            gauge_chart(approx_aqi, bucket_label, bucket_col),
            use_container_width=True
        )

    with f_col:
        hist_tail   = city_hist.tail(14)
        hist_dates  = list(hist_tail["Date"].dt.strftime("%b %d"))
        hist_values = list(hist_tail["AQI"])
        st.plotly_chart(
            forecast_chart(fore_days, forecast, hist_dates, hist_values),
            use_container_width=True
        )

    # ── ROW 2: POLLUTANT CHART + HISTORICAL TREND 
    pb_col, ht_col = st.columns(2)
    with pb_col:
        st.plotly_chart(pollutant_bar(poll_values), use_container_width=True)
    with ht_col:
        # st.plotly_chart(historical_trend_chart(city_hist), use_container_width=True)
        # Fetch live historical data if API mode
        live_hist = None
        if data_mode == "Live (OpenWeatherMap API)" and api_key:
            with st.spinner("Fetching historical data..."):
                live_hist = fetch_historical_aqi(city, days=30, api_key=api_key)

        # Pass it to the chart
        st.plotly_chart(
            historical_trend_chart(city_hist, live_hist),
            use_container_width=True
)

    # ── ROW 3: HEALTH ADVISORY 
    st.markdown("---")
    st.subheader("Health Advisory")

    adv = advisory

    # Spike alert
    if adv["spike_alert"]:
        st.error(f"⚠️  {adv['spike_message']}")

    # Festival warning
    if adv["festival_warning"]:
        st.warning(f"🪔  {adv['festival_message']}")

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown(
            f"<div class='advisory-box' style='border-color:{bucket_col}; "
            f"background:{bucket_col}18'>"
            f"<strong>General advisory</strong><br>{adv['general']}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("**Your profile advisory**")
        profile_msg = adv["population"].get(user_group, "")
        st.info(profile_msg)

    with adv_col2:
        st.markdown("**All population groups**")
        for group, msg in adv["population"].items():
            icon = {"children":"👶", "elderly":"🧓", "asthma":"💨", "healthy":"🏃"}[group]
            with st.expander(f"{icon} {group.title()}"):
                st.write(msg)

    # ── ROW 4: ACTIVITY RECOMMENDER 
    st.markdown("---")
    st.subheader("Activity Recommender")
    acts = adv["activities"]

    act_cols = st.columns(4)
    icons    = {"jogging":"🏃", "school_sports":"🏫", "cycling":"🚴", "outdoor_dining":"🍽️"}
    for i, (key, label) in enumerate(ACTIVITY_LABELS.items()):
        allowed = acts[key]
        color   = "#2e7d32" if allowed else "#c62828"
        verdict = "SAFE" if allowed else "AVOID"
        with act_cols[i]:
            st.markdown(
                f"<div style='text-align:center; background:{'#E8F5E9' if allowed else '#FFEBEE'};"
                f"border-radius:10px; padding:14px 10px;'>"
                f"<div style='font-size:28px'>{icons[key]}</div>"
                f"<div style='font-size:13px; color:#555; margin-top:4px'>{label}</div>"
                f"<div style='font-size:16px; font-weight:700; color:{color}; "
                f"margin-top:6px'>{verdict}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── ROW 5: SHAP EXPLAINABILITY 
    st.markdown("---")
    st.subheader("Why this prediction? (SHAP)")
    st.caption("Shows which pollutants pushed the AQI prediction up or down")

    with st.spinner("Computing SHAP values..."):
        show_shap_plot(clf, scaled_vec, features)

    # ── ROW 6: CONFIDENCE BREAKDOWN 
    st.markdown("---")
    st.subheader("Prediction confidence across AQI categories")

    conf_df = pd.DataFrame({
        "Category":   classes,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    conf_fig = px.bar(
        conf_df, x="Category", y="Probability",
        color="Probability",
        color_continuous_scale=["#E8F5E9", "#7F77DD"],
        text=[f"{p*100:.1f}%" for p in conf_df["Probability"]]
    )
    conf_fig.update_layout(
        height=280, showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40)
    )
    conf_fig.update_traces(textposition="outside")
    st.plotly_chart(conf_fig, use_container_width=True)

    # ── ROW 7: CITY VS CITY 
    if compare_mode and city2:
        st.markdown("---")
        st.subheader(f"City comparison — {city} vs {city2}")

        # city2_hist = hist_csv[hist_csv["City"] == city2].sort_values("Date")
        # city2_aqi  = list(city2_hist["AQI"].tail(30))

        city2_hist = hist_csv[hist_csv["City"] == city2].sort_values("Date")

        if len(city2_hist) == 0:
            HIST_FALLBACK = {"Mumbai": "Hyderabad"}
            fallback_city2 = HIST_FALLBACK.get(city2, "Delhi")
            city2_hist = hist_csv[hist_csv["City"] == fallback_city2].sort_values("Date")

        city2_aqi = list(city2_hist["AQI"].tail(30))

        poll2 = {p: city2_hist[p].iloc[-1] if p in city2_hist.columns
                 else poll_values.get(p, 0) for p in POLLUTANTS}

        fvec2 = build_feature_vector(
            city2, poll2, month, le_city, le_season, city2_aqi
        )
        scaled2 = scaler.transform(fvec2)
        lstm_seq2 = np.repeat(scaled2, LOOKBACK, axis=0)
        forecast2 = forecast_7_days(lstm, scaler, aqi_scaler, lstm_seq2)

        bucket2, conf2, _, _, _ = predict_aqi_bucket(fvec2, clf, scaler, le_target)

        cc1, cc2 = st.columns(2)
        with cc1:
            bc = {"Good":"#00C853","Satisfactory":"#AEEA00","Moderate":"#FFD600",
                  "Poor":"#FF6D00","Very Poor":"#DD2C00","Severe":"#6A1B9A"}
            c1_col = bc.get(bucket_label, "#888")
            st.markdown(
                f"<div style='text-align:center; background:{c1_col}18; "
                f"border:1px solid {c1_col}; border-radius:10px; padding:14px'>"
                f"<b>{city}</b><br>"
                f"<span style='font-size:22px; font-weight:700; color:{c1_col}'>"
                f"{bucket_label}</span><br>"
                f"<span style='font-size:12px;color:#888'>{confidence*100:.0f}% confidence</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        with cc2:
            c2_col = bc.get(bucket2, "#888")
            st.markdown(
                f"<div style='text-align:center; background:{c2_col}18; "
                f"border:1px solid {c2_col}; border-radius:10px; padding:14px'>"
                f"<b>{city2}</b><br>"
                f"<span style='font-size:22px; font-weight:700; color:{c2_col}'>"
                f"{bucket2}</span><br>"
                f"<span style='font-size:12px;color:#888'>{conf2*100:.0f}% confidence</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.plotly_chart(
            city_vs_city_chart(city, forecast, city2, forecast2, fore_days),
            use_container_width=True
        )

    # ── FOOTER 
    st.markdown("---")
    st.caption(
        "Data: Kaggle India AQI dataset (2015–2020) + OpenWeatherMap Air Pollution API  |  "
        "Models: Random Forest · XGBoost · LSTM  |  "
        "Built for AQI Early Warning System project"
    )


if __name__ == "__main__":
    main()