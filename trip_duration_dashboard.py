import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import io

# --- Background image & styling ---
def add_bg_from_url(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('{image_url}');
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3, p, div, label, span {{
            color: #ffffff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }}
        .big-output-block {{
            background-color: rgba(30,30,30,0.7);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}
        </style>
    """, unsafe_allow_html=True)

# --- Helpers to cache downloads ---
@st.cache_resource
def download_and_load_joblib(url: str):
    resp = requests.get(url)
    if resp.status_code == 200:
        return joblib.load(io.BytesIO(resp.content))
    st.error(f"Could not download {url}")
    st.stop()

@st.cache_resource
def load_medians(url: str):
    resp = requests.get(url)
    if resp.status_code == 200:
        return joblib.load(io.BytesIO(resp.content))
    st.error(f"Could not download medians from {url}")
    st.stop()

# --- Remote assets URLs ---
PICKUP_ENCODER_URL    = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/pickup_zone_encoder.pkl"
DROPOFF_ENCODER_URL   = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/dropoff_zone_encoder.pkl"
ZONE_PAIR_ENCODER_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/zone_pair_encoder.pkl"

DURATION_MODEL_URL    = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_duration_model.pkl"
FARE_MODEL_URL        = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_fare_model.pkl"

MEDIANS_URL           = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/zone_pair_medians.pkl"

# --- Load encoders, models & medians ---
with st.spinner("Loading models & data…"):
    le_pickup    = download_and_load_joblib(PICKUP_ENCODER_URL)
    le_dropoff   = download_and_load_joblib(DROPOFF_ENCODER_URL)
    le_zone_pair = download_and_load_joblib(ZONE_PAIR_ENCODER_URL)

    duration_model = download_and_load_joblib(DURATION_MODEL_URL)
    fare_model     = download_and_load_joblib(FARE_MODEL_URL)

    medians = load_medians(MEDIANS_URL)

# --- Page setup ---
add_bg_from_url("https://miro.medium.com/v2/resize:fit:1400/0*R8QowQaWQlH--sLX.jpg")
st.title("NYC Taxi Trip Estimator")

# --- Sidebar inputs ---
st.sidebar.header("Input Trip Details")
pickup_zone     = st.sidebar.selectbox("Pickup Zone",  le_pickup.classes_.tolist())
dropoff_zone    = st.sidebar.selectbox("Dropoff Zone", le_dropoff.classes_.tolist())
hour_of_day     = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week     = st.sidebar.selectbox("Day of Week (0=Mon)", list(range(7)))
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

# --- Encode zones & build the unordered pair key ---
pickup_enc, dropoff_enc = (
    le_pickup.transform([pickup_zone])[0],
    le_dropoff.transform([dropoff_zone])[0]
)
pair_str = "_".join(sorted([str(pickup_enc), str(dropoff_enc)]))

# safe‐transform for zone_pair_enc
try:
    zone_pair_enc = int(le_zone_pair.transform([pair_str])[0])
except ValueError:
    zone_pair_enc = -1

# --- Lookup median distance for this pair ---
global_default   = np.median(list(medians.values()))
median_distance = medians.get(pair_str, global_default)

# --- Rush‐hour flag ---
is_rush_hour = 1 if 7 <= hour_of_day <= 17 else 0

# --- Assemble feature DataFrame for duration & fare models ---
X = pd.DataFrame([{
    "pickup_zone_enc":      pickup_enc,
    "dropoff_zone_enc":     dropoff_enc,
    "zone_pair_enc":        zone_pair_enc,
    "hour_of_day_enc":      hour_of_day,
    "day_of_week_enc":      day_of_week,
    "is_rush_hour_enc":     is_rush_hour,
    "passenger_count_enc":  passenger_count,
    "predicted_distance":   median_distance
}])

# --- Predictions ---
distance_pred = median_distance

# Duration
log_dur_pred  = duration_model.predict(X)[0]
duration_pred = np.expm1(log_dur_pred)

# Fare
X["predicted_duration"] = duration_pred
fare_pred = fare_model.predict(X)[0]

# Speed
speed_estimate = distance_pred / (duration_pred / 60) if duration_pred > 0 else 0

# --- Display results ---
st.markdown(f"""
<div class='big-output-block'>
  <h2>Trip Prediction Results</h2>
  <p>Estimated Distance: <b>{distance_pred:.2f} miles</b></p>
  <p>Estimated Duration: <b>{duration_pred:.2f} minutes</b></p>
  <p>Estimated Fare: <b>${fare_pred:.2f}</b></p>
  <p>Average Speed: <b>{speed_estimate:.2f} mph</b></p>
</div>
""", unsafe_allow_html=True)
