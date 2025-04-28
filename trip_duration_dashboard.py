import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import io

# --- Set a background image and style text ---
def add_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('{image_url}');
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3, p, div, label, span {{
            color: #ffffff;
            font-size: 1.3rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }}
        .big-output-block {{
            background-color: rgba(30, 30, 30, 0.7);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper to download & cache models/encoders ---
@st.cache_resource
def download_and_load_joblib(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        st.error(f"Failed to download file from {url}")
        st.stop()

# --- Model & Encoder URLs ---
DURATION_MODEL_URL    = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_duration_model.pkl"
DISTANCE_MODEL_URL    = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_distance_model.pkl"
FARE_MODEL_URL        = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_fare_model.pkl"
PICKUP_ENCODER_URL    = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/pickup_zone_encoder.pkl"
DROPOFF_ENCODER_URL   = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/dropoff_zone_encoder.pkl"
ZONE_PAIR_ENCODER_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/zone_pair_encoder.pkl"

# --- Hardcoded RMSEs ---
DISTANCE_RMSE = 1.34  # miles
DURATION_RMSE = 5.76  # minutes
FARE_RMSE     = 5.40  # dollars

# --- Load models & encoders ---
with st.spinner("Loading models and encoders…"):
    duration_model = download_and_load_joblib(DURATION_MODEL_URL)
    distance_model = download_and_load_joblib(DISTANCE_MODEL_URL)
    fare_model     = download_and_load_joblib(FARE_MODEL_URL)
    le_pickup      = download_and_load_joblib(PICKUP_ENCODER_URL)
    le_dropoff     = download_and_load_joblib(DROPOFF_ENCODER_URL)
    le_zone_pair   = download_and_load_joblib(ZONE_PAIR_ENCODER_URL)

# --- Background & Title ---
add_bg_from_url("https://miro.medium.com/v2/resize:fit:1400/0*R8QowQaWQlH--sLX.jpg")
st.title("NYC Taxi Trip Estimator")

# --- Sidebar Inputs ---
st.sidebar.header("Input Trip Details")
pickup_zone     = st.sidebar.selectbox("Pickup Zone",     le_pickup.classes_.tolist())
dropoff_zone    = st.sidebar.selectbox("Dropoff Zone",    le_dropoff.classes_.tolist())
hour_of_day     = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week     = st.sidebar.selectbox("Day of Week", list(range(7)))
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

# --- Encode & feature-engineer ---
pickup_enc  = le_pickup.transform([pickup_zone])[0]
dropoff_enc = le_dropoff.transform([dropoff_zone])[0]
# build unordered zone-pair string
zone_pair_str = "_".join(sorted([str(pickup_enc), str(dropoff_enc)]))
zone_pair_enc = int(le_zone_pair.transform([zone_pair_str])[0])

is_rush_hour_enc   = 1 if 7 <= hour_of_day <= 17 else 0
hour_of_day_enc    = hour_of_day
day_of_week_enc    = day_of_week
passenger_count_enc= passenger_count

# --- Distance Prediction ---
X_dist = pd.DataFrame([{
    "pickup_zone_enc":    pickup_enc,
    "dropoff_zone_enc":   dropoff_enc,
    "zone_pair_enc":      zone_pair_enc,
    "hour_of_day_enc":    hour_of_day_enc,
    "day_of_week_enc":    day_of_week_enc,
    "is_rush_hour_enc":   is_rush_hour_enc,
    "passenger_count_enc":passenger_count_enc
}])

distance_pred = distance_model.predict(X_dist)[0]

# --- Duration Prediction ---
X_dur = X_dist.copy()
X_dur['predicted_distance'] = distance_pred
log_dur_pred = duration_model.predict(X_dur)[0]
duration_pred = np.expm1(log_dur_pred)

# --- Fare Prediction ---
X_fare = X_dur.copy()
X_fare['predicted_duration'] = duration_pred
fare_pred = fare_model.predict(X_fare)[0]

# --- Speed Calculation ---
speed_estimate = distance_pred / (duration_pred/60) if duration_pred>0 else 0

# --- Output ---
st.markdown(f"""
<div class='big-output-block'>
  <h2>Trip Prediction Results</h2>
  <p>Estimate how long a taxi ride will take, how much it will cost, and how fast you'll go — based on your trip details.</p>
  <p>Estimated Distance: <b>{distance_pred:.2f} miles</b> ± {DISTANCE_RMSE:.2f}</p>
  <p>Estimated Duration: <b>{duration_pred:.2f} minutes</b> ± {DURATION_RMSE:.2f}</p>
  <p>Estimated Fare: <b>${fare_pred:.2f}</b> ± ${FARE_RMSE:.2f}</p>
  <p>Estimated Average Speed: <b>{speed_estimate:.2f} mph</b></p>
</div>
""", unsafe_allow_html=True)
