# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import requests
# import io

# # --- Helper function to download and cache models ---
# @st.cache_resource
# def download_and_load_model(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         return joblib.load(io.BytesIO(response.content))
#     else:
#         st.error(f"Failed to download model from {url}")
#         st.stop()

# # --- Model URLs (HuggingFace) ---
# DURATION_MODEL_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_duration_model.pkl"
# DISTANCE_MODEL_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_distance_model.pkl"
# FARE_MODEL_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_fare_model.pkl"

# # --- Load models from Hugging Face ---
# with st.spinner("Loading machine learning models..."):
#     duration_model = download_and_load_model(DURATION_MODEL_URL)
#     distance_model = download_and_load_model(DISTANCE_MODEL_URL)
#     fare_model = download_and_load_model(FARE_MODEL_URL)

# # --- Load encoders locally (should be in your repo root) ---
# le_pickup = joblib.load("pickup_zone_encoder.pkl")
# le_dropoff = joblib.load("dropoff_zone_encoder.pkl")

# # --- Streamlit UI ---
# st.title("NYC Taxi Trip Estimator")
# st.markdown("Estimate how long a taxi ride will take, how much it will cost, and how fast you'll go — based on your trip details.")

# # --- Sidebar Inputs ---
# st.sidebar.header("Input Trip Details")
# pickup_zone = st.sidebar.selectbox("Pickup Zone", le_pickup.classes_.tolist())
# dropoff_zone = st.sidebar.selectbox("Dropoff Zone", le_dropoff.classes_.tolist())
# hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 8)
# day_of_week = st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
# passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

# # --- Feature Engineering ---
# pickup_enc = le_pickup.transform([pickup_zone])[0]
# dropoff_enc = le_dropoff.transform([dropoff_zone])[0]
# is_rush_hour = 1 if 7 <= hour_of_day <= 17 else 0

# # --- Distance Prediction ---
# X_dist = pd.DataFrame([{
#     "pickup_zone_enc": pickup_enc,
#     "dropoff_zone_enc": dropoff_enc,
#     "hour_of_day": hour_of_day,
#     "day_of_week": day_of_week,
#     "is_rush_hour": is_rush_hour,
#     "passenger_count": passenger_count
# }])

# distance_pred = distance_model.predict(X_dist)[0]

# # --- Duration Prediction ---
# X_dur = X_dist.copy()
# X_dur["predicted_distance"] = distance_pred

# log_duration_pred = duration_model.predict(X_dur)[0]
# duration_pred = np.expm1(log_duration_pred)

# # --- Fare Prediction ---
# X_fare = X_dur.copy()
# X_fare["predicted_duration"] = duration_pred

# fare_pred = fare_model.predict(X_fare)[0]

# # --- Speed Calculation ---
# speed_estimate = distance_pred / (duration_pred / 60) if duration_pred > 0 else 0

# # --- Output ---
# st.subheader("Trip Prediction Results")
# st.write(f"Estimated Distance: **{distance_pred:.2f} miles**")
# st.write(f"Estimated Duration: **{duration_pred:.2f} minutes**")
# st.write(f"Estimated Fare: **${fare_pred:.2f}**")
# st.write(f"Estimated Average Speed: **{speed_estimate:.2f} mph**")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import io

# --- Set a background image and dark transparent overlay ---
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
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper function to download and cache models/encoders ---
@st.cache_resource
def download_and_load_joblib(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        st.error(f"Failed to download file from {url}")
        st.stop()

# --- Model and Encoder URLs ---
DURATION_MODEL_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_duration_model.pkl"
DISTANCE_MODEL_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_distance_model.pkl"
FARE_MODEL_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/trip_fare_model.pkl"
PICKUP_ENCODER_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/pickup_zone_encoder.pkl"
DROPOFF_ENCODER_URL = "https://huggingface.co/datasets/timagonch/nyc-taxi-trip-models/resolve/main/dropoff_zone_encoder.pkl"

# --- Hardcoded RMSE values based on model evaluation ---
DISTANCE_RMSE = 1.34  # miles
DURATION_RMSE = 5.76  # minutes
FARE_RMSE = 5.40      # dollars

# --- Load models and encoders ---
with st.spinner("Loading machine learning models and encoders..."):
    duration_model = download_and_load_joblib(DURATION_MODEL_URL)
    distance_model = download_and_load_joblib(DISTANCE_MODEL_URL)
    fare_model = download_and_load_joblib(FARE_MODEL_URL)
    le_pickup = download_and_load_joblib(PICKUP_ENCODER_URL)
    le_dropoff = download_and_load_joblib(DROPOFF_ENCODER_URL)

# --- Set background image ---
add_bg_from_url("https://miro.medium.com/v2/resize:fit:1400/0*R8QowQaWQlH--sLX.jpg")  # Replace with final background image URL

# --- Main container ---
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.title("NYC Taxi Trip Estimator")
    st.markdown("Estimate how long a taxi ride will take, how much it will cost, and how fast you'll go — based on your trip details.")

    # --- Sidebar Inputs ---
    st.sidebar.header("Input Trip Details")
    pickup_zone = st.sidebar.selectbox("Pickup Zone", le_pickup.classes_.tolist())
    dropoff_zone = st.sidebar.selectbox("Dropoff Zone", le_dropoff.classes_.tolist())
    hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
    passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

    # --- Feature Engineering ---
    pickup_enc = le_pickup.transform([pickup_zone])[0]
    dropoff_enc = le_dropoff.transform([dropoff_zone])[0]
    is_rush_hour = 1 if 7 <= hour_of_day <= 17 else 0

    # --- Distance Prediction ---
    X_dist = pd.DataFrame([{
        "pickup_zone_enc": pickup_enc,
        "dropoff_zone_enc": dropoff_enc,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_rush_hour": is_rush_hour,
        "passenger_count": passenger_count
    }])

    distance_pred = distance_model.predict(X_dist)[0]

    # --- Duration Prediction ---
    X_dur = X_dist.copy()
    X_dur['predicted_distance'] = distance_pred

    log_duration_pred = duration_model.predict(X_dur)[0]
    duration_pred = np.expm1(log_duration_pred)

    # --- Fare Prediction ---
    X_fare = X_dur.copy()
    X_fare['predicted_duration'] = duration_pred

    fare_pred = fare_model.predict(X_fare)[0]

    # --- Speed Calculation ---
    speed_estimate = distance_pred / (duration_pred / 60) if duration_pred > 0 else 0

    # --- Output ---
    st.subheader("Trip Prediction Results")

    st.write(f"Estimated Distance: **{distance_pred:.2f} miles** ± {DISTANCE_RMSE:.2f}")
    st.write(f"Estimated Duration: **{duration_pred:.2f} minutes** ± {DURATION_RMSE:.2f}")
    st.write(f"Estimated Fare: **{fare_pred:.2f} dollars** ± {FARE_RMSE:.2f} dollars")
    st.write(f"Estimated Average Speed: **{speed_estimate:.2f} mph**")

    st.markdown("</div>", unsafe_allow_html=True)
