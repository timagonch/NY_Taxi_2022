import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and encoders
duration_model = joblib.load("trip_duration_model.pkl")
distance_model = joblib.load("trip_distance_model.pkl")
fare_model = joblib.load("trip_fare_model.pkl")  # New model for fare prediction
le_pickup = joblib.load("pickup_zone_encoder.pkl")
le_dropoff = joblib.load("dropoff_zone_encoder.pkl")

# Get list of original zone names
pickup_zones = le_pickup.classes_.tolist()
dropoff_zones = le_dropoff.classes_.tolist()

st.title("NYC Taxi Trip Estimator")
st.markdown("Estimate how long a taxi ride will take, how much it will cost, and how fast you'll go — all based on your trip details.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Trip Details")
pickup_zone = st.sidebar.selectbox("Pickup Zone", pickup_zones)
dropoff_zone = st.sidebar.selectbox("Dropoff Zone", dropoff_zones)
hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week = st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
rate_code = st.sidebar.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 6])

# --- Feature Engineering ---
pickup_enc = le_pickup.transform([pickup_zone])[0]
dropoff_enc = le_dropoff.transform([dropoff_zone])[0]
is_rush_hour = 1 if hour_of_day in [7, 8, 9, 16, 17, 18] else 0

# --- Distance model input ---
X_dist = pd.DataFrame([{
    "pickup_zone_enc": pickup_enc,
    "dropoff_zone_enc": dropoff_enc,
    "RatecodeID": rate_code,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_rush_hour": is_rush_hour
}])

distance_pred = distance_model.predict(X_dist)[0]

# --- Duration model input ---
X_dur = pd.DataFrame([{
    "pickup_zone_enc": pickup_enc,
    "dropoff_zone_enc": dropoff_enc,
    "RatecodeID": rate_code,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_rush_hour": is_rush_hour,
    "predicted_distance": distance_pred
}])

log_duration_pred = duration_model.predict(X_dur)[0]
duration_pred = np.expm1(log_duration_pred)

# --- Fare model input ---
X_fare = pd.DataFrame([{
    "pickup_zone_enc": pickup_enc,
    "dropoff_zone_enc": dropoff_enc,
    "RatecodeID": rate_code,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_rush_hour": is_rush_hour,
    "predicted_distance": distance_pred,
    "predicted_duration": duration_pred
}])

fare_pred = fare_model.predict(X_fare)[0]

# --- Calculate speed ---
speed_estimate = distance_pred / (duration_pred / 60) if duration_pred > 0 else 0

# --- Output ---
st.subheader("Trip Prediction Results")
st.write(f"Estimated Distance: **{distance_pred:.2f} miles**")
st.write(f"Estimated Duration: **{duration_pred:.2f} minutes**")
st.write(f"Estimated Fare: **${fare_pred:.2f}**")
st.write(f"Estimated Average Speed: **{speed_estimate:.2f} mph**")
