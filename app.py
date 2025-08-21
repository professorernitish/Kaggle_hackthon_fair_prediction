import pandas as pd
import numpy as np
import pickle
import streamlit as st

## loading the model:
with open('model.pickle','rb') as f:   # use binary mode
    model = pickle.load(f)

st.title("Fair Price Prediction :- ")

# Input features (excluding fare_amount)
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=6, step=1)
trip_distance = st.number_input('Trip Distance (miles/km)', min_value=0.0, step=0.1)
pu_location_id = st.number_input('Pickup Location ID', min_value=1, step=1)
do_location_id = st.number_input('Dropoff Location ID', min_value=1, step=1)
trip_duration_minutes = st.number_input('Trip Duration (minutes)', min_value=0.0, step=1.0)

pickup_year = st.number_input('Pickup Year', min_value=2000, max_value=2100, step=1)
pickup_month = st.selectbox('Pickup Month', list(range(1, 13)))
pickup_day = st.slider('Pickup Day', 1, 31, 1)
pickup_hour = st.slider('Pickup Hour', 0, 23, 0)

# Create dataframe without fare_amount
input_data = pd.DataFrame({
    'passenger_count': [passenger_count],
    'trip_distance': [trip_distance],
    'PULocationID': [pu_location_id],
    'DOLocationID': [do_location_id],
    'trip_duration_minutes': [trip_duration_minutes],
    'pickup_year': [pickup_year],
    'pickup_month': [pickup_month],
    'pickup_day': [pickup_day],
    'pickup_hour': [pickup_hour]
})

# Predict fare
predicted_fare = model.predict(input_data)

# Predict fare with condition
if trip_distance == 0 and trip_duration_minutes == 0:
    predicted_fare = [0]
else:
    predicted_fare = model.predict(input_data)

st.write(f"### Predicted Fare Amount: ${predicted_fare[0]:.2f}")
