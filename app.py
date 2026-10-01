import datetime
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Hotel Price Predictor", page_icon="🏨")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
CAT = ["hotel", "arrival_date_month", "meal", "market_segment", "reserved_room_type"]
NUM = ["arrival_date_week_number", "lead_time", "total_nights", "total_guests"]


def make_pipeline():
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep),
                     ("model", HistGradientBoostingRegressor(max_iter=200, random_state=42))])


@st.cache_resource(show_spinner="Training the model (first load only)...")
def load_model():
    df = pd.read_csv("hotel_bookings.csv")
    df = df[(df["adr"] > 0) & (df["adr"] < 1000)]
    df["children"] = df["children"].fillna(0)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df = df[(df["total_nights"] > 0) & (df["total_guests"] > 0)]
    feats = CAT + NUM

    # Honest check: learn from 2015-2016, test on 2017
    train, test = df[df["arrival_date_year"] < 2017], df[df["arrival_date_year"] == 2017]
    check = make_pipeline().fit(train[feats], train["adr"])
    mae = mean_absolute_error(test["adr"], check.predict(test[feats]))

    # Final model learns from all years
    final = make_pipeline().fit(df[feats], df["adr"])
    options = {c: sorted(df[c].dropna().unique().tolist()) for c in
               ["hotel", "meal", "market_segment", "reserved_room_type"]}
    return final, mae, options


model, mae, options = load_model()

st.title("🏨 Hotel Price Predictor")
st.write("Predicts the average nightly price of a booking, trained on 119,000+ real hotel "
         "bookings from a city hotel and a resort hotel in Portugal (2015-2017).")

col1, col2 = st.columns(2)
with col1:
    hotel = st.selectbox("Hotel type", options["hotel"])
    arrival = st.date_input("Arrival date (2017)", datetime.date(2017, 8, 15),
                            min_value=datetime.date(2017, 1, 1), max_value=datetime.date(2017, 12, 31))
    nights = st.slider("Number of nights", 1, 14, 3)
    guests = st.slider("Number of guests", 1, 6, 2)
with col2:
    lead = st.slider("Days booked in advance", 0, 365, 30)
    room = st.selectbox("Room type", options["reserved_room_type"])
    meal = st.selectbox("Meal plan", options["meal"])
    segment = st.selectbox("Booking channel", options["market_segment"])

row = pd.DataFrame([{
    "hotel": hotel,
    "arrival_date_month": MONTHS[arrival.month - 1],
    "meal": meal,
    "market_segment": segment,
    "reserved_room_type": room,
    "arrival_date_week_number": arrival.isocalendar()[1],
    "lead_time": lead,
    "total_nights": nights,
    "total_guests": guests,
}])

price = float(model.predict(row)[0])
st.metric("Predicted price per night", f"{price:,.0f}")
st.caption(f"Typical error: about ±{mae:.0f} per night, measured by training on 2015-2016 "
           "and testing on 2017 bookings. Prices are in the dataset's currency units (euros).")
st.caption("Data: Hotel Booking Demand (Kaggle), CC BY 4.0. Demo model for learning purposes.")