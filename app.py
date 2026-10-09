import datetime as dt

import pandas as pd
import streamlit as st

from predictor import find_file, load_bundle, predict

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="centered")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_tables():
    airlines = pd.read_csv(find_file("airlines.csv"))
    airports = pd.read_csv(find_file("airports.csv")).sort_values("code")
    routes = pd.read_csv(find_file("routes.csv")).set_index("route")
    return airlines, airports, routes


bundle, (airlines, airports, routes) = get_bundle(), get_tables()
airline_names = dict(zip(airlines["AIRLINE"], airlines["AIRLINE_CODE"]))
airport_label = {r.code: f"{r.code} — {r.city}" for r in airports.itertuples()}
codes = list(airport_label)

st.title("✈️ Flight Delay Predictor")
st.caption("Will your flight arrive more than 15 minutes late? Model trained on 100k US flights (2019–2023).")

mode_label = st.radio(
    "When are you checking?",
    ["Before departure (flight hasn't left yet)", "At the gate / after departure (I know the departure delay)"],
)
mode = "A_pre_departure" if mode_label.startswith("Before") else "B_at_gate"

c1, c2 = st.columns(2)
airline = c1.selectbox("Airline", sorted(airline_names))
date = c2.date_input("Flight date", dt.date.today())
origin = c1.selectbox("From", codes, index=codes.index("JFK") if "JFK" in codes else 0, format_func=airport_label.get)
dest = c2.selectbox("To", codes, index=codes.index("LAX") if "LAX" in codes else 1, format_func=airport_label.get)

route = f"{origin}_{dest}"
known = route in routes.index
default_dist = float(routes.loc[route, "distance"]) if known else 800.0
default_dur = float(routes.loc[route, "duration"]) if known else 120.0

t1, t2 = st.columns(2)
dep_time = t1.time_input("Scheduled departure (local)", dt.time(14, 30))
default_arr = (dt.datetime.combine(date, dep_time) + dt.timedelta(minutes=default_dur)).time().replace(second=0)
arr_time = t2.time_input("Scheduled arrival (local) — check your ticket", default_arr)

d1, d2 = st.columns(2)
distance = d1.number_input("Distance (miles)", 1.0, 6000.0, default_dist, help="Auto-filled from past flights on this route; edit if needed.")
duration = d2.number_input("Scheduled duration (min)", 1.0, 900.0, default_dur)
if not known:
    st.info("No history for this route in the training data, so the route's own delay rate is unknown. Please check distance and duration.")

dep_delay = None
if mode == "B_at_gate":
    dep_delay = st.number_input("Departure delay (minutes; negative = left early)", -60, 600, 15)

if origin == dest:
    st.warning("Origin and destination are the same — please pick two different airports.")
elif st.button("Predict", type="primary", use_container_width=True):
    p, delayed, thr = predict(
        bundle, mode, airline_code=airline_names[airline], origin=origin, dest=dest, date=date,
        dep_time=dep_time, arr_time=arr_time, distance=distance, duration=duration, dep_delay=dep_delay,
    )
    st.metric("Probability of arriving >15 min late", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    (st.error if delayed else st.success)("⚠️ Likely DELAYED" if delayed else "✅ Likely ON TIME")
    if known:
        st.write(f"Historically, **{routes.loc[route, 'delay_rate']:.0%}** of {origin}→{dest} flights in the data arrived late "
                 f"({int(routes.loc[route, 'n'])} flights).")
    if mode == "A_pre_departure":
        st.caption("Pre-departure predictions are only moderately reliable (ROC-AUC ≈ 0.65): the dataset has no weather or "
                   "live traffic information. Once the flight has left, use the other mode (ROC-AUC ≈ 0.93).")
    else:
        st.caption("At-gate model: ROC-AUC ≈ 0.93, precision ≈ 0.89, recall ≈ 0.74 on newer, unseen flights.")

with st.expander("About this model"):
    st.markdown(
        "- **Algorithm:** Gradient Boosting (compared with Random Forest)\n"
        "- **Target:** arrival delay > 15 minutes\n"
        "- **Features:** month, weekday, scheduled departure/arrival hour, distance, duration, airline, and historical delay rates "
        "of the origin, destination, airline and route\n"
        "- **Validation:** time-based split (trained on older flights, tested on newer ones)\n"
        "- **Limitations:** no weather data; the share of delayed flights changed over time, so probabilities are approximate."
    )
