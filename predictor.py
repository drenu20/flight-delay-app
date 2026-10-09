"""Prediction logic (kept separate from the UI so it can be tested on its own)."""
import joblib
import pandas as pd

BUNDLE_PATH = "model/flight_delay_bundle.joblib"


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def build_features(bundle, *, airline_code, origin, dest, date, dep_time, arr_time,
                   distance, duration, dep_delay=None):
    """Create one row of model features, in the same way as the training notebook."""
    meta = bundle["meta"]
    g, maps = meta["global_rate"], meta["te_maps"]
    route = f"{origin}_{dest}"
    row = {
        "month": date.month,
        "dow": date.weekday(),                      # Monday = 0, like pandas dt.dayofweek
        "dep_hour": dep_time.hour,
        "arr_hour": arr_time.hour,
        "CRS_ELAPSED_TIME": float(duration),
        "DISTANCE": float(distance),
        "airline_id": float(meta["airline_map"].get(airline_code, float("nan"))),
        "rate_origin": maps["ORIGIN"].get(origin, g),
        "rate_dest": maps["DEST"].get(dest, g),
        "rate_airline": maps["AIRLINE_CODE"].get(airline_code, g),
        "rate_route": maps["route"].get(route, g),
    }
    if dep_delay is not None:
        row["DEP_DELAY"] = float(dep_delay)
    return row


def predict(bundle, mode, **inputs):
    """mode: 'A_pre_departure' or 'B_at_gate'. Returns (probability, is_delayed, threshold)."""
    spec = bundle[mode]
    dep_delay = inputs.pop("dep_delay", None)
    if mode != "B_at_gate":
        dep_delay = None                      # pre-departure model never sees departure delay
    row = build_features(bundle, dep_delay=dep_delay, **inputs)
    X = pd.DataFrame([row])[spec["features"]]
    if spec["fill_na"] is not None:
        X = X.fillna(spec["fill_na"])
    p = float(spec["model"].predict_proba(X)[0, 1])
    return p, p >= spec["threshold"], spec["threshold"]
