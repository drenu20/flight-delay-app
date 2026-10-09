# ✈️ Flight Delay Predictor (Streamlit)

Predicts whether a US domestic flight will arrive more than 15 minutes late.

* **Before departure** model — ROC-AUC ≈ 0.65 (no weather in the dataset)
* **At the gate** model (uses departure delay) — ROC-AUC ≈ 0.93, precision ≈ 0.89, recall ≈ 0.74
* Gradient Boosting, trained with a time-based split on 100k flights (2019–2023)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repo and upload everything in this folder (keep the `model/` and `data/` folders).
2. Go to https://share.streamlit.io → sign in with GitHub → **Create app**.
3. Choose the repo, branch `main`, main file `app.py` → **Deploy**.

> `requirements.txt` pins `scikit-learn==1.6.1` because the model was trained with that version. Don't change it unless you retrain.
