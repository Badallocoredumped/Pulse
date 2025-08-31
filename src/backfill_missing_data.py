import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from util.model_ops import load_model, predict_next_hour
from model.model_definitions import EnergyLSTM
from model.data_processor import EnergyDataProcessor
from util.db_ops import MLDatabaseOps

MODEL_VERSION = "v1"           
HORIZON_HOURS = 1
SEQ_LEN = 24

def main():
    print("🧹 Historical backfill: forecasts + per-hour evals")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # --- Paths / Scaler ---
    processor = EnergyDataProcessor(sequence_length=SEQ_LEN)
    models_dir = Path(__file__).resolve().parents[1] / "models"
    scaler_path = models_dir / "scaler.joblib"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- DB ---
    db = MLDatabaseOps()
    db.connect()
    print("✅ DB connected")

    # --- Load full history ---
    
    hist = db.fetch_consumption()
    if hist is None or hist.empty:
        print("❌ No historical data found, aborting.")
        return

    hist = hist.sort_values("datetime").reset_index(drop=True)
    print(f"📚 History rows: {len(hist)} (from {hist['datetime'].iloc[0]} to {hist['datetime'].iloc[-1]})")

    # --- Scaler ---
    if scaler_path.exists():
        processor.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler: {scaler_path}")
    else:
        # Fit scaler on ALL historical consumption
        processor.scaler.fit(hist["consumption_mwh"].values.reshape(-1, 1))
        joblib.dump(processor.scaler, scaler_path)
        print(f"✅ Fitted + saved scaler at {scaler_path}")

    # --- Model ---
    model_path = models_dir / "best_energy_model.pth"
    model = load_model(EnergyLSTM, model_path, device)
    print("✅ Model loaded")

    # --- Iterate through hours and predict T from [T-24h, T-1h] ---
    vals = hist["consumption_mwh"].astype(float).values
    times = hist["datetime"].values  

    made = 0
    skipped = 0

    for i in range(SEQ_LEN, len(hist)):
        T = pd.Timestamp(times[i])  
        window = vals[i-SEQ_LEN:i]  # strictly the previous 24 hours

        # Basic quality check on the window
        if np.any(pd.isna(window)) or len(window) < SEQ_LEN:
            skipped += 1
            continue

        # Forecast next hour (which is the current row's timestamp T)
        yhat = float(predict_next_hour(model, window, processor.scaler, device))

        # Write/Upsert prediction for exactly T
        db.store_forecast(
            forecast=yhat,
            forecast_datetime=T,                    
            model_version=MODEL_VERSION,
            confidence_score=None,
            horizon=HORIZON_HOURS
        )
        made += 1

        # --- Tiny progress print ---
        if made % 500 == 0:
            print(f"… wrote {made} predictions so far (up to {T})")

    print(f"✅ Backfill predictions done. Wrote/updated: {made}, skipped windows: {skipped}")

    # --- Score everything that's ready (historical truth exists) ---
    # watermark_minutes=0 so we compute immediately for all matched timestamps
    db.score_any_ready_hours(model_version=MODEL_VERSION,
                             horizon_hours=HORIZON_HOURS,
                             watermark_minutes=0)
    print("✅ Filled evaluation_metrics for all matching hours (MAE/RMSE/MAPE per point).")

    db.disconnect()
    print("🔌 DB disconnected. Done!")

if __name__ == "__main__":
    main()
