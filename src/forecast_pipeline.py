import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from pathlib import Path
import joblib

from util.model_ops import load_model, predict_next_hour
from model.model_definitions import EnergyLSTM
from model.data_processor import EnergyDataProcessor
from util.db_ops import MLDatabaseOps
from util.evaluation import ForecastEvaluator

if __name__ == "__main__":

    print("🚀 Starting the forecast pipeline...")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # --- Processor & Scaler ---
    processor = EnergyDataProcessor(sequence_length=24)
    models_dir = Path(__file__).resolve().parents[1] / "models"
    scaler_path = models_dir / "scaler.joblib"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Database ---
    print("📡 Connecting to the database...")
    db = MLDatabaseOps()
    db.connect()
    print("✅ Database connection established.")

    # --- Load or fit scaler ---
    if scaler_path.exists():
        processor.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded fitted scaler from {scaler_path}")
    else:
        print("📊 Fetching historical data to fit the scaler...")
        history_df = db.fetch_consumption()
        print(f"✅ Retrieved {len(history_df)} rows of historical data.")
        processor.scaler.fit(history_df['consumption_mwh'].values.reshape(-1, 1))
        joblib.dump(processor.scaler, scaler_path)
        print(f"✅ Fitted scaler on historical data and saved to {scaler_path}.")

    # --- Load model ---
    print("📦 Loading the trained model...")
    model_path = models_dir / "best_energy_model.pth"
    model = load_model(EnergyLSTM, model_path, device)
    print("✅ Model loaded successfully.")

    # --- Forecast next hour ---
    print("📊 Fetching the last 24 hours of data for prediction...")
    df = db.fetch_last_24_hours()
    print(f"✅ Retrieved {len(df)} rows of recent data.")

    if len(df) >= processor.sequence_length:
        print("🔮 Generating forecast for the next hour...")
        last_seq = df['consumption_mwh'].values[-processor.sequence_length:]
        forecast = predict_next_hour(model, last_seq, processor.scaler, device)
        print(f"✅ Forecast generated: {forecast:.2f} MWh")

        # Store forecast
        next_hour = df['datetime'].max() + pd.Timedelta(hours=1)
        db.store_forecast(
            forecast, next_hour, model_version="v1", confidence_score=None, horizon=1
        )
        print(f"🕒 Forecast for {next_hour} stored in DB.")
    else:
        print("⚠️ Not enough data for a 24-hour sequence, skipping forecast.")





    # --- Evaluate model ---
    print("🔍 Starting evaluation...")

    db.upsert_pending_point_eval(next_hour, model_version="v1", yhat=forecast)
    print("📝 Pending evaluation placeholder inserted.")


    db.score_any_ready_hours(model_version="v1", horizon_hours=1, watermark_minutes=30)

    """ evaluator = ForecastEvaluator(db)
    metrics = evaluator.evaluate_and_store(model_version="v1")
    if metrics:
        print("📊 Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}") """

    # --- Disconnect ---
    db.disconnect()
    print("🔌 Disconnected from the database.")
    print("🎉 Forecast pipeline completed successfully!")
