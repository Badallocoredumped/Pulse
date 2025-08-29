import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import pandas as pd
from datetime import datetime
from util.model_ops import load_model, predict_next_hour
from model.model_definitions import EnergyLSTM
from model.data_processor import EnergyDataProcessor
from util.db_ops import MLDatabaseOps
from util.evaluation import ForecastEvaluator
import joblib
from pathlib import Path

if __name__ == "__main__":

    processor = EnergyDataProcessor()
    models_dir = Path(__file__).resolve().parents[1] / "models"
    scaler_path = models_dir / "scaler.joblib"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting the forecast pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Initialize database connection
    print("📡 Connecting to the database...")
    db = MLDatabaseOps()
    db.connect()
    print("✅ Database connection established.")

    if scaler_path.exists():
        processor.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded fitted scaler from {scaler_path}")
    else:
        # Fit once on historical data and save
        print("📊 Fetching historical data to fit the scaler...")
        history_df = db.fetch_consumption()
        print(f"✅ Retrieved {len(history_df)} rows of historical data.")
        processor.scaler.fit(history_df['consumption_mwh'].values.reshape(-1, 1))
        joblib.dump(processor.scaler, scaler_path)
        print(f"✅ Fitted scaler on historical data and saved to {scaler_path}.")
    #print(history_df)



    # Fetch the last 24 hours for prediction
    print("📊 Fetching the last 24 hours of data for prediction...")
    df = db.fetch_last_24_hours()
    print(f"✅ Retrieved {len(df)} rows of recent data.")
    print(df)


    

    # Load the trained model
    print("📦 Loading the trained model...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_energy_model.pth")
    model = load_model(EnergyLSTM, model_path, device)
    print("✅ Model loaded successfully.")

    """ # Generate forecast
    print("🔮 Generating forecast for the next hour...")
    forecast = predict_next_hour(model, df['consumption_mwh'].values[-18:], processor.scaler, device)
    print(f"✅ Forecast generated: {forecast:.2f} MWh") """

    # Forecast for the next hour
    """     next_hour = df['datetime'].max() + pd.Timedelta(hours=1)
        print(f"🕒 Forecasting for the next hour: {next_hour}")
        db.store_forecast(forecast, next_hour, model_version="v1", confidence_score=None, horizon=1)
        print("✅ Forecast stored in the database.") """

    # Evaluate forecasts
    print("🔍 Starting evaluation...")
    evaluator = ForecastEvaluator(db)
    metrics = evaluator.evaluate_and_store(model_version="v1")
    if metrics:
        print("📊 Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    # Disconnect from the database
    db.disconnect()
    print("🔌 Disconnected from the database.")
    print("🎉 Forecast pipeline completed successfully!")