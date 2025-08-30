import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ForecastEvaluator:
    def __init__(self, db):
        """
        Initialize the evaluator with a database connection.
        :param db: Instance of MLDatabaseOps for fetching/storing data.
        """
        self.db = db

    def calculate_metrics(self, actuals, predictions):
        """
        Calculate evaluation metrics: MAE, RMSE, Mean Consumption, MAPE.
        :param actuals: Array of actual values.
        :param predictions: Array of predicted values.
        :return: Dictionary of metrics.
        """
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mean_consumption = np.mean(actuals)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        return {
            "MAE": mae,
            "RMSE": rmse,
            "Mean Consumption": mean_consumption,
            "MAPE": mape
        }

    def evaluate_and_store(self, model_version="v1"):
        """
        Dynamically fetch the latest prediction and evaluate it against actuals.
        :param model_version: Model version being evaluated.
        """
        # Fetch the latest prediction from the database
        print("📊 Fetching the latest prediction for evaluation...")
        query = """
            SELECT datetime, prediction_horizon_hours
            FROM predictions
            WHERE model_version = %s
            ORDER BY prediction_generated_at DESC
            LIMIT 1
        """
        cur = self.db.conn.cursor()
        cur.execute(query, (model_version,))
        latest_prediction = cur.fetchone()
        cur.close()

        if not latest_prediction:
            print("⚠️ No predictions found for evaluation.")
            return None

        forecast_datetime, horizon = latest_prediction
        start_datetime = forecast_datetime - pd.Timedelta(hours=horizon)
        end_datetime = forecast_datetime

        print(f"📊 Evaluating prediction for {forecast_datetime} (horizon: {horizon} hours)...")
        print(f"📊 Evaluation period: {start_datetime} to {end_datetime}")

        # Fetch actuals and predictions for the evaluation period
        actuals_df = self.db.fetch_consumption(start_datetime, end_datetime)
        predictions_df = self.db.fetch_predictions(start_datetime, end_datetime)

        # Ensure alignment of actuals and predictions
        merged_df = actuals_df.merge(predictions_df, on="datetime", suffixes=("_actual", "_predicted"))
        if merged_df.empty:
            print("⚠️ No matching data found between actuals and predictions. Evaluation skipped.")
            return None

        actuals = merged_df["consumption_mwh"].values
        predictions = merged_df["predicted_consumption_mwh"].values

        # Check if actuals or predictions are empty
        if len(actuals) == 0 or len(predictions) == 0:
            print("⚠️ Actuals or predictions are empty. Evaluation skipped.")
            return None

        # Calculate metrics
        print("📈 Calculating evaluation metrics...")
        metrics = self.calculate_metrics(actuals, predictions)
        metrics["data_points_count"] = len(actuals)  # Add data points count

        # Store metrics in the database
        print("💾 Storing evaluation metrics in the database...")
        self.db.store_evaluation_metrics(metrics, model_version, start_datetime, end_datetime)

        print("✅ Evaluation completed and metrics stored.")
        return metrics