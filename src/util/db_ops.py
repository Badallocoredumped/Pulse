import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class MLDatabaseOps:
    """Reusable database operations for ML pipeline."""

    def __init__(self):
        self.conn = None

    def connect(self):
        """Connect to PostgreSQL using environment variables."""
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST_DEV" , "POSTGRES_HOST_DEV"),
            database=os.getenv("POSTGRES_DB_DEV" , "POSTGRES_DB_DEV"),
            user=os.getenv("POSTGRES_USER_DEV" , "POSTGRES_USER_DEV"),
            password=os.getenv("POSTGRES_PASSWORD_DEV" , "POSTGRES_PASSWORD_DEV"),
            port=os.getenv("POSTGRES_PORT" , "POSTGRES_PORT_DEV")
        )

    def disconnect(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def fetch_last_24_hours(self):
        """Fetch last 24 hours from epias_power_consumption."""
        query = """
            SELECT datetime, consumption_mwh
            FROM epias_power_consumption
            ORDER BY datetime DESC
            LIMIT 24
        """
        df = pd.read_sql(query, self.conn)
        df = df.sort_values('datetime')
        return df

    def store_forecast(self, forecast, forecast_datetime, model_version="v1", confidence_score=None, horizon=1):
        """Insert forecast into predictions table."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (
                datetime, predicted_consumption_mwh, model_version, prediction_generated_at, confidence_score, prediction_horizon_hours
            ) VALUES (%s, %s, %s, NOW(), %s, %s)
            """,
            (
                forecast_datetime,
                float(forecast),  # Convert numpy.float32 to Python float
                model_version,
                confidence_score,
                horizon
            )
        )
        self.conn.commit()
        cur.close()

    def fetch_consumption(self, start_datetime=None, end_datetime=None):
        """
        Fetch consumption data from epias_power_consumption table.
        :param start_datetime: Start of the time range.
        :param end_datetime: End of the time range.
        :return: DataFrame with consumption data.
        """
        query = "SELECT datetime, consumption_mwh FROM epias_power_consumption"
        params = []
        if start_datetime and end_datetime:
            query += " WHERE datetime BETWEEN %s AND %s"
            params = [start_datetime, end_datetime]
        df = pd.read_sql(query, self.conn, params=params)
        return df

    def fetch_predictions(self, start_datetime=None, end_datetime=None):
        """
        Fetch predictions from the predictions table.
        :param start_datetime: Start of the time range.
        :param end_datetime: End of the time range.
        :return: DataFrame with predictions.
        """
        query = "SELECT datetime, predicted_consumption_mwh FROM predictions"
        params = []
        if start_datetime and end_datetime:
            query += " WHERE datetime BETWEEN %s AND %s"
            params = [start_datetime, end_datetime]
        df = pd.read_sql(query, self.conn, params=params)
        return df

    def store_evaluation_metrics(self, metrics, model_version, start_datetime, end_datetime):
        """
        Store evaluation metrics in the evaluation_metrics table.
        :param metrics: Dictionary of calculated metrics.
        :param model_version: Model version being evaluated.
        :param start_datetime: Start of the evaluation period.
        :param end_datetime: End of the evaluation period.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO evaluation_metrics (
                evaluation_date, model_version, mae, rmse, mape, r2_score, data_points_count, 
                evaluation_period_start, evaluation_period_end, created_at
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """,
            (
                model_version,
                metrics["MAE"],
                metrics["RMSE"],
                metrics["MAPE"],
                metrics.get("R2", None),  # Optional R2 score
                len(metrics["actuals"]),  # Number of data points
                start_datetime,
                end_datetime
            )
        )
        self.conn.commit()
        cur.close()