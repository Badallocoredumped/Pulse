import os
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class MLDatabaseOps:
    """Reusable database operations for ML pipeline."""

    def __init__(self):
        self.conn = None

    def connect(self):
        """Connect to PostgreSQL using environment variables."""
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST" , "POSTGRES_HOST_DEV"),
            database=os.getenv("POSTGRES_DB" , "POSTGRES_DB_DEV"),
            user=os.getenv("POSTGRES_USER" , "POSTGRES_USER_DEV"),
            password=os.getenv("POSTGRES_PASSWORD" , "POSTGRES_PASSWORD_DEV"),
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
            ON CONFLICT (datetime, model_version, prediction_horizon_hours)
            DO UPDATE SET
                predicted_consumption_mwh = EXCLUDED.predicted_consumption_mwh,
                confidence_score = EXCLUDED.confidence_score,
                prediction_generated_at = NOW() 
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
        # Convert NumPy types to native Python types
        metrics = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in metrics.items()}

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
                metrics["data_points_count"],
                start_datetime,
                end_datetime
            )
        )
        self.conn.commit()
        cur.close()


    def upsert_pending_point_eval(self, target_ts, model_version, yhat):
        sql = """
        INSERT INTO evaluation_metrics (
            evaluation_date, model_version, mae, rmse, mape, r2_score,
            data_points_count, evaluation_period_start, evaluation_period_end
        )
        VALUES (%s, %s, NULL, NULL, NULL, NULL, NULL, %s, %s)
        ON CONFLICT (evaluation_date, model_version) DO NOTHING;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (target_ts, model_version, target_ts, target_ts))
            self.conn.commit()


    def score_any_ready_hours(self, model_version: Optional[str] = None,
                              horizon_hours: Optional[int] = 1,
                              watermark_minutes: int = 30):
        """
        Fill per-hour metrics in evaluation_metrics for hours where we have truth.
        - evaluation_metrics must have UNIQUE(evaluation_date, model_version)
        - predictions join key: predictions.datetime == epias_power_consumption.datetime
        """

        sql = """
        WITH ready AS (
        SELECT
            p.datetime                         AS target_ts,
            p.model_version                    AS model_version,
            p.prediction_horizon_hours         AS horizon,
            p.predicted_consumption_mwh::numeric AS yhat,
            a.consumption_mwh::numeric         AS y
        FROM predictions p
        JOIN epias_power_consumption a
            ON a.datetime = p.datetime
        WHERE (%(model_version)s IS NULL OR p.model_version = %(model_version)s)
            AND (%(horizon)s IS NULL OR p.prediction_horizon_hours = %(horizon)s)
            AND a.datetime <= now() - (%(watermark)s || ' minutes')::interval
        )
        INSERT INTO evaluation_metrics (
            evaluation_date, model_version, mae, rmse, mape, r2_score,
            data_points_count, evaluation_period_start, evaluation_period_end
        )
        SELECT
            r.target_ts                              AS evaluation_date,
            r.model_version                          AS model_version,
            ABS(r.y - r.yhat)                        AS mae,
            ABS(r.y - r.yhat)                        AS rmse,  -- single-point RMSE
            CASE
            WHEN ABS(r.y) > 0.000001::numeric
            THEN ABS(r.y - r.yhat) / ABS(r.y)
            ELSE NULL
            END                                      AS mape,
            NULL                                     AS r2_score,
            1                                        AS data_points_count,
            r.target_ts                              AS evaluation_period_start,
            r.target_ts                              AS evaluation_period_end
        FROM ready r
        ON CONFLICT (evaluation_date, model_version)
        DO UPDATE SET
            mae = EXCLUDED.mae,
            rmse = EXCLUDED.rmse,
            mape = EXCLUDED.mape,
            r2_score = NULL,
            data_points_count = 1,
            evaluation_period_start = EXCLUDED.evaluation_period_start,
            evaluation_period_end   = EXCLUDED.evaluation_period_end;
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, {
                "model_version": model_version,
                "horizon": horizon_hours,
                "watermark": str(watermark_minutes),
            })
        self.conn.commit()
