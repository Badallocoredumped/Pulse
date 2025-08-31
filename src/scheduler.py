import time
import subprocess
from datetime import datetime
import schedule
import logging

# Configure logging to log to both file and console
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    handlers=[
        logging.FileHandler("scheduler.log"),  # Log to file
        logging.StreamHandler()  # Log to console (stdout)
    ]
)

def run_forecast_pipeline():
    """
    Function to run the forecast pipeline script.
    """
    logging.info("⏰ Running forecast pipeline...")
    try:
        # Run the forecast_pipeline.py script


        subprocess.run(["python", "/app/src/forecast_pipeline.py"], check=True)
        logging.info("✅ Forecast pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error while running forecast pipeline: {e}")


""" logging.info("🚀 Performing an initial run of the forecast pipeline...")
run_forecast_pipeline() """

# Schedule the pipeline to run every hour at the 15th minute
schedule.every().hour.at(":15").do(run_forecast_pipeline)
#schedule.every(1).minutes.do(run_forecast_pipeline)


logging.info("🔄 Scheduler started. Waiting for the 15th minute of every hour...")

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(1)