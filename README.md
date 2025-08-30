# ⚡️ Pulse  

**Pulse** is a machine learning pipeline that forecasts **hourly energy consumption** using a custom-trained LSTM model.  
It stores results in PostgreSQL, runs automatically on a schedule, and visualizes predictions in Grafana dashboards.  

---



```mermaid
flowchart TB
    subgraph AUTOMATION["🤖 Automation Layer"]
        SCHEDULER[🕐 Python Schedule<br/><br/>• Runs every hour at :07<br/>• Background execution<br/>• Error handling & retry<br/>• Continuous monitoring]
        style SCHEDULER fill:#fff3e0,stroke:#f57c00,stroke-width:3px,color:#f57c00
    end
    
    subgraph INPUT["📊 Data Source"]
        direction TB
        DB_IN[🐘 PostgreSQL<br/><br/>⚡ Historical Energy Data<br/>📈 Last 24 hours consumption<br/>🔄 Real-time updates]
        style DB_IN fill:#e1f5fe,stroke:#01579b,stroke-width:3px,color:#01579b
    end
    
    subgraph PROCESSING["⚙️ ML Processing Pipeline"]
        direction TB
        FETCH[📡 Data Fetcher<br/><br/>• Query last 24h data<br/>• Data validation<br/>• Error handling]
        PREPROCESS[🔄 Preprocessor<br/><br/>• Apply saved scaler<br/>• Create input tensors<br/>• Feature engineering]
        PREDICT[🧠 LSTM Model<br/><br/>• Load trained checkpoint<br/>• Generate prediction<br/>• Next-hour forecast]
        
        FETCH --> PREPROCESS
        PREPROCESS --> PREDICT
        
        style FETCH fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
        style PREPROCESS fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
        style PREDICT fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    end
    
    subgraph STORAGE["💾 Storage Layer"]
        direction LR
        PREDICTIONS[📈 Predictions Table<br/><br/>• Forecasted values<br/>• Timestamps<br/>• Model metadata]
        METRICS[📊 Metrics Table<br/><br/>• MAE, RMSE, MAPE<br/>• Performance tracking<br/>• Error analysis]
        
        PREDICTIONS -.->|"📊 Evaluation<br/>When actuals arrive"| METRICS
        
        style PREDICTIONS fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#2e7d32
        style METRICS fill:#e3f2fd,stroke:#1565c0,stroke-width:3px,color:#1565c0
    end
    
    subgraph OUTPUT["📈 Monitoring Layer"]
        direction TB
        GRAFANA[📊 Grafana Dashboards<br/><br/>• Predictions vs Actuals<br/>• Rolling error metrics<br/>• Model performance<br/>• System health KPIs]
        
        style GRAFANA fill:#fff8e1,stroke:#f57f17,stroke-width:3px,color:#f57f17
    end
    
    %% Automation connections
    SCHEDULER ==>|"🕐 Trigger every hour"| FETCH
    SCHEDULER -.->|"🐍 Pipeline orchestration"| PROCESSING
    
    %% Main flow connections
    DB_IN ==>|"📊 Query data<br/>Last 24h consumption"| FETCH
    PREDICT ==>|"💾 Store predictions"| PREDICTIONS
    
    %% Output connections
    PREDICTIONS ==>|"📈 Visualization queries"| GRAFANA
    METRICS ==>|"📊 Performance metrics"| GRAFANA
    
    %% Styling for subgraphs
    style AUTOMATION fill:#fef7e0,stroke:#f57c00,stroke-width:4px,color:#ef6c00
    style INPUT fill:#e8eaf6,stroke:#3f51b5,stroke-width:4px,color:#1a237e
    style PROCESSING fill:#f3e5f5,stroke:#7b1fa2,stroke-width:4px,color:#4a148c
    style STORAGE fill:#e0f2f1,stroke:#00695c,stroke-width:4px,color:#004d40
    style OUTPUT fill:#fff3e0,stroke:#ef6c00,stroke-width:4px,color:#e65100
    
    %% Custom connection styling
    linkStyle 0 stroke:#f57c00,stroke-width:4px
    linkStyle 1 stroke:#f57c00,stroke-width:2px,stroke-dasharray: 5 5
    linkStyle 2 stroke:#1976d2,stroke-width:4px
    linkStyle 3 stroke:#2e7d32,stroke-width:3px
    linkStyle 4 stroke:#f57f17,stroke-width:3px
    linkStyle 5 stroke:#1565c0,stroke-width:3px
    linkStyle 6 stroke:#d32f2f,stroke-width:2px,stroke-dasharray: 5 5
```

## ✨ Features  
- 🧠 **Custom LSTM model** trained on ~4 years of historical data (not a premade model)  
- ⏳ **Forecasts**: Uses the last 24 hours to predict the next 1 hour  
- 🔄 **Automated pipeline**: Runs every hour (at minute 7)  
- 🗄️ **PostgreSQL integration**: Saves predictions, actuals, and error metrics (MAE, RMSE, MAPE)  
- 📊 **Visualization**: Grafana dashboards to monitor predictions and accuracy  

---

## ⚙️ How it works  
1. Fetch latest data from the database  
2. Preprocess with the same scaler used during training  
3. Load the saved LSTM checkpoint (`best_energy_model.pth`)  
4. Predict next-hour consumption  
5. Store results and metrics in PostgreSQL  
6. Visualize everything in Grafana  

---

## 🏋️ Model Training  
The LSTM model was trained **offline** on ~4 years of energy consumption data.  

- Framework: **PyTorch**  
- Input: 24h sliding window  
- Output: Next 1h forecast  
- Best checkpoint: `best_energy_model.pth`  

Training scripts and configs are included in the repo for reproducibility.  

---

## 📊 Dashboards  
Pulse provides **Grafana-powered dashboards**:  
- Predictions vs. actuals  
- Rolling error metrics (MAE, MAPE)  

---

## 🏗️ Architecture  

1. **Energy Forecasting Pipeline**  
   - **ETL Workflow**: Extract → Preprocess (scaling & sliding windows) → Forecast → Store  
   - **Automated Scheduling**: Hourly execution at minute 7 with error handling  
   - **PostgreSQL Integration**: Predictions + actuals + evaluation metrics stored for analysis  

2. **Model Serving & Inference**  
   - **Pre-Trained LSTM Model**: Custom-trained on ~4 years of historical data  
   - **Inference-Only Runtime**: Model checkpoint (`best_energy_model.pth`) loaded for forecasts  
   - **Evaluation Layer**: MAE, RMSE, MAPE calculated when actuals arrive  

3. **Production-Ready Infrastructure**  
   - **Dockerized Services**: Database, pipeline, and Grafana dashboards packaged as containers  
   - **Persistent Volumes**: Durable storage for models, scaler, and database data  
   - **Health & Monitoring**: Logs, metrics, and dashboards for system reliability  

4. **Scalable & Maintainable Design**  
   - **Separation of Concerns**: Independent modules for training, inference, database I/O, and scheduling  
   - **Database Connection Pooling**: Efficient queries and inserts  
   - **Reusable Components**: Modular codebase with configs for easy retraining and redeployment  

---

## 🛠️ Technology Stack  

### Core Technologies  
- **Python 3.11** – Modern runtime for model training and inference  
- **PostgreSQL 15** – Relational database for predictions, actuals, and metrics  

### Machine Learning & Data Processing  
- **PyTorch** – Deep learning framework for the LSTM model  
- **pandas** – Data manipulation and preprocessing  
- **scikit-learn** – MinMaxScaler for normalization  

### Automation & Scheduling  
- **schedule** – Lightweight job scheduling for hourly pipeline runs  
- **cron (via scheduler.py)** – Automated pipeline execution  

### Monitoring & Visualization  
- **Grafana** – Dashboards for predictions and evaluation metrics  
- **Python logging** – Application-level logging  
- **traceback** – Error tracking and debugging  

### DevOps & Deployment  
- **Docker** – Containerization for portability  
- **Docker Compose** – Multi-service orchestration (pipeline + database + Grafana)

- ## Performance Overview

### Lab (Historical Test)
- **MAE** ≈ 466 MWh  
- **RMSE** ≈ 704 MWh  
- **MAPE** ≈ 1.14% — strong performance on in-distribution data.
<p align="center">
  <img src="/docs/output.png" width="900">
</p>
<p align="center">
  <img src="/docs/3.png" width="400">
</p>
  

### Live Production (Aug 29–30 Run)
- MAPE stayed low (~5%) initially but **spiked above 20%** around Aug 30, before settling to elevated levels (~5–15%).
- This reflects **real-world anomalies**—such as abrupt demand drops or unexpected events—not represented in training data.
  
  
## Grafana Dashboards
<p align="center">
  <img src="/docs/1.png" width="900">
</p>
<p align="center">
  <img src="/docs/2.png" width="900">
</p>

