import numpy as np
from sklearn.preprocessing import MinMaxScaler

class EnergyDataProcessor:
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df):
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Scale the consumption data
        consumption = df['consumption_mwh'].values.reshape(-1, 1)
        consumption_scaled = self.scaler.fit_transform(consumption)
        
        # Create sequences (past 24 hours to predict next hour)
        X, y = [], []
        for i in range(self.sequence_length, len(consumption_scaled)):
            X.append(consumption_scaled[i-self.sequence_length:i, 0])
            y.append(consumption_scaled[i, 0])
        
        return np.array(X), np.array(y)