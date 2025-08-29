import torch
import numpy as np

def load_model(model_class, model_path, device):
    """Load a trained PyTorch model from disk."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_next_hour(model, last_24_hours, scaler, device):
    """
    Predict next hour consumption given last 24 hours.
    last_24_hours: np.array of shape (24,)
    scaler: fitted sklearn MinMaxScaler
    """
    model.eval()
    scaled_input = scaler.transform(last_24_hours.reshape(-1, 1)).flatten()
    input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1))[0, 0]
    return prediction