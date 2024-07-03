import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def preprocess_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= '2008-01-01') & (data['date'] <= '2021-12-31')]
    data = data.sort_values(by='date')
    return data

def train_model(data):
    data = preprocess_data(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['close']])

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 60
    X, y = create_sequences(data_scaled, seq_length)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001

    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, scaler

def predict_future(model, scaler, data, future_steps=30):
    data = preprocess_data(data)
    model.eval()
    data_scaled = scaler.transform(data[['close']])

    inputs = torch.tensor(data_scaled[-60:], dtype=torch.float32).unsqueeze(0)
    predictions = []
    for _ in range(future_steps):
        with torch.no_grad():
            pred = model(inputs)
            predictions.append(pred.item())
            inputs = torch.cat((inputs[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)), dim=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    df = pd.read_json(json.dumps(data))
    global model, scaler
    model, scaler = train_model(df)
    return "Model trained successfully"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.read_json(json.dumps(data))
    predictions = predict_future(model, scaler, df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    if os.getenv('FLASK_ENV') == 'development':
        app.run(debug=True, port=5001)
    else:
        app.run(port=5001)
