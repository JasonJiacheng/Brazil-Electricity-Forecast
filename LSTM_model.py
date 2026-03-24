import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# -------------------------
# Load and clean
# -------------------------
def load_and_process():
    df = pd.read_csv('energy_demand_hourly_brazil.csv')
    df['index'] = pd.to_datetime(df['index'])
    df = df.sort_values('index')
    df = df[df['index'].dt.year >= 2009].reset_index(drop=True)

    weather = pd.read_csv('open-meteo-10.02S55.01W420m.csv', skiprows=3)
    weather = weather.rename(columns={
        'time': 'index',
        'temperature_2m (°C)': 'temperature',
        'relative_humidity_2m (%)': 'humidity',
        'wind_speed_10m (km/h)': 'wind_speed',
        'apparent_temperature (°C)': 'feels_like'
    })
    weather['index'] = pd.to_datetime(weather['index'])
    df = df.merge(weather, on='index', how='left')

    df['hourly_demand'] = df['hourly_demand'].replace(0, np.nan)
    df['hourly_demand'] = df['hourly_demand'].interpolate(method='linear')

    df['hour'] = df['index'].dt.hour
    df['day_of_week'] = df['index'].dt.dayofweek
    df['month'] = df['index'].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "hourly_demand",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "temperature", "humidity",
        "wind_speed", "feels_like"
    ]

    return df, feature_cols

# -------------------------
# Train / Val / Test split
# -------------------------
def split_data(df):
    n = len(df)

    train_end = int(n * 0.7)
    validation_end = int(n * 0.8)

    train_data = df.iloc[:train_end]
    validation_data = df.iloc[train_end:validation_end]
    test_data = df.iloc[validation_end:]

    return train_data, validation_data, test_data, validation_end

# -------------------------
# Scaling
# -------------------------
def scale_data(feature_cols, train_data, validation_data, test_data):
    demand_scaler = StandardScaler()
    feature_scaler = StandardScaler()

    train_demand = demand_scaler.fit_transform(train_data[['hourly_demand']])
    validation_demand = demand_scaler.transform(validation_data[['hourly_demand']])
    test_demand = demand_scaler.transform(test_data[['hourly_demand']])

    other_cols = feature_cols[1:]

    train_feat = feature_scaler.fit_transform(train_data[other_cols])
    validation_feat = feature_scaler.transform(validation_data[other_cols])
    test_feat = feature_scaler.transform(test_data[other_cols])

    train_scaled = np.hstack([train_demand, train_feat])
    validation_scaled = np.hstack([validation_demand, validation_feat])
    test_scaled = np.hstack([test_demand, test_feat])

    return train_scaled, validation_scaled, test_scaled, demand_scaler

# -------------------------
# Sequence builder
# -------------------------
def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def build_sequence(train_scaled, validation_scaled, test_scaled):
    seq_len = 24 * 7

    X_train, y_train = make_sequences(train_scaled, seq_len)

    validation_with_context = np.vstack([train_scaled[-seq_len:], validation_scaled])
    X_validation, y_validation = make_sequences(validation_with_context, seq_len)

    test_with_context = np.vstack([validation_scaled[-seq_len:], test_scaled])
    X_test, y_test = make_sequences(test_with_context, seq_len)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

# -------------------------
# DataLoaders
# -------------------------
def data_loaders(X_train, y_train, X_validation, y_validation, X_test, y_test):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=128, shuffle=True
    )

    validation_loader = DataLoader(
        TensorDataset(torch.tensor(X_validation), torch.tensor(y_validation)),
        batch_size=128, shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=128, shuffle=False
    )

    return train_loader, validation_loader, test_loader

# -------------------------
# Model
# -------------------------
class LSTMModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -------------------------
# Training loop
# -------------------------
def training_loop(feature_cols, train_loader, validation_loader, test_loader, demand_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(input_size=len(feature_cols)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        validation_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                validation_loss += criterion(model(X_batch), y_batch).item()

        validation_loss /= len(validation_loader)

        print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Validation Loss {validation_loss:.4f}")

        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), "best_model.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    preds, acts = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).cpu().numpy())
            acts.append(y_batch.numpy())

    preds = demand_scaler.inverse_transform(np.concatenate(preds))
    acts = demand_scaler.inverse_transform(np.concatenate(acts))

    return acts, preds

# -------------------------
# Metrics
# -------------------------
def metrics(acts, preds):
    mae = mean_absolute_error(acts, preds)
    mape = mean_absolute_percentage_error(acts, preds) * 100
    rmse = np.sqrt(np.mean((acts - preds) ** 2))

    print(f"\nMAE {mae:.2f}")
    print(f"MAPE {mape:.2f}%")
    print(f"RMSE {rmse:.2f}")

def plot_graph(df, validation_end, acts, preds):
    test_dates = df['index'].iloc[validation_end:].reset_index(drop=True)

    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, acts, label="Actual")
    plt.plot(test_dates, preds, label="Predicted", alpha=0.8)

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Energy Demand (MW)")
    plt.title("LSTM Energy Demand Forecast — Brazil")

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df, feature_cols = load_and_process()
    print(df['index'].min())
    print(df['index'].max())
    print(f"Train ends: {df['index'].iloc[int(len(df)*0.7)]}")
    print(f"Val ends: {df['index'].iloc[int(len(df)*0.8)]}")
    train_data, validation_data, test_data, validation_end = split_data(df)
    train_scaled, validation_scaled, test_scaled, demand_scaler = scale_data(feature_cols, train_data, validation_data, test_data)
    X_train, y_train, X_validation, y_validation, X_test, y_test = build_sequence(train_scaled, validation_scaled, test_scaled)
    train_loader, validation_loader, test_loader = data_loaders(X_train, y_train, X_validation, y_validation, X_test, y_test)
    acts, preds = training_loop(feature_cols, train_loader, validation_loader, test_loader, demand_scaler)
    metrics(acts, preds)
    plot_graph(df, validation_end, acts, preds)