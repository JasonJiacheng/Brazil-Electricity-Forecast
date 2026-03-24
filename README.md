# Brazil Energy Demand Forecasting

An LSTM-based deep learning model for next-hour electricity demand forecasting in Brazil, achieving approximately 0.90% MAPE on unseen test data (2022–2023) when trained on 2020–2022.  
When COVID years are included in the test data, the MAPE increases to approximately 1.40%.

## Overview

This project uses a Long Short-Term Memory (LSTM) neural network to model hourly electricity demand using:

- Historical demand (autoregressive input)  
- Temporal features (hour, day, month)  
- Weather data  

It also investigates how training data selection affects performance, highlighting the impact of distribution shift (COVID-19) on forecasting accuracy.

## Model Architecture

Input (168 timesteps × 11 features)  
→ LSTM (2 layers, 128 hidden units, dropout=0.2)  
→ Last hidden state  
→ Linear (128 → 64) + ReLU  
→ Linear (64 → 1)  
→ Output: Next-hour demand (MW)

### Training Configuration

- Loss: Mean Squared Error (MSE)  
- Optimiser: Adam (lr = 0.001)  
- Batch size: 128  
- Epochs: up to 30  
- Early stopping: patience = 5  
- Best model checkpoint saved and restored  

## Data

### Energy Demand

- [`energy_demand_hourly_brazil.csv`](https://www.kaggle.com/datasets/arusouza/23-years-of-hourly-eletric-energy-demand-brazil)  
- Hourly electricity demand (MW)  
- Covers 2000–2023 

### Weather Data

- `open-meteo-10.02S55.01W420m.csv`  
- Hourly weather data from Central-West Brazil (Open-Meteo.com)  
- Features: temperature, humidity, wind speed, feels-like temperature  
- Missing values interpolated linearly  
- Merged with energy demand data on timestamp  

## Features

| Feature | Description |
|--------|------------|
| `hourly_demand` | Target + autoregressive input |
| `hour_sin`, `hour_cos` | Hour-of-day cyclic encoding |
| `dow_sin`, `dow_cos` | Day-of-week cyclic encoding |
| `month_sin`, `month_cos` | Month cyclic encoding |
| `temperature` | °C |
| `humidity` | % |
| `wind_speed` | km/h |
| `feels_like` | °C |

Cyclic encoding ensures continuity in time features (e.g., hour 23 ≈ hour 0).

## Methodology

### Train / Validation / Test Split

Chronological split to avoid leakage:

| Split | Proportion |
|------|-----------|
| Train | 70% |
| Validation | 10% |
| Test | 20% |

### Sequence Construction

- Input window: 168 hours (7 days)  
- Target: next-hour demand  
- Sliding window approach  
- Validation and test sets prepended with prior sequence context  

### Scaling

- `StandardScaler` applied separately to demand and other features  
- Fit on training data only  

## Results

### Best Model Performance

- MAPE: ~0.90%  
- MAE: ~595 MW  
- RMSE: ~810 MW  

## Data Cutoff Experiments

Models were trained with different start years while keeping the same chronological split.

| Start Year | MAE | MAPE | RMSE |
|------------|-----|------|------|
| 2000 | 989 | 1.43% | 1482 |
| 2005 | 972 | 1.39% | 1488 |
| 2009 | 1008 | 1.43% | 1473 |
| 2010 | 914 | 1.30% | 1321 |
| 2015 | 869 | 1.24% | 1175 |
| 2020 | 595 | 0.89% | 810 |

### Key Insight

Performance improves when training data includes recent (COVID-era) patterns. Test data (2021–2023) contains post-COVID demand behavior. Models trained only on pre-2020 data must generalize across a structural break. Models trained from 2020 onward perform best because they match the test distribution.  
Note that 2010 is when COVID-era patterns start being included in the validation set, hence a noticeable performance improvement.

> Forecast accuracy depends strongly on alignment between training and test data distributions.

## Forecasting Horizon

The model predicts one hour ahead using the previous 168 hours of demand and weather data. This horizon is operationally relevant for real-time grid balancing and short-term planning.

## Limitations

- Single weather location for a national grid  
- Requires full 168-hour input window  
- Cannot forecast long horizons without future input features  
- Sensitive to distribution shifts (e.g., COVID)  
- Run-to-run variance due to stochastic training  

## Future Work

- Multi-step forecasting (e.g., 24-hour ahead)  
- Multi-location weather integration  
- Transformer-based architectures (e.g., Temporal Fusion Transformer)  
- Rolling retraining to handle concept drift  
- Benchmarking against classical models (SARIMA, naive baselines)  

## Installation

```bash
pip install -r requirements.txt
python LSTM_model.py
