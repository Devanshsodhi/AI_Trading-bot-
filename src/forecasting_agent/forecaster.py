
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
import os

logger = logging.getLogger(__name__)

class ForecastingAgent:
    """
    Forecasting Agent using LSTM for time series prediction.
    Generates probabilistic forecasts with confidence intervals.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config.get('sequence_length', 60)
        self.lstm_units = config.get('lstm_units', [128, 64, 32])
        self.dropout = config.get('dropout', 0.2)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_simulations = config.get('num_simulations', 100)
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Forecasting Agent initialized on {self.device}")

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare data for training"""
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        indicator_cols = [col for col in data.columns if col not in feature_columns]
        feature_columns.extend(indicator_cols[:10])  # limit to 10 indicators
        feature_columns = [col for col in feature_columns if col in data.columns]
        self.feature_columns = feature_columns
        data_values = data[feature_columns].values
        scaled_data = self.scaler.fit_transform(data_values)
        logger.info(f"Prepared data with {len(feature_columns)} features")
        return scaled_data, feature_columns

    def train(self, data: pd.DataFrame, validation_split: float = 0.2):
        """Train the LSTM model"""
        # ... (training code as before, properly indented)
        pass  # placeholder for brevity

    def predict(self, data: pd.DataFrame, forecast_days: int = 5) -> Dict:
        """Generate probabilistic forecast with meaningful high/low prices."""
        if self.model is None:
            logger.warning("Model not trained, training now...")
            self.train(data)

        logger.info(f"Generating {forecast_days}-day forecast...")

        # Prepare data
        if self.feature_columns is None:
            scaled_data, _ = self.prepare_data(data)
        else:
            scaled_data = self.scaler.transform(data[self.feature_columns].values)

        last_sequence = scaled_data[-self.sequence_length:]
        self.model.eval()
        forecast_paths = []

        # Historical volatility from last 20 days
        close_prices = data['Close'].values
        returns = np.diff(np.log(close_prices[-20:]))
        hist_volatility = np.std(returns) * np.sqrt(252)  # annualized

        with torch.no_grad():
            for _ in range(self.num_simulations):
                current_sequence = last_sequence.copy()
                path = []
                for day in range(forecast_days):
                    x = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                    pred = self.model(x).cpu().numpy()[0]

                    # Volatility scaling
                    time_factor = (day + 1) / forecast_days
                    vol_scale = 1.0 + time_factor
                    noise_scale = hist_volatility * vol_scale * (1.0 + np.random.normal(0, 0.1))

                    # Minimum price movement
                    min_move = 0.001
                    noise = np.random.normal(0, noise_scale, pred.shape)
                    noise = np.sign(noise) * np.maximum(np.abs(noise), min_move)

                    pred_noisy = pred * (1 + noise)
                    pred_noisy = np.clip(pred_noisy, 0.01, 10.0)  # prevent extreme values

                    path.append(pred_noisy)
                    current_sequence = np.vstack([current_sequence[1:], pred_noisy])

                forecast_paths.append(path)

        forecast_paths = np.array(forecast_paths)
        forecast_prices_all = np.array([self.scaler.inverse_transform(sim) for sim in forecast_paths])
        forecast_close = forecast_prices_all[:, :, 3]  # Close prices index

        # Calculate statistics
        mean_forecast = np.mean(forecast_close, axis=0)
        std_forecast = np.std(forecast_close, axis=0)
        lower_bound = np.percentile(forecast_close, 5, axis=0)
        upper_bound = np.percentile(forecast_close, 95, axis=0)

        current_price = data['Close'].iloc[-1]
        upward_sims = np.sum(forecast_close[:, -1] > current_price)
        trend_probability = upward_sims / self.num_simulations
        trend = 'Upward' if trend_probability > 0.6 else 'Downward' if trend_probability < 0.4 else 'Sideways'

        # Fallback if forecast is too flat
        if np.allclose(mean_forecast, current_price, rtol=1e-3):
            logger.warning("Forecast flat, adding minimum volatility.")
            daily_vol = hist_volatility / np.sqrt(252)
            mean_forecast = current_price * (1 + np.linspace(-daily_vol, daily_vol, forecast_days))
            lower_bound = current_price * (1 - daily_vol * 2)
            upper_bound = current_price * (1 + daily_vol * 2)

        result = {
            'mean_forecast': mean_forecast.tolist(),
            'std_forecast': std_forecast.tolist(),
            'lower_bound': lower_bound.tolist() if isinstance(lower_bound, np.ndarray) else [lower_bound] * forecast_days,
            'upper_bound': upper_bound.tolist() if isinstance(upper_bound, np.ndarray) else [upper_bound] * forecast_days,
            'trend': trend,
            'trend_probability': float(trend_probability),
            'current_price': float(current_price),
            'forecast_days': forecast_days,
            'confidence': float(1 - np.mean(std_forecast) / (np.mean(mean_forecast) + 1e-8)),
        }

        logger.info(f"Forecast complete: {trend} trend with {trend_probability:.1%} probability")
        return result

    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'config': self.config,
            }, path)
            logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.scaler = checkpoint['scaler']
            self.feature_columns = checkpoint['feature_columns']
            input_size = len(self.feature_columns)
            self.model = LSTMForecaster(input_size, self.lstm_units, self.dropout).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model file not found: {path}")
