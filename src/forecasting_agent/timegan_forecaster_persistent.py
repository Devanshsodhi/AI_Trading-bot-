import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import logging
import joblib
import random
from sklearn.preprocessing import MinMaxScaler
from .realistic_constraints import make_predictions_realistic

logger = logging.getLogger(__name__)

# ----------------- Generator -----------------
class TimeSeriesGenerator(nn.Module):
    """Generator network for TimeGAN"""
    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = max(4, hidden_dim)
        mid = max(1, hidden_dim // 2)

        self.model = nn.Sequential(
            nn.LSTM(noise_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mid, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

    def forward(self, z, seq_len: int):
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.model[0](z_seq)
        out = out.permute(0, 2, 1)
        out = self.model[1](out).permute(0, 2, 1)
        out, _ = self.model[2](out)
        out = out.permute(0, 2, 1)
        out = self.model[3](out).permute(0, 2, 1)
        return torch.tanh(self.fc(out))

# ----------------- Discriminator -----------------
class TimeSeriesDiscriminator(nn.Module):
    """Discriminator network for TimeGAN"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(4, hidden_dim)
        mid = max(1, hidden_dim // 2)

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, max(4, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(4, hidden_dim // 2), 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mid, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

    def get_features(self, x):
        out, _ = self.lstm1(x)
        out = out.permute(0, 2, 1)
        out = self.bn1(out).permute(0, 2, 1)
        out, _ = self.lstm2(out)
        out = out.permute(0, 2, 1)
        out = self.bn2(out).permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        return (out * attn_weights).sum(dim=1)

    def forward(self, x):
        return self.fc(self.get_features(x))

# ----------------- TimeGAN Forecaster -----------------
class TimeGANForecaster:
    def __init__(self, config: dict, ticker: str = None, model_dir: str = "models"):
        self.config = config or {}
        self.hidden_dim = max(16, config.get("hidden_dim", 64))  # Reduced from 128
        self.noise_dim = max(8, config.get("noise_dim", 32))  # Reduced from 64
        self.epochs = int(config.get("epochs", 15))  # Reduced from 50
        self.batch_size = max(4, min(config.get("batch_size", 64), 128))  # Increased for speed
        self.learning_rate = float(config.get("learning_rate", 2e-4))  # Increased for faster convergence
        self.seq_len = max(2, int(config.get("seq_len", 30)))  # Reduced from 60
        self.ticker = (ticker or "default").upper()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.generator = None
        self.discriminator = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.last_price = None
        self.historical_data = None
        self.input_dim = None  # Store input dimension

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        logger.info(f"TimeGAN initialized for {self.ticker}")
        
        # Try to load existing models
        self.load_models()

    # ---------- Utility ----------
    def _path(self, name: str):
        return self.model_dir / f"timegan_{self.ticker.lower()}_{name}.pth"

    def save_models(self):
        """Save model weights, scaler, and metadata"""
        if self.generator and self.discriminator:
            torch.save(self.generator.state_dict(), self._path("generator"))
            torch.save(self.discriminator.state_dict(), self._path("discriminator"))
            joblib.dump(self.scaler, self.model_dir / f"timegan_{self.ticker.lower()}_scaler.pkl")
            # Save metadata for model reconstruction
            metadata = {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'noise_dim': self.noise_dim,
                'last_price': self.last_price
            }
            joblib.dump(metadata, self.model_dir / f"timegan_{self.ticker.lower()}_metadata.pkl")
            logger.info("Models saved")
        else:
            logger.warning("Models not initialized")
    
    def load_models(self) -> bool:
        """Load pre-trained models if they exist"""
        try:
            gen_path = self._path("generator")
            disc_path = self._path("discriminator")
            scaler_path = self.model_dir / f"timegan_{self.ticker.lower()}_scaler.pkl"
            meta_path = self.model_dir / f"timegan_{self.ticker.lower()}_metadata.pkl"
            
            if gen_path.exists() and disc_path.exists() and scaler_path.exists() and meta_path.exists():
                logger.info(f"Loading models for {self.ticker}")
                
                # Load metadata
                metadata = joblib.load(meta_path)
                self.input_dim = metadata['input_dim']
                self.hidden_dim = metadata['hidden_dim']
                self.noise_dim = metadata['noise_dim']
                self.last_price = metadata.get('last_price')
                
                # Initialize models with correct dimensions
                self.generator = TimeSeriesGenerator(self.noise_dim, self.hidden_dim, self.input_dim).to(self.device)
                self.discriminator = TimeSeriesDiscriminator(self.input_dim, self.hidden_dim).to(self.device)
                
                # Load weights
                self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
                self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))
                self.scaler = joblib.load(scaler_path)
                
                self.generator.eval()
                self.discriminator.eval()
                
                logger.info(f"Models loaded for {self.ticker}")
                return True
            else:
                logger.info(f"No cached models for {self.ticker}")
                return False
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
            self.generator = None
            self.discriminator = None
            return False

    def _prepare_sequences(self, data: pd.DataFrame) -> np.ndarray:
        if data is None or len(data) < self.seq_len + 1:
            raise ValueError("Insufficient data for sequence preparation")
        self.last_price = float(data.iloc[-1, 0])
        scaled = self.scaler.fit_transform(data.values)
        return np.array([scaled[i:i + self.seq_len] for i in range(len(scaled) - self.seq_len)])

    # ---------- Training ----------
    def train(self, data: pd.DataFrame, force_retrain: bool = False) -> bool:
        """
        Train the TimeGAN model.
        
        Args:
            data: Historical price data
            force_retrain: If True, retrain even if models exist. If False, use existing models.
        """
        try:
            # Check if we should use existing models
            if not force_retrain and self.generator is not None and self.discriminator is not None:
                logger.info(f"Using cached models for {self.ticker}")
                self.historical_data = data.copy()
                self.last_price = float(data.iloc[-1, 0])
                return True
            
            logger.info(f"Training TimeGAN for {self.ticker}")
            
            self.historical_data = data.copy()
            sequences = self._prepare_sequences(data)
            self.input_dim = sequences.shape[2]

            self.generator = TimeSeriesGenerator(self.noise_dim, self.hidden_dim, self.input_dim).to(self.device)
            self.discriminator = TimeSeriesDiscriminator(self.input_dim, self.hidden_dim).to(self.device)

            opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate * 0.5, betas=(0.5, 0.999))
            bce_loss, mse_loss = nn.BCELoss(), nn.MSELoss()

            num_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
            logger.info(f"Training: {self.epochs} epochs, batch_size={self.batch_size}")
            
            # Early stopping parameters
            best_g_loss = float('inf')
            patience = self.patience if hasattr(self, 'patience') else 20  # Much higher patience
            min_delta = self.min_delta if hasattr(self, 'min_delta') else 0.001  # Minimum improvement
            min_epochs = 30  # Don't stop before this many epochs
            patience_counter = 0
            
            for epoch in range(self.epochs):
                perm = np.random.permutation(len(sequences))
                d_losses, g_losses = [], []

                for batch_num, i in enumerate(range(0, len(perm), self.batch_size)):
                    batch_idx = perm[i:i + self.batch_size]
                    real = torch.tensor(sequences[batch_idx], dtype=torch.float32, device=self.device)
                    valid = torch.ones((real.size(0), 1), device=self.device)
                    fake = torch.zeros_like(valid)
                    noise = torch.randn(real.size(0), self.noise_dim, device=self.device)

                    # Train Discriminator
                    if batch_num % 2 == 0:
                        opt_D.zero_grad()
                        gen = self.generator(noise, self.seq_len).detach()
                        loss_D = 0.5 * (bce_loss(self.discriminator(real), valid) + bce_loss(self.discriminator(gen), fake))
                        loss_D.backward()
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                        opt_D.step()
                        d_losses.append(loss_D.item())
                    else:
                        d_losses.append(d_losses[-1] if d_losses else 0.69)

                    # Train Generator
                    opt_G.zero_grad()
                    gen = self.generator(noise, self.seq_len)
                    adv_loss = bce_loss(self.discriminator(gen), valid)
                    loss_G = adv_loss
                    loss_G.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    opt_G.step()
                    g_losses.append(adv_loss.item())

                avg_d, avg_g = np.mean(d_losses), np.mean(g_losses)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - D_loss={avg_d:.4f}, G_loss={avg_g:.4f}")
                
                # Early stopping check
                if epoch + 1 >= min_epochs:
                    if avg_g < (best_g_loss - min_delta):
                        best_g_loss = avg_g
                        patience_counter = 0
                        if epoch % 5 == 0:
                            self.save_models()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                elif avg_g < best_g_loss:
                    best_g_loss = avg_g

            self.save_models()
            logger.info("Training complete")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    # ---------- Scenario Generation ----------
    def generate_scenarios(self, num_scenarios: int, forecast_days: int = 5) -> Dict[str, Any]:
        try:
            if not self.generator or self.last_price is None:
                return self._simple_generation(num_scenarios, forecast_days)

            self.generator.eval()
            
            # Get historical volatility and statistics
            if self.historical_data is not None and len(self.historical_data) > 60:
                hist_close = self.historical_data.iloc[:, 3].values  # Close prices
                hist_high = self.historical_data.iloc[:, 1].values   # High prices
                hist_low = self.historical_data.iloc[:, 2].values    # Low prices
                
                # Calculate daily returns and volatility
                returns = np.diff(np.log(hist_close[-60:]))
                daily_vol = float(np.std(returns))
                daily_return_mean = float(np.mean(returns))
                
                # Calculate average high-low range
                hl_range = np.mean((hist_high[-60:] - hist_low[-60:]) / hist_close[-60:])
            else:
                daily_vol = 0.02
                daily_return_mean = 0
                hl_range = 0.02
            
            # Generate scenarios using TimeGAN - NO CONSTRAINTS
            close_scenarios = []
            high_scenarios = []
            low_scenarios = []
            
            with torch.no_grad():
                for i in range(num_scenarios):
                    # Generate with random noise
                    noise = torch.randn(1, self.noise_dim, device=self.device)
                    gen_output = self.generator(noise, forecast_days).squeeze(0).cpu().numpy()
                    
                    # Inverse transform to get actual prices
                    dummy = np.zeros((forecast_days, self.scaler.scale_.shape[0]))
                    dummy[:, :min(gen_output.shape[1], dummy.shape[1])] = gen_output[:, :min(gen_output.shape[1], dummy.shape[1])]
                    ohlcv = self.scaler.inverse_transform(dummy)
                    
                    # OHLCV format: [Open, High, Low, Close, Volume]
                    gen_close = ohlcv[:, 3]
                    gen_high = ohlcv[:, 1]
                    gen_low = ohlcv[:, 2]
                    
                    # Scale from last known price (continuity)
                    scale_factor = self.last_price / gen_close[0] if gen_close[0] > 0 else 1.0
                    
                    scenario_close = gen_close * scale_factor
                    scenario_high = gen_high * scale_factor
                    scenario_low = gen_low * scale_factor
                    
                    # Only ensure logical relationship: high > close > low
                    for j in range(len(scenario_close)):
                        if scenario_high[j] < scenario_close[j]:
                            scenario_high[j] = scenario_close[j] * 1.005
                        if scenario_low[j] > scenario_close[j]:
                            scenario_low[j] = scenario_close[j] * 0.995
                    
                    close_scenarios.append(scenario_close)
                    high_scenarios.append(scenario_high)
                    low_scenarios.append(scenario_low)
            
            # Convert to arrays
            close_scenarios = np.array(close_scenarios)
            high_scenarios = np.array(high_scenarios)
            low_scenarios = np.array(low_scenarios)
            
            # Calculate statistics
            mean_close = np.mean(close_scenarios, axis=0)
            mean_high = np.mean(high_scenarios, axis=0)
            mean_low = np.mean(low_scenarios, axis=0)
            
            std_forecast = np.std(close_scenarios, axis=0)
            
            # Confidence bands (10th and 90th percentiles for tighter bounds)
            lower_bound = np.percentile(close_scenarios, 10, axis=0)
            upper_bound = np.percentile(close_scenarios, 90, axis=0)
            
            # Trend analysis
            final_prices = close_scenarios[:, -1]
            trend_prob = float(np.mean(final_prices > self.last_price))
            trend = 'Upward' if trend_prob > 0.55 else 'Downward' if trend_prob < 0.45 else 'Sideways'
            
            # Confidence calculation (based on coefficient of variation)
            cv = np.mean(std_forecast) / (np.mean(mean_close) + 1e-8)
            confidence = max(0.3, min(0.95, 1.0 - cv))

            return {
                'scenarios': close_scenarios.tolist(),
                'mean_forecast': mean_close.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'high_forecast': mean_high.tolist(),
                'low_forecast': mean_low.tolist(),
                'std_forecast': std_forecast.tolist(),
                'trend': trend,
                'trend_probability': trend_prob,
                'confidence': confidence,
                'current_price': float(self.last_price),
                'num_scenarios': num_scenarios,
                'forecast_days': forecast_days,
                'generation_method': 'TimeGAN'
            }
        except Exception as e:
            logger.error(f"Scenario generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._simple_generation(num_scenarios, forecast_days)

    # ---------- Fallback Generator ----------
    def _simple_generation(self, num_scenarios: int, forecast_days: int):
        if self.last_price is None:
            self.last_price = 100.0
        daily_vol = 0.3 / np.sqrt(252)
        scenarios = []
        for _ in range(num_scenarios):
            returns = np.random.normal(0, daily_vol, forecast_days)
            scenarios.append(self.last_price * np.cumprod(1 + returns))
        scenarios = np.array(scenarios)
        mean_f, std_f = np.mean(scenarios, axis=0), np.std(scenarios, axis=0)
        lower, upper = np.percentile(scenarios, 5, axis=0), np.percentile(scenarios, 95, axis=0)
        trend_prob = float(np.mean(scenarios[:, -1] > self.last_price))
        trend = 'Upward' if trend_prob > 0.6 else 'Downward' if trend_prob < 0.4 else 'Sideways'
        confidence = max(0.1, 1.0 - min(np.mean(std_f) / (np.mean(mean_f) + 1e-8), 1.0))
        return {
            'scenarios': scenarios.tolist(),
            'mean_forecast': mean_f.tolist(),
            'lower_bound': lower.tolist(),
            'upper_bound': upper.tolist(),
            'std_forecast': std_f.tolist(),
            'trend': trend,
            'trend_probability': trend_prob,
            'confidence': confidence,
            'current_price': float(self.last_price),
            'num_scenarios': num_scenarios,
            'forecast_days': forecast_days,
            'generation_method': 'Simple (Fallback)'
        }
