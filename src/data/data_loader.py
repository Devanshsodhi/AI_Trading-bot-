"""
Data Loader Module
Handles fetching and preprocessing stock market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Fetch and preprocess stock market data"""
    
    def __init__(self, ticker: str, period: str = '2y', interval: str = '1d'):
        """
        Initialize DataLoader
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Data period (e.g., '1y', '2y', '5y')
            interval: Data interval (e.g., '1d', '1h')
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period, interval=self.interval)
            
            if self.data.empty:
                logger.warning(f"No data found for {self.ticker}, using mock data")
                self.data = self._generate_mock_data()
            else:
                logger.info(f"Successfully fetched {len(self.data)} data points")
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            logger.info("Using mock data instead")
            self.data = self._generate_mock_data()
            return self.data
    
    def _generate_mock_data(self, days: int = 500) -> pd.DataFrame:
        """
        Generate mock stock data for testing
        
        Args:
            days: Number of days of data to generate
            
        Returns:
            DataFrame with mock OHLCV data
        """
        logger.info(f"Generating {days} days of mock data...")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movement using geometric Brownian motion
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, days)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'Open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
            'High': price * (1 + np.random.uniform(0, 0.02, days)),
            'Low': price * (1 - np.random.uniform(0, 0.02, days)),
            'Close': price,
            'Volume': np.random.randint(1000000, 10000000, days),
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def get_latest_price(self) -> float:
        """Get the most recent closing price"""
        if self.data is None:
            self.fetch_data()
        return float(self.data['Close'].iloc[-1])
    
    def get_price_history(self, days: int = 30) -> pd.Series:
        """
        Get recent price history
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            Series of closing prices
        """
        if self.data is None:
            self.fetch_data()
        return self.data['Close'].tail(days)
    
    def split_train_test(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            self.fetch_data()
            
        split_idx = int(len(self.data) * train_ratio)
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data
    
    def normalize_data(self, data: pd.DataFrame, 
                      fit_on: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize data using min-max scaling
        
        Args:
            data: Data to normalize
            fit_on: Optional data to fit scaler on (for train/test consistency)
            
        Returns:
            Tuple of (normalized_data, scaler_params)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        
        if fit_on is not None:
            scaler.fit(fit_on)
        else:
            scaler.fit(data)
            
        normalized = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns,
            index=data.index
        )
        
        scaler_params = {
            'min': scaler.data_min_,
            'max': scaler.data_max_,
            'scale': scaler.scale_,
        }
        
        return normalized, scaler_params
    
    def get_company_info(self) -> dict:
        """Get company information"""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            return {
                'name': info.get('longName', self.ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
            }
        except Exception as e:
            logger.warning(f"Could not fetch company info: {e}")
            return {
                'name': self.ticker,
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 1000000000,
            }
