"""
Technical Indicators Module
Calculate various technical indicators for stock analysis
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        try:
            # Moving Averages
            df = TechnicalIndicators.add_moving_averages(df)
            
            # RSI
            df = TechnicalIndicators.add_rsi(df)
            
            # MACD
            df = TechnicalIndicators.add_macd(df)
            
            # Bollinger Bands
            df = TechnicalIndicators.add_bollinger_bands(df)
            
            # Volume indicators
            df = TechnicalIndicators.add_volume_indicators(df)
            
            # Momentum indicators
            df = TechnicalIndicators.add_momentum_indicators(df)
            
            # Fill any NaN values
            df = df.bfill().ffill()
            
            logger.info(f"Added {len(df.columns) - 5} technical indicators")
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            
        return df
    
    @staticmethod
    def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # Exponential Moving Averages
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        return df
    
    @staticmethod
    def add_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        df = data.copy()
        df['RSI'] = ta.momentum.rsi(df['Close'], window=period)
        return df
    
    @staticmethod
    def add_macd(data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators"""
        df = data.copy()
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        return df
    
    @staticmethod
    def add_bollinger_bands(data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df = data.copy()
        
        bollinger = ta.volatility.BollingerBands(df['Close'], window=period, window_dev=std)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_width'] = bollinger.bollinger_wband()
        
        return df
    
    @staticmethod
    def add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        df = data.copy()
        
        # On-Balance Volume
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Money Flow Index
        df['MFI'] = ta.volume.money_flow_index(
            df['High'], df['Low'], df['Close'], df['Volume'], window=14
        )
        
        return df
    
    @staticmethod
    def add_momentum_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        df = data.copy()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Rate of Change
        df['ROC'] = ta.momentum.roc(df['Close'], window=12)
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        return df
    
    @staticmethod
    def get_current_signals(data: pd.DataFrame) -> Dict[str, str]:
        """
        Get current trading signals from technical indicators
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Dictionary of indicator signals
        """
        latest = data.iloc[-1]
        signals = {}
        
        # RSI Signal
        if latest['RSI'] < 30:
            signals['RSI'] = 'Oversold (Bullish)'
        elif latest['RSI'] > 70:
            signals['RSI'] = 'Overbought (Bearish)'
        else:
            signals['RSI'] = 'Neutral'
        
        # MACD Signal
        if latest['MACD'] > latest['MACD_signal']:
            signals['MACD'] = 'Bullish'
        else:
            signals['MACD'] = 'Bearish'
        
        # Moving Average Signal
        if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
            signals['MA_Trend'] = 'Strong Uptrend'
        elif latest['Close'] > latest['SMA_50']:
            signals['MA_Trend'] = 'Uptrend'
        elif latest['Close'] < latest['SMA_50'] < latest['SMA_200']:
            signals['MA_Trend'] = 'Strong Downtrend'
        else:
            signals['MA_Trend'] = 'Downtrend'
        
        # Bollinger Bands Signal
        if latest['Close'] > latest['BB_high']:
            signals['Bollinger'] = 'Above Upper Band (Overbought)'
        elif latest['Close'] < latest['BB_low']:
            signals['Bollinger'] = 'Below Lower Band (Oversold)'
        else:
            signals['Bollinger'] = 'Within Bands'
        
        return signals
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, window: int = 20) -> float:
        """Calculate historical volatility"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return float(volatility * np.sqrt(252))  # Annualized
    
    @staticmethod
    def calculate_sharpe_ratio(data: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        returns = data['Close'].pct_change()
        excess_returns = returns.mean() - risk_free_rate / 252
        std_returns = returns.std()
        
        if std_returns == 0:
            return 0.0
            
        sharpe = (excess_returns / std_returns) * np.sqrt(252)
        return float(sharpe)
