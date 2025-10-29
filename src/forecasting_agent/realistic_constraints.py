"""
Realistic constraints and validation for stock price predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class RealisticConstraints:
    """Apply realistic constraints to stock predictions"""
    
    @staticmethod
    def apply_volatility_constraint(predictions: np.ndarray, 
                                   historical_data: pd.DataFrame,
                                   max_daily_change: float = 0.15) -> np.ndarray:
        """
        Constrain predictions to realistic daily changes
        
        Args:
            predictions: Predicted prices
            historical_data: Historical price data
            max_daily_change: Maximum realistic daily change (default 15%)
        """
        # Calculate historical volatility
        returns = historical_data['Close'].pct_change().dropna()
        hist_volatility = returns.std()
        
        # Apply constraints
        constrained = predictions.copy()
        for i in range(1, len(constrained)):
            daily_change = (constrained[i] - constrained[i-1]) / constrained[i-1]
            
            # If change is too large, constrain it
            if abs(daily_change) > max_daily_change:
                max_allowed_change = max_daily_change if daily_change > 0 else -max_daily_change
                constrained[i] = constrained[i-1] * (1 + max_allowed_change)
        
        return constrained
    
    @staticmethod
    def smooth_predictions(predictions: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply moving average smoothing to reduce noise"""
        smoothed = predictions.copy()
        for i in range(window, len(smoothed)):
            smoothed[i] = np.mean(smoothed[i-window:i+1])
        return smoothed
    
    @staticmethod
    def respect_support_resistance(predictions: np.ndarray,
                                   historical_data: pd.DataFrame,
                                   days_lookback: int = 30) -> np.ndarray:
        """
        Ensure predictions respect support/resistance levels
        """
        recent_data = historical_data.tail(days_lookback)
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        # Constrain predictions within reasonable bounds
        constrained = predictions.copy()
        constrained = np.clip(constrained, support * 0.90, resistance * 1.10)
        
        return constrained
    
    @staticmethod
    def validate_prediction_quality(predictions: np.ndarray,
                                   historical_data: pd.DataFrame,
                                   current_price: float) -> Dict[str, Any]:
        """
        Validate prediction quality and return metrics
        """
        # Calculate metrics
        returns = historical_data['Close'].pct_change().dropna()
        hist_volatility = returns.std()
        
        # Predicted returns
        pred_returns = np.diff(predictions) / predictions[:-1]
        pred_volatility = np.std(pred_returns)
        
        # Check if prediction is realistic
        volatility_ratio = pred_volatility / hist_volatility
        
        # Average daily change
        avg_change = np.mean(np.abs(pred_returns))
        hist_avg_change = np.mean(np.abs(returns))
        
        is_realistic = (
            0.5 < volatility_ratio < 2.0 and  # Volatility similar to history
            predictions[0] * 0.8 < predictions[-1] < predictions[0] * 1.2  # Not too extreme
        )
        
        return {
            'is_realistic': is_realistic,
            'volatility_ratio': volatility_ratio,
            'avg_daily_change': avg_change,
            'hist_avg_daily_change': hist_avg_change,
            'predicted_volatility': pred_volatility,
            'historical_volatility': hist_volatility,
            'warning': None if is_realistic else "Predictions may be unrealistic"
        }


def make_predictions_realistic(scenarios: List[np.ndarray],
                               historical_data: pd.DataFrame,
                               current_price: float) -> List[np.ndarray]:
    """
    Apply minimal realistic constraints to prediction scenarios
    
    Args:
        scenarios: List of price prediction scenarios
        historical_data: Historical price data
        current_price: Current stock price
        
    Returns:
        List of constrained realistic scenarios
    """
    constraints = RealisticConstraints()
    realistic_scenarios = []
    
    # Calculate historical volatility for reference
    returns = historical_data.iloc[:, 3].pct_change().dropna()
    hist_vol = returns.std()
    max_daily_move = min(0.08, hist_vol * 4)  # Max 8% or 4x historical volatility
    
    for scenario in scenarios:
        # Convert to numpy array if it's a list
        scenario = np.array(scenario) if not isinstance(scenario, np.ndarray) else scenario
        
        # Only apply volatility constraint (no smoothing to preserve GAN patterns)
        constrained = [scenario[0]]
        for i in range(1, len(scenario)):
            daily_return = (scenario[i] - constrained[-1]) / constrained[-1]
            # Only clip if exceeds reasonable bounds
            if abs(daily_return) > max_daily_move:
                capped_return = np.sign(daily_return) * max_daily_move
                next_price = constrained[-1] * (1 + capped_return)
            else:
                next_price = scenario[i]
            constrained.append(next_price)
        
        realistic_scenarios.append(np.array(constrained))
    
    return realistic_scenarios
