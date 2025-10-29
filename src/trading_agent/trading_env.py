"""
Trading Environment for Reinforcement Learning
Simulates stock trading with transaction costs and slippage
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for RL agents
    Follows OpenAI Gym interface
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, config: dict):
        """
        Initialize Trading Environment
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            config: Environment configuration
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.config = config
        
        # Environment parameters
        self.initial_balance = config.get('initial_balance', 10000)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        self.max_position = config.get('max_position', 1.0)
        self.reward_scaling = config.get('reward_scaling', 1e-4)
        
        # State variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.trades = []
        
        # Feature columns (exclude date if present)
        self.feature_columns = [col for col in data.columns 
                               if col not in ['Date', 'Datetime']]
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market features + portfolio state
        n_features = len(self.feature_columns) + 3  # +3 for balance, shares, total_value
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        logger.info(f"Trading environment initialized with {len(data)} steps")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Market features
        market_features = self.data.loc[self.current_step, self.feature_columns].values
        
        # Normalize market features
        market_features = market_features / (np.abs(market_features) + 1e-8)
        
        # Portfolio state (normalized)
        current_price = self.data.loc[self.current_step, 'Close']
        portfolio_value = self.balance + self.shares_held * current_price
        
        portfolio_features = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,
            portfolio_value / self.initial_balance,
        ])
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_features]).astype(np.float32)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        current_price = self.data.loc[self.current_step, 'Close']
        
        # Execute action
        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
        # action == 0 is Hold, do nothing
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Calculate reward
        new_total_value = self.balance + self.shares_held * current_price
        reward = self._calculate_reward(new_total_value)
        self.total_value = new_total_value
        
        # Get new observation
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape)
        
        # Additional info
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_value': self.total_value,
            'current_price': current_price,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_buy(self, price: float):
        """Execute buy order"""
        # Apply slippage (buy at slightly higher price)
        execution_price = price * (1 + self.slippage)
        
        # Calculate maximum shares we can buy
        available_balance = self.balance * self.max_position
        max_shares = int(available_balance / execution_price)
        
        if max_shares > 0:
            # Calculate cost with transaction fees
            cost = max_shares * execution_price
            transaction_fee = cost * self.transaction_cost
            total_cost = cost + transaction_fee
            
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.shares_held += max_shares
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': max_shares,
                    'price': execution_price,
                    'cost': total_cost,
                })
    
    def _execute_sell(self, price: float):
        """Execute sell order"""
        if self.shares_held > 0:
            # Apply slippage (sell at slightly lower price)
            execution_price = price * (1 - self.slippage)
            
            # Calculate proceeds
            proceeds = self.shares_held * execution_price
            transaction_fee = proceeds * self.transaction_cost
            net_proceeds = proceeds - transaction_fee
            
            self.balance += net_proceeds
            
            self.trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'shares': self.shares_held,
                'price': execution_price,
                'proceeds': net_proceeds,
            })
            
            self.shares_held = 0
    
    def _calculate_reward(self, new_total_value: float) -> float:
        """
        Calculate reward based on portfolio value change
        Uses risk-adjusted returns (Sharpe-like ratio)
        """
        # Simple return
        value_change = new_total_value - self.total_value
        simple_return = value_change / self.total_value
        
        # Scale reward
        reward = simple_return * self.reward_scaling
        
        # Penalize for holding too much cash (encourage trading)
        cash_ratio = self.balance / new_total_value
        if cash_ratio > 0.9:
            reward -= 0.01 * self.reward_scaling
        
        return float(reward)
    
    def render(self, mode='human'):
        """Render the environment"""
        current_price = self.data.loc[self.current_step, 'Close']
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares_held}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Profit: ${self.total_value - self.initial_balance:.2f}")
        print("-" * 50)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        final_value = self.total_value
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Calculate returns series
        if len(self.trades) > 1:
            trade_returns = []
            for i in range(1, len(self.trades)):
                if self.trades[i]['action'] == 'SELL' and self.trades[i-1]['action'] == 'BUY':
                    buy_price = self.trades[i-1]['price']
                    sell_price = self.trades[i]['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
            
            if trade_returns:
                sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_return': float(total_return),
            'final_value': float(final_value),
            'num_trades': len(self.trades),
            'sharpe_ratio': float(sharpe_ratio),
        }
