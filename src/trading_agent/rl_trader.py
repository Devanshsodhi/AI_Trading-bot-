"""
RL Trading Agent
Uses PPO or DQN for optimal trading policy
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    PPO = None
    DQN = None
    BaseCallback = None
    DummyVecEnv = None
    gym = None

from .trading_env import TradingEnvironment

logger = logging.getLogger(__name__)


if RL_AVAILABLE and BaseCallback:
    class TrainingCallback(BaseCallback):
        """Callback for monitoring training progress"""
        
        def __init__(self, verbose=0):
            super(TrainingCallback, self).__init__(verbose)
            self.episode_rewards = []
            
        def _on_step(self) -> bool:
            return True
        
        def _on_rollout_end(self) -> None:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                self.episode_rewards.append(mean_reward)
else:
    TrainingCallback = None


class TradingAgent:
    """
    Reinforcement Learning Trading Agent
    Uses PPO or DQN to learn optimal trading policy
    """
    
    def __init__(self, rl_config: dict, env_config: dict):
        """
        Initialize Trading Agent
        
        Args:
            rl_config: RL algorithm configuration
            env_config: Trading environment configuration
        """
        if not RL_AVAILABLE:
            logger.warning("RL libraries not available - install stable-baselines3 and gymnasium")
            
        self.rl_config = rl_config
        self.env_config = env_config
        self.algorithm = rl_config.get('algorithm', 'PPO')
        
        self.model = None
        self.env = None
        
        logger.info(f"Trading Agent initialized with {self.algorithm}")
    
    def create_environment(self, data: pd.DataFrame) -> TradingEnvironment:
        """
        Create trading environment
        
        Args:
            data: Market data with indicators
            
        Returns:
            Trading environment
        """
        env = TradingEnvironment(data, self.env_config)
        return env
    
    def train(self, train_data: pd.DataFrame, verbose: int = 0):
        """
        Train the RL agent
        
        Args:
            train_data: Training data
            verbose: Verbosity level
        """
        logger.info(f"Training {self.algorithm} agent...")
        
        # Create environment
        env = self.create_environment(train_data)
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize model
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=self.rl_config.get('learning_rate', 0.0003),
                n_steps=self.rl_config.get('n_steps', 2048),
                batch_size=self.rl_config.get('batch_size', 64),
                n_epochs=self.rl_config.get('n_epochs', 10),
                gamma=self.rl_config.get('gamma', 0.99),
                gae_lambda=self.rl_config.get('gae_lambda', 0.95),
                clip_range=self.rl_config.get('clip_range', 0.2),
                ent_coef=self.rl_config.get('ent_coef', 0.01),
                verbose=verbose,
            )
        elif self.algorithm == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                vec_env,
                learning_rate=self.rl_config.get('learning_rate', 0.0003),
                buffer_size=self.rl_config.get('buffer_size', 50000),
                learning_starts=self.rl_config.get('learning_starts', 1000),
                batch_size=self.rl_config.get('batch_size', 64),
                gamma=self.rl_config.get('gamma', 0.99),
                exploration_fraction=self.rl_config.get('exploration_fraction', 0.1),
                exploration_final_eps=self.rl_config.get('exploration_final_eps', 0.05),
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Train
        callback = TrainingCallback()
        total_timesteps = self.rl_config.get('total_timesteps', 100000)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True if verbose > 0 else False,
        )
        
        logger.info(f"Training completed after {total_timesteps} timesteps")
        
        # Store environment for later use
        self.env = env
    
    def predict_action(self, data: pd.DataFrame, 
                      forecast_result: Optional[Dict] = None,
                      sentiment_result: Optional[Dict] = None) -> Dict:
        """
        Predict trading action
        
        Args:
            data: Current market data
            forecast_result: Optional forecast from forecasting agent
            sentiment_result: Optional sentiment from sentiment agent
            
        Returns:
            Dictionary with action and recommendation
        """
        if self.model is None:
            logger.warning("Model not trained, training now...")
            self.train(data)
        
        logger.info("Predicting trading action...")
        
        # Create environment with current data
        env = self.create_environment(data)
        
        # Get observation
        obs, _ = env.reset()
        
        # Predict action
        action, _states = self.model.predict(obs, deterministic=True)
        action = int(action)
        
        # Map action to recommendation
        action_map = {
            0: 'HOLD',
            1: 'BUY',
            2: 'SELL',
        }
        
        recommendation = action_map[action]
        
        # Calculate position size based on confidence
        current_price = data['Close'].iloc[-1]
        max_shares = int(self.env_config['initial_balance'] / current_price)
        
        # Adjust position size based on forecast and sentiment if available
        confidence = 0.5  # Base confidence
        
        if forecast_result:
            forecast_confidence = forecast_result.get('confidence', 0.5)
            confidence = (confidence + forecast_confidence) / 2
        
        if sentiment_result:
            sentiment_score = sentiment_result.get('overall_score', 0)
            sentiment_confidence = abs(sentiment_score)
            confidence = (confidence + sentiment_confidence) / 2
        
        position_size = int(max_shares * confidence)
        
        # Simulate the action to get expected metrics
        expected_return = self._simulate_action(env, action)
        
        result = {
            'action': recommendation,
            'action_code': action,
            'position_size': position_size,
            'confidence': float(confidence),
            'expected_return': float(expected_return),
            'current_price': float(current_price),
            'rationale': self._generate_rationale(
                recommendation, confidence, forecast_result, sentiment_result
            ),
        }
        
        logger.info(f"Predicted action: {recommendation} (confidence: {confidence:.2%})")
        return result
    
    def _simulate_action(self, env: TradingEnvironment, action: int) -> float:
        """Simulate action to estimate expected return"""
        # Run a short simulation
        obs, _ = env.reset()
        total_reward = 0
        
        for _ in range(min(30, len(env.data) - 1)):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return total_reward
    
    def _generate_rationale(self, action: str, confidence: float,
                           forecast_result: Optional[Dict],
                           sentiment_result: Optional[Dict]) -> str:
        """Generate human-readable rationale for the action"""
        rationale_parts = []
        
        # Base recommendation
        rationale_parts.append(f"RL agent recommends {action} with {confidence:.0%} confidence.")
        
        # Add forecast information
        if forecast_result:
            trend = forecast_result.get('trend', 'Unknown')
            trend_prob = forecast_result.get('trend_probability', 0)
            rationale_parts.append(
                f"Forecast indicates {trend.lower()} trend with {trend_prob:.0%} probability."
            )
        
        # Add sentiment information
        if sentiment_result:
            sentiment_label = sentiment_result.get('overall_label', 'Neutral')
            sentiment_score = sentiment_result.get('overall_score', 0)
            rationale_parts.append(
                f"Market sentiment is {sentiment_label.lower()} (score: {sentiment_score:+.2f})."
            )
        
        # Action-specific advice
        if action == 'BUY':
            rationale_parts.append("Consider entering a long position with appropriate risk management.")
        elif action == 'SELL':
            rationale_parts.append("Consider reducing exposure or taking profits.")
        else:
            rationale_parts.append("Maintain current position and monitor market conditions.")
        
        return ' '.join(rationale_parts)
    
    def backtest(self, test_data: pd.DataFrame) -> Dict:
        """
        Backtest the trained agent
        
        Args:
            test_data: Test data
            
        Returns:
            Performance metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before backtesting")
        
        logger.info("Running backtest...")
        
        # Create test environment
        env = self.create_environment(test_data)
        
        # Run episode
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        
        logger.info(f"Backtest complete. Total return: {metrics['total_return']:.2%}")
        return metrics
    
    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, data: pd.DataFrame):
        """Load model from disk"""
        if os.path.exists(path):
            # Create environment
            env = self.create_environment(data)
            vec_env = DummyVecEnv([lambda: env])
            
            # Load model
            if self.algorithm == 'PPO':
                self.model = PPO.load(path, env=vec_env)
            elif self.algorithm == 'DQN':
                self.model = DQN.load(path, env=vec_env)
            
            self.env = env
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model file not found: {path}")
