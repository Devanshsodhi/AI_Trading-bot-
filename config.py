"""Configuration file for AI Trading System"""

# Data Configuration
DATA_CONFIG = {
    'default_ticker': 'AAPL',
    'default_period': '2y',
    'default_interval': '1d',
    'forecast_days': 5,
    'train_test_split': 0.8,
}

# Forecasting Agent Configuration
FORECAST_CONFIG = {
    'sequence_length': 60,
    'lstm_units': [128, 64, 32],
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_simulations': 100,
    'forecast_days': 5,
}

# Sentiment Agent Configuration
SENTIMENT_CONFIG = {
    'model_name': 'ProsusAI/finbert',
    'max_articles': 20,
    'sentiment_threshold': 0.1,
}

# RL Trading Agent Configuration
RL_CONFIG = {
    'algorithm': 'PPO',
    'total_timesteps': 200000,
    'learning_rate': 0.0001,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
}

# Trading Environment Configuration
ENV_CONFIG = {
    'initial_balance': 10000,
    'transaction_cost': 0.001,
    'slippage': 0.0005,
    'max_position': 1.0,
    'reward_scaling': 1e-4,
}

# Decision Engine Configuration
DECISION_CONFIG = {
    'confidence_threshold': 0.6,
    'sentiment_weight': 0.3,
    'forecast_weight': 0.4,
    'rl_weight': 0.3,
}

# Technical Indicators Configuration
INDICATORS_CONFIG = {
    'sma_periods': [20, 50, 200],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
}

# Generative AI Configuration
GENERATIVE_CONFIG = {
    'use_timegan': True,
    'use_llm': True,
    'use_strategy_gen': True,
    'use_scenario_gen': True,
}

# TimeGAN Configuration (Optimized for speed + accuracy)
TIMEGAN_CONFIG = {
    'hidden_dim': 128,
    'noise_dim': 64,
    'epochs': 60,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_scenarios': 200,
    'seq_len': 60,
    'patience': 15,
    'min_delta': 0.0001,
}

# LLM Configuration
LLM_CONFIG = {
    'provider': 'groq',
    'llm_model': 'llama-3.1-8b-instant',
    'openai_model': 'gpt-3.5-turbo',
    'temperature': 0.2,
    'max_tokens': 2000,
    'use_mock_if_unavailable': True,
}

# Strategy Generator Configuration
STRATEGY_GEN_CONFIG = {
    'population_size': 50,
    'generations': 20,
    'mutation_rate': 0.2,
    'crossover_rate': 0.7,
    'num_strategies': 10,
}

# Scenario Generator Configuration
SCENARIO_GEN_CONFIG = {
    'num_scenarios': 100,
    'scenario_days': 30,
    'include_stress_tests': True,
}
