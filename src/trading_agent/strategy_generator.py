"""
Generative Strategy Generator
Uses genetic algorithms to evolve trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from deap import base, creator, tools, algorithms
import random

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Represents a trading strategy with rules"""
    
    def __init__(self, genes: List[float]):
        """
        Initialize strategy with genes
        
        Genes encode:
        - RSI thresholds (buy/sell)
        - MACD thresholds
        - Moving average periods
        - Position sizing rules
        - Stop loss/take profit levels
        """
        self.genes = genes
        self.fitness = 0.0
        
        # Decode genes into strategy parameters
        self.rsi_buy = 30 + genes[0] * 20  # 30-50
        self.rsi_sell = 50 + genes[1] * 30  # 50-80
        self.macd_threshold = genes[2] * 0.5  # 0-0.5
        self.ma_short = int(10 + genes[3] * 40)  # 10-50
        self.ma_long = int(50 + genes[4] * 150)  # 50-200
        self.position_size = 0.5 + genes[5] * 0.5  # 0.5-1.0
        self.stop_loss = 0.02 + genes[6] * 0.08  # 2%-10%
        self.take_profit = 0.05 + genes[7] * 0.15  # 5%-20%
    
    def generate_signal(self, data: pd.Series) -> str:
        """Generate trading signal based on strategy rules"""
        
        # Check if we have required indicators
        if 'RSI' not in data or 'MACD' not in data:
            return 'HOLD'
        
        rsi = data['RSI']
        macd = data['MACD']
        macd_signal = data.get('MACD_signal', 0)
        
        # Buy conditions
        if rsi < self.rsi_buy and macd > macd_signal + self.macd_threshold:
            return 'BUY'
        
        # Sell conditions
        if rsi > self.rsi_sell or macd < macd_signal - self.macd_threshold:
            return 'SELL'
        
        return 'HOLD'
    
    def __repr__(self):
        return f"Strategy(RSI:{self.rsi_buy:.0f}/{self.rsi_sell:.0f}, MACD:{self.macd_threshold:.2f}, MA:{self.ma_short}/{self.ma_long})"


class StrategyGenerator:
    """
    Generative Strategy Generator using Genetic Algorithms
    Evolves trading strategies to optimize performance
    """
    
    def __init__(self, config: dict):
        """
        Initialize Strategy Generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.population_size = config.get('population_size', 50)
        self.generations = config.get('generations', 20)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        
        self.best_strategies = []
        
        # Setup DEAP
        self._setup_deap()
        
        logger.info(f"Strategy Generator initialized (pop={self.population_size}, gen={self.generations})")
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Gene: random float between 0 and 1
        self.toolbox.register("attr_float", random.random)
        
        # Individual: 8 genes
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_float, n=8)
        
        # Population
        self.toolbox.register("population", tools.initRepeat, list,
                             self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def generate_strategies(self, data: pd.DataFrame, num_strategies: int = 10) -> List[TradingStrategy]:
        """
        Generate optimized trading strategies
        
        Args:
            data: Historical market data for backtesting
            num_strategies: Number of strategies to generate
            
        Returns:
            List of best trading strategies
        """
        logger.info(f"Generating {num_strategies} trading strategies...")
        
        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_strategy, data=data)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        # Run genetic algorithm
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=stats,
            verbose=False
        )
        
        # Get best individuals
        best_individuals = tools.selBest(population, num_strategies)
        
        # Convert to strategies
        strategies = []
        for individual in best_individuals:
            strategy = TradingStrategy(individual)
            strategy.fitness = individual.fitness.values[0]
            strategies.append(strategy)
        
        self.best_strategies = strategies
        
        logger.info(f"Generated {len(strategies)} strategies. Best fitness: {strategies[0].fitness:.2f}")
        
        return strategies
    
    def _evaluate_strategy(self, individual: List[float], data: pd.DataFrame) -> Tuple[float]:
        """
        Evaluate strategy fitness through backtesting
        
        Args:
            individual: Gene sequence
            data: Historical data
            
        Returns:
            Tuple with fitness score
        """
        strategy = TradingStrategy(individual)
        
        # Simulate trading
        balance = 10000
        shares = 0
        trades = 0
        
        for i in range(len(data) - 1):
            current_data = data.iloc[i]
            current_price = current_data['Close']
            
            signal = strategy.generate_signal(current_data)
            
            if signal == 'BUY' and balance > current_price:
                # Buy shares
                shares_to_buy = int((balance * strategy.position_size) / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    balance -= cost
                    shares += shares_to_buy
                    trades += 1
            
            elif signal == 'SELL' and shares > 0:
                # Sell shares
                proceeds = shares * current_price
                balance += proceeds
                shares = 0
                trades += 1
        
        # Final portfolio value
        final_price = data.iloc[-1]['Close']
        final_value = balance + shares * final_price
        
        # Calculate return
        total_return = (final_value - 10000) / 10000
        
        # Penalize if no trades
        if trades == 0:
            total_return = -0.1
        
        # Fitness is the return
        return (total_return,)
    
    def get_strategy_description(self, strategy: TradingStrategy) -> str:
        """Get human-readable description of strategy"""
        
        description = f"""
**Generated Trading Strategy**

**Entry Rules:**
- Buy when RSI < {strategy.rsi_buy:.0f}
- AND MACD crosses above signal by {strategy.macd_threshold:.2f}

**Exit Rules:**
- Sell when RSI > {strategy.rsi_sell:.0f}
- OR MACD crosses below signal by {strategy.macd_threshold:.2f}

**Risk Management:**
- Position Size: {strategy.position_size:.0%} of capital
- Stop Loss: {strategy.stop_loss:.1%}
- Take Profit: {strategy.take_profit:.1%}

**Technical Indicators:**
- Short MA: {strategy.ma_short} periods
- Long MA: {strategy.ma_long} periods

**Performance:**
- Fitness Score: {strategy.fitness:.2%}
"""
        
        return description
    
    def generate_ensemble_signal(self, data: pd.Series) -> Dict:
        """
        Generate ensemble signal from multiple strategies
        
        Args:
            data: Current market data
            
        Returns:
            Dictionary with ensemble recommendation
        """
        if not self.best_strategies:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'strategy_votes': {},
                'generation_method': 'Genetic Algorithm'
            }
        
        # Get signals from all strategies
        signals = []
        for strategy in self.best_strategies:
            signal = strategy.generate_signal(data)
            signals.append(signal)
        
        # Count votes
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        hold_votes = signals.count('HOLD')
        
        total_votes = len(signals)
        
        # Determine action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            action = 'BUY'
            confidence = buy_votes / total_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            action = 'SELL'
            confidence = sell_votes / total_votes
        else:
            action = 'HOLD'
            confidence = hold_votes / total_votes
        
        return {
            'action': action,
            'confidence': float(confidence),
            'strategy_votes': {
                'BUY': buy_votes,
                'SELL': sell_votes,
                'HOLD': hold_votes
            },
            'num_strategies': total_votes,
            'generation_method': 'Genetic Algorithm Ensemble'
        }
    
    def export_strategies(self) -> List[Dict]:
        """Export strategies as dictionaries"""
        
        exported = []
        for i, strategy in enumerate(self.best_strategies, 1):
            exported.append({
                'id': i,
                'genes': strategy.genes,
                'fitness': strategy.fitness,
                'parameters': {
                    'rsi_buy': strategy.rsi_buy,
                    'rsi_sell': strategy.rsi_sell,
                    'macd_threshold': strategy.macd_threshold,
                    'ma_short': strategy.ma_short,
                    'ma_long': strategy.ma_long,
                    'position_size': strategy.position_size,
                    'stop_loss': strategy.stop_loss,
                    'take_profit': strategy.take_profit
                }
            })
        
        return exported
