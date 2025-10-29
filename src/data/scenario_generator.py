"""
Market Scenario Generator
Generates synthetic market scenarios for stress testing and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketScenarioGenerator:
    """
    Generates realistic market scenarios using various models
    Useful for stress testing and what-if analysis
    """
    
    def __init__(self, config: dict):
        """
        Initialize Market Scenario Generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.num_scenarios = config.get('num_scenarios', 100)
        self.scenario_days = config.get('scenario_days', 30)
        
        logger.info("Market Scenario Generator initialized")
    
    def generate_scenarios(self, base_price: float, volatility: float = 0.02) -> Dict:
        """
        Generate multiple market scenarios
        
        Args:
            base_price: Starting price
            volatility: Daily volatility (default 2%)
            
        Returns:
            Dictionary with generated scenarios
        """
        logger.info(f"Generating {self.num_scenarios} market scenarios...")
        
        scenarios = {
            'bullish': self._generate_bullish_scenario(base_price, volatility),
            'bearish': self._generate_bearish_scenario(base_price, volatility),
            'volatile': self._generate_volatile_scenario(base_price, volatility),
            'sideways': self._generate_sideways_scenario(base_price, volatility),
            'crash': self._generate_crash_scenario(base_price, volatility),
            'recovery': self._generate_recovery_scenario(base_price, volatility),
            'monte_carlo': self._generate_monte_carlo_scenarios(base_price, volatility)
        }
        
        logger.info("Market scenarios generated successfully")
        return scenarios
    
    def _generate_bullish_scenario(self, base_price: float, volatility: float) -> pd.DataFrame:
        """Generate bullish market scenario"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # Upward trend with noise
        for i in range(1, self.scenario_days):
            drift = 0.001  # 0.1% daily drift
            noise = np.random.normal(0, volatility)
            change = drift + noise
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Scenario': 'Bullish'
        })
        
        return df
    
    def _generate_bearish_scenario(self, base_price: float, volatility: float) -> pd.DataFrame:
        """Generate bearish market scenario"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # Downward trend with noise
        for i in range(1, self.scenario_days):
            drift = -0.001  # -0.1% daily drift
            noise = np.random.normal(0, volatility)
            change = drift + noise
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Scenario': 'Bearish'
        })
        
        return df
    
    def _generate_volatile_scenario(self, base_price: float, volatility: float) -> pd.DataFrame:
        """Generate high volatility scenario"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # High volatility, no clear trend
        high_vol = volatility * 3
        for i in range(1, self.scenario_days):
            change = np.random.normal(0, high_vol)
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Scenario': 'Volatile'
        })
        
        return df
    
    def _generate_sideways_scenario(self, base_price: float, volatility: float) -> pd.DataFrame:
        """Generate sideways/ranging market scenario"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # Mean-reverting behavior
        mean = base_price
        reversion_speed = 0.1
        
        for i in range(1, self.scenario_days):
            # Mean reversion
            deviation = prices[-1] - mean
            reversion = -reversion_speed * deviation / mean
            noise = np.random.normal(0, volatility)
            change = reversion + noise
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Scenario': 'Sideways'
        })
        
        return df
    
    def _generate_crash_scenario(self, base_price: float, volatility: float) -> pd.DataFrame:
        """Generate market crash scenario"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # Sudden drop followed by recovery
        crash_day = self.scenario_days // 3
        
        for i in range(1, self.scenario_days):
            if i == crash_day:
                # Crash: -10% to -20%
                change = -0.10 - np.random.random() * 0.10
            elif i < crash_day:
                # Before crash: normal
                change = np.random.normal(0, volatility)
            else:
                # After crash: slow recovery
                change = 0.002 + np.random.normal(0, volatility * 1.5)
            
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Scenario': 'Crash'
        })
        
        return df
    
    def _generate_recovery_scenario(self, base_price: float, volatility: float) -> pd.DataFrame:
        """Generate recovery scenario (V-shaped)"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # V-shaped recovery
        bottom_day = self.scenario_days // 2
        
        for i in range(1, self.scenario_days):
            if i < bottom_day:
                # Decline phase
                drift = -0.002
            else:
                # Recovery phase
                drift = 0.003
            
            noise = np.random.normal(0, volatility)
            change = drift + noise
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Scenario': 'Recovery'
        })
        
        return df
    
    def _generate_monte_carlo_scenarios(self, base_price: float, 
                                       volatility: float) -> List[pd.DataFrame]:
        """Generate multiple Monte Carlo scenarios"""
        
        scenarios = []
        
        for scenario_id in range(min(self.num_scenarios, 20)):  # Limit to 20 for performance
            dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
            prices = [base_price]
            
            # Geometric Brownian Motion
            drift = 0.0005  # Small positive drift
            
            for i in range(1, self.scenario_days):
                change = np.random.normal(drift, volatility)
                prices.append(prices[-1] * (1 + change))
            
            df = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Scenario': f'MonteCarlo_{scenario_id+1}'
            })
            
            scenarios.append(df)
        
        return scenarios
    
    def analyze_scenarios(self, scenarios: Dict) -> Dict:
        """
        Analyze generated scenarios
        
        Args:
            scenarios: Dictionary of generated scenarios
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        for scenario_name, scenario_data in scenarios.items():
            if scenario_name == 'monte_carlo':
                # Analyze Monte Carlo scenarios
                returns = []
                for mc_scenario in scenario_data:
                    initial = mc_scenario['Close'].iloc[0]
                    final = mc_scenario['Close'].iloc[-1]
                    ret = (final - initial) / initial
                    returns.append(ret)
                
                analysis[scenario_name] = {
                    'mean_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'min_return': float(np.min(returns)),
                    'max_return': float(np.max(returns)),
                    'positive_scenarios': int(np.sum(np.array(returns) > 0)),
                    'total_scenarios': len(returns)
                }
            else:
                # Analyze single scenario
                initial = scenario_data['Close'].iloc[0]
                final = scenario_data['Close'].iloc[-1]
                total_return = (final - initial) / initial
                
                # Calculate volatility
                returns = scenario_data['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)  # Annualized
                
                # Max drawdown
                cummax = scenario_data['Close'].cummax()
                drawdown = (scenario_data['Close'] - cummax) / cummax
                max_drawdown = drawdown.min()
                
                analysis[scenario_name] = {
                    'total_return': float(total_return),
                    'annualized_volatility': float(vol),
                    'max_drawdown': float(max_drawdown),
                    'final_price': float(final)
                }
        
        return analysis
    
    def generate_stress_test(self, base_price: float) -> Dict:
        """
        Generate stress test scenarios
        
        Args:
            base_price: Starting price
            
        Returns:
            Stress test results
        """
        logger.info("Generating stress test scenarios...")
        
        stress_scenarios = {
            'extreme_crash': self._generate_extreme_crash(base_price),
            'flash_crash': self._generate_flash_crash(base_price),
            'prolonged_bear': self._generate_prolonged_bear(base_price),
            'black_swan': self._generate_black_swan(base_price)
        }
        
        return stress_scenarios
    
    def _generate_extreme_crash(self, base_price: float) -> pd.DataFrame:
        """Generate extreme crash scenario (-30% to -50%)"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        # Rapid decline
        crash_magnitude = -0.40  # -40%
        daily_decline = crash_magnitude / (self.scenario_days // 2)
        
        for i in range(1, self.scenario_days):
            if i < self.scenario_days // 2:
                change = daily_decline + np.random.normal(0, 0.01)
            else:
                # Stabilization
                change = np.random.normal(0, 0.015)
            
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({'Date': dates, 'Close': prices, 'Scenario': 'ExtremeCrash'})
    
    def _generate_flash_crash(self, base_price: float) -> pd.DataFrame:
        """Generate flash crash scenario (sudden drop and recovery)"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        flash_day = self.scenario_days // 3
        
        for i in range(1, self.scenario_days):
            if i == flash_day:
                # Flash crash: -15%
                change = -0.15
            elif i == flash_day + 1:
                # Immediate recovery: +10%
                change = 0.10
            else:
                # Normal volatility
                change = np.random.normal(0, 0.02)
            
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({'Date': dates, 'Close': prices, 'Scenario': 'FlashCrash'})
    
    def _generate_prolonged_bear(self, base_price: float) -> pd.DataFrame:
        """Generate prolonged bear market"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days * 2, freq='D')
        prices = [base_price]
        
        # Slow, steady decline
        for i in range(1, len(dates)):
            change = -0.002 + np.random.normal(0, 0.015)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({'Date': dates, 'Close': prices, 'Scenario': 'ProlongedBear'})
    
    def _generate_black_swan(self, base_price: float) -> pd.DataFrame:
        """Generate black swan event (rare, extreme event)"""
        
        dates = pd.date_range(start=datetime.now(), periods=self.scenario_days, freq='D')
        prices = [base_price]
        
        event_day = np.random.randint(5, self.scenario_days - 5)
        
        for i in range(1, self.scenario_days):
            if i == event_day:
                # Black swan: -25% to -35%
                change = -0.25 - np.random.random() * 0.10
            else:
                # Normal market
                change = np.random.normal(0.0005, 0.02)
            
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({'Date': dates, 'Close': prices, 'Scenario': 'BlackSwan'})
    
    def visualize_scenarios(self, scenarios: Dict) -> str:
        """Generate text visualization of scenarios"""
        
        viz = "Market Scenario Summary:\n"
        viz += "=" * 60 + "\n\n"
        
        for name, data in scenarios.items():
            if name != 'monte_carlo':
                initial = data['Close'].iloc[0]
                final = data['Close'].iloc[-1]
                change = ((final - initial) / initial) * 100
                
                viz += f"{name.upper():20s}: ${initial:.2f} → ${final:.2f} ({change:+.1f}%)\n"
        
        return viz
