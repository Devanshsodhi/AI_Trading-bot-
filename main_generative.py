"""
Generative AI Trading System - Main Pipeline
Combines TimeGAN, LLM, RL Agent, and Sentiment Analysis
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_CONFIG, FORECAST_CONFIG, SENTIMENT_CONFIG,
    RL_CONFIG, ENV_CONFIG, DECISION_CONFIG, INDICATORS_CONFIG,
    GENERATIVE_CONFIG, TIMEGAN_CONFIG, LLM_CONFIG,
    STRATEGY_GEN_CONFIG, SCENARIO_GEN_CONFIG
)
from src.data import DataLoader, TechnicalIndicators, MarketScenarioGenerator
from src.forecasting_agent import ForecastingAgent
from src.forecasting_agent.timegan_forecaster_persistent import TimeGANForecaster
from src.sentiment_agent import SentimentAgent
from src.trading_agent import TradingAgent, StrategyGenerator
from src.decision_engine import DecisionEngine, LLMRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenerativeTradingSystem:
    """Generative AI Trading System combining multiple AI agents"""
    
    def __init__(self, ticker: str, days: int = 30, force_retrain: bool = False):
        self.ticker = ticker
        self.days = days
        self.force_retrain = force_retrain
        
        logger.info(f"Initializing system for {ticker}")
        
        self.data_loader = DataLoader(ticker, period=DATA_CONFIG['default_period'])
        self.load_data()
        
        # Initialize forecasting agent
        if GENERATIVE_CONFIG['use_timegan']:
            self.forecasting_agent = TimeGANForecaster(
                config=TIMEGAN_CONFIG,
                ticker=ticker,
                model_dir='models'
            )
        else:
            self.forecasting_agent = ForecastingAgent(FORECAST_CONFIG)
        
        # Initialize other agents
        self.sentiment_agent = SentimentAgent(SENTIMENT_CONFIG)
        self.trading_agent = TradingAgent(RL_CONFIG, ENV_CONFIG)
        self.decision_engine = DecisionEngine(DECISION_CONFIG)
        
        # Initialize optional components
        self.llm_recommender = LLMRecommender(LLM_CONFIG) if GENERATIVE_CONFIG['use_llm'] else None
        self.strategy_generator = StrategyGenerator(STRATEGY_GEN_CONFIG) if GENERATIVE_CONFIG['use_strategy_gen'] else None
        self.scenario_generator = MarketScenarioGenerator(SCENARIO_GEN_CONFIG) if GENERATIVE_CONFIG['use_scenario_gen'] else None
        
        self.data = None
        self.data_with_indicators = None
    
    def load_data(self):
        """Load and preprocess market data"""
        self.data = self.data_loader.fetch_data()
        self.data_with_indicators = TechnicalIndicators.add_all_indicators(self.data)
        
        if not self.data.empty:
            self.last_price = self.data['Close'].iloc[-1]
        
        return self.data_with_indicators
    
    def run_forecasting(self):
        """Run forecasting agent"""
        if self.data_with_indicators is None:
            self.load_data()
        
        last_price = self.data_with_indicators['Close'].iloc[-1]
        
        try:
            if GENERATIVE_CONFIG['use_timegan']:
                self.forecasting_agent.last_price = last_price
                
                if not self.force_retrain and self.forecasting_agent.load_models(last_price=last_price):
                    logger.info("Using cached TimeGAN models")
                else:
                    self.forecasting_agent.train(self.data_with_indicators, force_retrain=self.force_retrain)
            
            forecast_result = self.forecasting_agent.generate_scenarios(
                num_scenarios=TIMEGAN_CONFIG.get('num_scenarios', 100),
                forecast_days=FORECAST_CONFIG.get('forecast_days', 5)
            )
            forecast_result['current_price'] = last_price
            return forecast_result
            
        except Exception as e:
            logger.error(f"Forecasting error: {str(e)}")
            return {
                'mean_forecast': [last_price] * FORECAST_CONFIG.get('forecast_days', 5),
                'confidence': 0.5,
                'trend': 'Neutral',
                'current_price': last_price
            }
    
    def run_sentiment_analysis(self):
        """Run sentiment agent"""
        return self.sentiment_agent.analyze_ticker(self.ticker)
    
    def run_trading_agent(self, forecast_result, sentiment_result):
        """Run RL trading agent"""
        train_data = self.data_with_indicators.iloc[:-self.days]
        
        if len(train_data) > 100:
            self.trading_agent.train(train_data, verbose=0)
        
        return self.trading_agent.predict_action(
            self.data_with_indicators,
            forecast_result=forecast_result,
            sentiment_result=sentiment_result
        )
    
    def generate_strategies(self):
        """Generate trading strategies using genetic algorithms"""
        if not self.strategy_generator:
            return None
        
        train_data = self.data_with_indicators.iloc[:-self.days]
        strategies = self.strategy_generator.generate_strategies(
            train_data,
            num_strategies=STRATEGY_GEN_CONFIG['num_strategies']
        )
        
        current_data = self.data_with_indicators.iloc[-1]
        ensemble_signal = self.strategy_generator.generate_ensemble_signal(current_data)
        
        return {
            'strategies': strategies,
            'ensemble_signal': ensemble_signal,
            'num_strategies': len(strategies)
        }
    
    def generate_scenarios(self):
        """Generate market scenarios"""
        if not self.scenario_generator:
            return None
        
        current_price = self.data_loader.get_latest_price()
        volatility = TechnicalIndicators.calculate_volatility(self.data_with_indicators)
        
        scenarios = self.scenario_generator.generate_scenarios(current_price, volatility)
        analysis = self.scenario_generator.analyze_scenarios(scenarios)
        
        result = {'scenarios': scenarios, 'analysis': analysis}
        
        if SCENARIO_GEN_CONFIG['include_stress_tests']:
            result['stress_tests'] = self.scenario_generator.generate_stress_test(current_price)
        
        return result
    
    def generate_llm_recommendation(self, forecast_result, sentiment_result, rl_result):
        """Generate LLM-based recommendation"""
        if not self.llm_recommender:
            return None
        
        current_price = self.data_loader.get_latest_price()
        return self.llm_recommender.generate_recommendation(
            self.ticker, forecast_result, sentiment_result, rl_result, current_price
        )
    
    def generate_final_decision(self, forecast_result, sentiment_result, 
                               rl_result, llm_result=None, strategy_result=None):
        """Generate final recommendation combining all agents"""
        current_price = self.data_loader.get_latest_price()
        
        decision = self.decision_engine.make_decision(
            forecast_result=forecast_result,
            sentiment_result=sentiment_result,
            rl_result=rl_result,
            ticker=self.ticker,
            current_price=current_price
        )
        
        if llm_result:
            decision['llm_recommendation'] = llm_result
            if llm_result['action'] == decision['recommendation']:
                decision['confidence'] = min(decision['confidence'] * 1.1, 0.99)
        
        if strategy_result:
            decision['generated_strategies'] = strategy_result
            if strategy_result['ensemble_signal']['action'] == decision['recommendation']:
                decision['confidence'] = min(decision['confidence'] * 1.05, 0.99)
        
        return decision
    
    def run(self):
        """Run complete generative trading system pipeline"""
        logger.info(f"Starting analysis for {self.ticker}")
        
        try:
            self.load_data()
            forecast_result = self.run_forecasting()
            sentiment_result = self.run_sentiment_analysis()
            rl_result = self.run_trading_agent(forecast_result, sentiment_result)
            strategy_result = self.generate_strategies()
            scenario_result = self.generate_scenarios()
            llm_result = self.generate_llm_recommendation(forecast_result, sentiment_result, rl_result)
            
            decision = self.generate_final_decision(
                forecast_result, sentiment_result, rl_result, llm_result, strategy_result
            )
            
            if scenario_result:
                decision['scenario_analysis'] = scenario_result
            
            self.display_results(
                forecast_result, sentiment_result, rl_result,
                decision, llm_result, strategy_result, scenario_result
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise
    
    def display_results(self, forecast_result, sentiment_result, rl_result,
                       decision, llm_result=None, strategy_result=None, 
                       scenario_result=None):
        """Display comprehensive results"""
        
        print("\n" + "="*70)
        print(f"🤖 GENERATIVE AI TRADING RECOMMENDATION FOR {self.ticker}")
        print("="*70)
        
        # Current price
        current_price = self.data_loader.get_latest_price()
        print(f"\n📊 Current Price: ${current_price:.2f}")
        
        # Forecast
        print(f"\n📈 GENERATIVE FORECASTING ({forecast_result.get('generation_method', 'LSTM')})")
        print(f"   Trend: {forecast_result['trend']}")
        print(f"   Probability: {forecast_result['trend_probability']:.1%}")
        print(f"   Confidence: {forecast_result.get('confidence', 0):.1%}")
        if forecast_result.get('mean_forecast'):
            expected_price = forecast_result['mean_forecast'][-1]
            change = ((expected_price - current_price) / current_price) * 100
            print(f"   {DATA_CONFIG['forecast_days']}-day target: ${expected_price:.2f} ({change:+.1f}%)")
        if forecast_result.get('num_scenarios'):
            print(f"   Generated Scenarios: {forecast_result['num_scenarios']}")
        
        # Sentiment
        print(f"\n💭 SENTIMENT ANALYSIS")
        print(f"   Overall: {sentiment_result['overall_label']}")
        print(f"   Score: {sentiment_result['overall_score']:+.2f}")
        print(f"   Confidence: {sentiment_result['confidence']:.1%}")
        print(f"   Articles analyzed: {sentiment_result['article_count']}")
        
        # RL Agent
        print(f"\n🤖 RL AGENT")
        print(f"   Action: {rl_result['action']}")
        print(f"   Confidence: {rl_result['confidence']:.1%}")
        print(f"   Position size: {rl_result['position_size']} shares")
        
        # Generated Strategies
        if strategy_result:
            print(f"\n🧬 GENERATED STRATEGIES (Genetic Algorithm)")
            print(f"   Strategies evolved: {strategy_result['num_strategies']}")
            ensemble = strategy_result['ensemble_signal']
            print(f"   Ensemble signal: {ensemble['action']}")
            print(f"   Confidence: {ensemble['confidence']:.1%}")
            print(f"   Votes: BUY={ensemble['strategy_votes']['BUY']}, "
                  f"SELL={ensemble['strategy_votes']['SELL']}, "
                  f"HOLD={ensemble['strategy_votes']['HOLD']}")
        
        # LLM Recommendation
        if llm_result:
            print(f"\n🧠 LLM RECOMMENDATION ({llm_result.get('generation_method', 'LLM')})")
            print(f"   Action: {llm_result['action']}")
            print(f"   Confidence: {llm_result['confidence']:.1%}")
            print(f"   Reasoning: {llm_result['reasoning'][:150]}...")
        
        # Scenario Analysis
        if scenario_result and scenario_result.get('analysis'):
            print(f"\n🎲 SCENARIO ANALYSIS")
            mc_analysis = scenario_result['analysis'].get('monte_carlo', {})
            if mc_analysis:
                print(f"   Mean Return: {mc_analysis.get('mean_return', 0):.1%}")
                print(f"   Positive Scenarios: {mc_analysis.get('positive_scenarios', 0)}/{mc_analysis.get('total_scenarios', 0)}")
        
        # Final Decision
        print(f"\n✅ FINAL RECOMMENDATION")
        print(f"   Action: {decision['strength']} {decision['recommendation']}")
        print(f"   Confidence: {decision['confidence']:.1%}")
        print(f"   Combined Score: {decision['combined_score']:+.2f}")
        
        # Risk Assessment
        risk = decision['risk_assessment']
        print(f"\n⚠️  RISK ASSESSMENT")
        print(f"   Level: {risk['level']}")
        print(f"   Score: {risk['score']:.2f}")
        print(f"   {risk['description']}")
        
        # Target Range
        target = decision['target_range']
        print(f"\n🎯 TARGET RANGE")
        print(f"   Current: ${target['current']:.2f}")
        print(f"   Target Low: ${target['target_low']:.2f}")
        print(f"   Target High: ${target['target_high']:.2f}")
        print(f"   Stop Loss: ${target['stop_loss']:.2f}")
        print(f"   Take Profit: ${target['take_profit']:.2f}")
        
        print("\n" + "="*70)
        print("🎨 GENERATIVE AI COMPONENTS USED:")
        print("="*70)
        components_used = []
        if GENERATIVE_CONFIG['use_timegan']:
            components_used.append("✅ TimeGAN (Generative Forecasting)")
        if GENERATIVE_CONFIG['use_llm']:
            components_used.append("✅ LLM (Natural Language Generation)")
        if GENERATIVE_CONFIG['use_strategy_gen']:
            components_used.append("✅ Genetic Algorithm (Strategy Generation)")
        if GENERATIVE_CONFIG['use_scenario_gen']:
            components_used.append("✅ Scenario Generator (Market Simulation)")
        
        for component in components_used:
            print(f"   {component}")
        
        print("\n" + "="*70)
        print("⚠️  DISCLAIMER")
        print("="*70)
        print("This recommendation is generated by Generative AI for educational")
        print("purposes only. It does not constitute financial advice. Always do")
        print("your own research and consult with a qualified financial advisor")
        print("before making investment decisions.")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generative AI Trading System')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for analysis (default: 30)')
    parser.add_argument('--use-timegan', action='store_true',
                       help='Force use of TimeGAN')
    parser.add_argument('--use-llm', action='store_true',
                       help='Force use of LLM')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retrain TimeGAN model even if cached version exists')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.use_timegan:
        GENERATIVE_CONFIG['use_timegan'] = True
    if args.use_llm:
        GENERATIVE_CONFIG['use_llm'] = True
    
    # Create and run trading system
    system = GenerativeTradingSystem(
        ticker=args.ticker, 
        days=args.days,
        force_retrain=args.retrain
    )
    decision = system.run()
    
    return decision


if __name__ == '__main__':
    main()
