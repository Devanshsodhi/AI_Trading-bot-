import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_CONFIG, FORECAST_CONFIG, SENTIMENT_CONFIG, RL_CONFIG,
    ENV_CONFIG, DECISION_CONFIG, INDICATORS_CONFIG,
    GENERATIVE_CONFIG, TIMEGAN_CONFIG, LLM_CONFIG,
    STRATEGY_GEN_CONFIG, SCENARIO_GEN_CONFIG
)
from src.data import DataLoader, TechnicalIndicators, MarketScenarioGenerator
from src.forecasting_agent import ForecastingAgent
from src.forecasting_agent.timegan_forecaster_persistent import TimeGANForecaster
from src.sentiment_agent import SentimentAgent
from src.trading_agent import TradingAgent, StrategyGenerator
from src.decision_engine import DecisionEngine, LLMRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingSystem:
    def __init__(self, ticker, days=30, retrain=False):
        self.ticker = ticker
        self.days = days
        self.retrain = retrain

        self.data_loader = DataLoader(ticker, period=DATA_CONFIG['default_period'])
        self.data = self.data_loader.fetch_data()
        self.data = TechnicalIndicators.add_all_indicators(self.data)

        self.forecaster = (
            TimeGANForecaster(TIMEGAN_CONFIG, ticker, 'models')
            if GENERATIVE_CONFIG['use_timegan'] else ForecastingAgent(FORECAST_CONFIG)
        )
        self.sentiment = SentimentAgent(SENTIMENT_CONFIG)
        self.rl = TradingAgent(RL_CONFIG, ENV_CONFIG)
        self.decision = DecisionEngine(DECISION_CONFIG)

        self.llm = LLMRecommender(LLM_CONFIG) if GENERATIVE_CONFIG['use_llm'] else None
        self.strategy_gen = StrategyGenerator(STRATEGY_GEN_CONFIG) if GENERATIVE_CONFIG['use_strategy_gen'] else None
        self.scenario_gen = MarketScenarioGenerator(SCENARIO_GEN_CONFIG) if GENERATIVE_CONFIG['use_scenario_gen'] else None

    def forecast(self):
        last_price = self.data['Close'].iloc[-1]
        try:
            if GENERATIVE_CONFIG['use_timegan']:
                self.forecaster.last_price = last_price
                if not self.retrain and self.forecaster.load_models(last_price=last_price):
                    logger.info("Using cached TimeGAN model")
                else:
                    self.forecaster.train(self.data, force_retrain=self.retrain)

            result = self.forecaster.generate_scenarios(
                num_scenarios=TIMEGAN_CONFIG.get('num_scenarios', 100),
                forecast_days=FORECAST_CONFIG.get('forecast_days', 5)
            )
            result['current_price'] = last_price
            return result

        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return {'mean_forecast': [last_price]*5, 'trend': 'Neutral', 'confidence': 0.5}

    def analyze_sentiment(self):
        return self.sentiment.analyze_ticker(self.ticker)

    def trade(self, forecast, sentiment):
        data_train = self.data.iloc[:-self.days]
        if len(data_train) > 100:
            self.rl.train(data_train)
        return self.rl.predict_action(self.data, forecast, sentiment)

    def generate_strategies(self):
        if not self.strategy_gen:
            return None
        data_train = self.data.iloc[:-self.days]
        strategies = self.strategy_gen.generate_strategies(
            data_train, num_strategies=STRATEGY_GEN_CONFIG['num_strategies']
        )
        current = self.data.iloc[-1]
        ensemble = self.strategy_gen.generate_ensemble_signal(current)
        return {'strategies': strategies, 'ensemble': ensemble}

    def generate_scenarios(self):
        if not self.scenario_gen:
            return None
        price = self.data_loader.get_latest_price()
        vol = TechnicalIndicators.calculate_volatility(self.data)
        scenarios = self.scenario_gen.generate_scenarios(price, vol)
        return self.scenario_gen.analyze_scenarios(scenarios)

    def llm_recommend(self, forecast, sentiment, rl):
        if not self.llm:
            return None
        price = self.data_loader.get_latest_price()
        return self.llm.generate_recommendation(self.ticker, forecast, sentiment, rl, price)

    def decide(self, forecast, sentiment, rl, llm=None, strategy=None):
        price = self.data_loader.get_latest_price()
        decision = self.decision.make_decision(forecast, sentiment, rl, self.ticker, price)
        if llm and llm['action'] == decision['recommendation']:
            decision['confidence'] = min(decision['confidence'] * 1.1, 0.99)
        if strategy and strategy['ensemble']['action'] == decision['recommendation']:
            decision['confidence'] = min(decision['confidence'] * 1.05, 0.99)
        return decision

    def run(self):
        logger.info(f"Running trading system for {self.ticker}")
        forecast = self.forecast()
        sentiment = self.analyze_sentiment()
        rl_result = self.trade(forecast, sentiment)
        strategies = self.generate_strategies()
        scenarios = self.generate_scenarios()
        llm_result = self.llm_recommend(forecast, sentiment, rl_result)
        decision = self.decide(forecast, sentiment, rl_result, llm_result, strategies)
        self.display(forecast, sentiment, rl_result, decision)
        return decision

    def display(self, forecast, sentiment, rl_result, decision):
        print(f"\n--- {self.ticker} TRADING SUMMARY ---")
        print(f"Price: {self.data_loader.get_latest_price():.2f}")
        print(f"Forecast: {forecast['trend']} | Confidence: {forecast.get('confidence', 0):.2f}")
        print(f"Sentiment: {sentiment['overall_label']} ({sentiment['overall_score']:+.2f})")
        print(f"RL Action: {rl_result['action']} ({rl_result['confidence']:.2f})")
        print(f"Decision: {decision['strength']} {decision['recommendation']} ({decision['confidence']:.2f})")
        print("-----------------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()

    system = TradingSystem(args.ticker, args.days, args.retrain)
    system.run()


if __name__ == '__main__':
    main()
