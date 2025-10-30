"""
Generative Trading System
Combines Forecasting, Sentiment, RL, and LLM Agents
"""

import argparse
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv

# setup
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_CONFIG, FORECAST_CONFIG, SENTIMENT_CONFIG, RL_CONFIG,
    GENERATIVE_CONFIG, TIMEGAN_CONFIG, LLM_CONFIG, STRATEGY_GEN_CONFIG,
    SCENARIO_GEN_CONFIG, DECISION_CONFIG, ENV_CONFIG
)

from src.data import DataLoader, TechnicalIndicators, MarketScenarioGenerator
from src.forecasting_agent import ForecastingAgent
from src.forecasting_agent.timegan_forecaster_persistent import TimeGANForecaster
from src.sentiment_agent import SentimentAgent
from src.trading_agent import TradingAgent, StrategyGenerator
from src.decision_engine import DecisionEngine, LLMRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

class GenerativeTradingSystem:
    def __init__(self, ticker='AAPL', days=30, retrain=False):
        self.ticker = ticker
        self.days = days
        self.retrain = retrain

        self.data_loader = DataLoader(ticker, period=DATA_CONFIG['default_period'])
        self.data = self.data_loader.fetch_data()
        self.data = TechnicalIndicators.add_all_indicators(self.data)

        if GENERATIVE_CONFIG.get('use_timegan'):
            self.forecaster = TimeGANForecaster(TIMEGAN_CONFIG, ticker, 'models')
        else:
            self.forecaster = ForecastingAgent(FORECAST_CONFIG)

        self.sentiment = SentimentAgent(SENTIMENT_CONFIG)
        self.trader = TradingAgent(RL_CONFIG, ENV_CONFIG)
        self.decision = DecisionEngine(DECISION_CONFIG)

        self.llm = LLMRecommender(LLM_CONFIG) if GENERATIVE_CONFIG.get('use_llm') else None
        self.strategy_gen = StrategyGenerator(STRATEGY_GEN_CONFIG) if GENERATIVE_CONFIG.get('use_strategy_gen') else None
        self.scenario_gen = MarketScenarioGenerator(SCENARIO_GEN_CONFIG) if GENERATIVE_CONFIG.get('use_scenario_gen') else None

    def run_forecast(self):
        last_price = self.data['Close'].iloc[-1]
        if GENERATIVE_CONFIG.get('use_timegan'):
            try:
                if not self.retrain and self.forecaster.load_models(last_price=last_price):
                    log.info("Loaded cached TimeGAN")
                else:
                    self.forecaster.train(self.data, force_retrain=self.retrain)
                res = self.forecaster.generate_scenarios()
            except Exception as e:
                log.warning(f"TimeGAN failed: {e}")
                res = {'mean_forecast': [last_price]*5, 'trend': 'Neutral'}
        else:
            res = self.forecaster.forecast(self.data)
        return res

    def run_sentiment(self):
        return self.sentiment.analyze_ticker(self.ticker)

    def run_rl(self, forecast, sentiment):
        train_data = self.data.iloc[:-self.days]
        if len(train_data) > 100:
            self.trader.train(train_data)
        return self.trader.predict_action(self.data, forecast, sentiment)

    def run(self):
        log.info(f"Running pipeline for {self.ticker}")

        forecast = self.run_forecast()
        sentiment = self.run_sentiment()
        rl_out = self.run_rl(forecast, sentiment)

        strategy_out = self.strategy_gen.generate_strategies(self.data) if self.strategy_gen else None
        llm_out = self.llm.generate_recommendation(self.ticker, forecast, sentiment, rl_out, self.data['Close'].iloc[-1]) if self.llm else None
        scenario_out = self.scenario_gen.generate_scenarios(self.data['Close'].iloc[-1], 0.02) if self.scenario_gen else None

        final = self.decision.make_decision(forecast, sentiment, rl_out, self.ticker, self.data['Close'].iloc[-1])
        if llm_out: final['llm'] = llm_out

        print(f"\n=== {self.ticker} Recommendation ===")
        print(f"Forecast: {forecast.get('trend')}")
        print(f"Sentiment: {sentiment.get('overall_label')}")
        print(f"RL Agent: {rl_out.get('action')}")
        if llm_out: print(f"LLM Suggests: {llm_out.get('action')}")
        print(f"Final: {final.get('recommendation')} ({final.get('confidence',0):.1%})")

        return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='AAPL')
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--use-timegan', action='store_true')
    p.add_argument('--use-llm', action='store_true')
    a = p.parse_args()

    if a.use_timegan: GENERATIVE_CONFIG['use_timegan'] = True
    if a.use_llm: GENERATIVE_CONFIG['use_llm'] = True

    sys = GenerativeTradingSystem(a.ticker, a.days, a.retrain)
    sys.run()

if __name__ == "__main__":
    main()
