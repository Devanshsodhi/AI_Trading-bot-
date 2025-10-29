"""Conversational AI Trading Agent using LangChain and Groq"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

from src.data.data_loader import DataLoader
from src.data.technical_indicators import TechnicalIndicators
from src.forecasting_agent import TimeGANForecaster
from src.sentiment_agent import SentimentAgent
from src.trading_agent import TradingAgent
from src.decision_engine import DecisionEngine

logging.basicConfig(level=logging.ERROR, format='%(message)s')
for logger_name in ['src', 'transformers', 'urllib3', 'httpx']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class TradingChatAgent:
    """Conversational AI agent for trading recommendations"""
    
    def __init__(self, use_timegan: bool = True, cache_duration: int = 300):
        self.use_timegan = use_timegan
        self.cache_duration = cache_duration
        self.cache = {}
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("No GROQ_API_KEY found")
            self.llm = None
        else:
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.2,
                    max_tokens=2000,
                    groq_api_key=api_key
                )
            except Exception as e:
                logger.error(f"Error initializing Groq: {e}")
                self.llm = None
        
        # Chat history stored as simple list
        self.chat_history = []
        
        self.sentiment_analyzer = SentimentAgent(config={
            'model_name': 'ProsusAI/finbert',
            'max_articles': 20,
            'sentiment_threshold': 0.1
        })
        
        self.decision_engine = DecisionEngine(config={
            'confidence_threshold': 0.6,
            'sentiment_weight': 0.3,
            'forecast_weight': 0.4,
            'rl_weight': 0.3
        })
        
        # System prompt for the agent
        self.system_prompt = """You are an expert AI trading advisor with deep knowledge of financial markets, 
technical analysis, and investment strategies. You have access to real-time market data, sentiment analysis, 
AI-powered forecasting models (TimeGAN), and reinforcement learning trading agents.

Your capabilities:
1. **Stock Analysis**: Analyze any stock with BUY/SELL/HOLD recommendations
2. **Market Insights**: Explain market trends, technical indicators, and price movements
3. **Risk Assessment**: Provide risk levels and position sizing advice
4. **Comparison**: Compare multiple stocks when asked
5. **Education**: Explain trading concepts, indicators, and strategies
6. **Portfolio Advice**: Discuss diversification and allocation strategies

You can answer questions like:
- "Should I invest in [stock]?"
- "What's the outlook for [stock]?"
- "Compare Apple and Microsoft"
- "What does RSI mean?"
- "Is this a good time to buy [stock]?"
- "What's the risk level of [stock]?"
- "Explain the current market trend"

When analyzing a stock, you receive:
- Current price and technical indicators
- AI forecast (trend, probability, target price)
- Sentiment analysis from news
- RL agent recommendation
- Risk assessment

Be conversational, insightful, and always include a disclaimer that this is not financial advice."""

        # Silent initialization
    
    def _get_cache_key(self, ticker: str) -> str:
        """Generate cache key for ticker analysis"""
        return f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached analysis is still valid"""
        cache_key = self._get_cache_key(ticker)
        if cache_key in self.cache:
            cached_time = self.cache[cache_key].get('timestamp')
            if cached_time:
                age = (datetime.now() - cached_time).total_seconds()
                return age < self.cache_duration
        return False
    
    def analyze_stock(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """
        Perform comprehensive stock analysis
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data to analyze
            
        Returns:
            Dictionary with analysis results
        """
        ticker = ticker.upper()
        
        # Check cache first
        cache_key = self._get_cache_key(ticker)
        if self._is_cache_valid(ticker):
            # Using cached analysis
            return self.cache[cache_key]['data']
        
        print(f"\n🔍 Analyzing {ticker}...", flush=True)
        
        try:
            # Load market data
            data_loader = DataLoader(ticker=ticker)
            data = data_loader.fetch_data()
            if data is None or len(data) == 0:
                return {
                    'error': f"Could not fetch data for {ticker}",
                    'ticker': ticker
                }
            
            # Add technical indicators
            data = TechnicalIndicators.add_all_indicators(data)
            current_price = float(data['Close'].iloc[-1])
            
            # Run forecasting
            print("📊 Running AI forecast...", flush=True)
            if self.use_timegan:
                config = {
                    'hidden_dim': 128,
                    'noise_dim': 64,
                    'epochs': 50,
                    'batch_size': 32,
                    'seq_len': 60
                }
                forecaster = TimeGANForecaster(config, ticker=ticker)
                forecaster.train(data, force_retrain=False)
                forecast_result = forecaster.generate_scenarios(num_scenarios=100, forecast_days=5)
            else:
                forecaster = ForecastingAgent(ticker=ticker)
                forecaster.train(data)
                forecast_result = forecaster.predict(forecast_days=5)
            
            # Run sentiment analysis
            print("💭 Analyzing market sentiment...", flush=True)
            sentiment_result = self.sentiment_analyzer.analyze_ticker(ticker)
            
            # Run RL agent
            print("🤖 Running RL trading agent...", flush=True)
            rl_config = {
                'algorithm': 'PPO',
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'total_timesteps': 10000  # Fast training for quick responses
            }
            env_config = {
                'initial_balance': 10000,
                'commission': 0.001,
                'max_position': 100
            }
            rl_agent = TradingAgent(rl_config=rl_config, env_config=env_config)
            rl_agent.train(data, verbose=0)
            rl_result = rl_agent.predict_action(data)
            
            # Generate final decision
            print("✨ Generating recommendation...\n", flush=True)
            decision = self.decision_engine.make_decision(
                ticker=ticker,
                current_price=current_price,
                forecast_result=forecast_result,
                sentiment_result=sentiment_result,
                rl_result=rl_result
            )
            
            # Compile results
            analysis = {
                'ticker': ticker,
                'current_price': current_price,
                'forecast': forecast_result,
                'sentiment': sentiment_result,
                'rl_agent': rl_result,
                'decision': decision,
                'timestamp': datetime.now()
            }
            
            # Cache the results
            self.cache[cache_key] = {
                'data': analysis,
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
            return {
                'error': str(e),
                'ticker': ticker
            }
    
    def format_analysis_for_llm(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results for LLM consumption"""
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        decision = analysis['decision']
        forecast = analysis['forecast']
        sentiment = analysis['sentiment']
        rl_agent = analysis['rl_agent']
        
        # Get forecast bounds with validation
        lower_bound = forecast.get('lower_bound', [analysis['current_price']])
        upper_bound = forecast.get('upper_bound', [analysis['current_price']])
        mean_forecast = forecast.get('mean_forecast', [analysis['current_price']])
        
        # Ensure bounds are different
        if isinstance(lower_bound, list) and isinstance(upper_bound, list):
            lower_val = lower_bound[-1] if lower_bound else analysis['current_price']
            upper_val = upper_bound[-1] if upper_bound else analysis['current_price']
            
            # If bounds are too close, use a reasonable spread
            if abs(upper_val - lower_val) < 0.01:
                mean_val = mean_forecast[-1] if mean_forecast else analysis['current_price']
                spread = mean_val * 0.03  # 3% spread
                lower_val = mean_val - spread
                upper_val = mean_val + spread
        else:
            lower_val = analysis['current_price'] * 0.97
            upper_val = analysis['current_price'] * 1.03
        
        formatted = f"""
STOCK ANALYSIS FOR {analysis['ticker']}
Current Price: ${analysis['current_price']:.2f}

AI FORECAST (TimeGAN):
- Trend: {forecast.get('trend', 'Unknown')}
- Probability: {forecast.get('trend_probability', 0)*100:.1f}%
- Confidence: {forecast.get('confidence', 0)*100:.1f}%
- 5-day Target: ${mean_forecast[-1] if mean_forecast else analysis['current_price']:.2f}
- Price Range: ${lower_val:.2f} - ${upper_val:.2f}

SENTIMENT ANALYSIS:
- Overall: {sentiment.get('overall_label', 'Neutral')} ({sentiment.get('overall_score', 0):+.2f})
- Confidence: {sentiment.get('confidence', 0)*100:.1f}%
- Articles Analyzed: {sentiment.get('article_count', 0)}
- Summary: {sentiment.get('summary', 'No summary available')}

RL AGENT RECOMMENDATION:
- Action: {rl_agent.get('action', 'HOLD')}
- Confidence: {rl_agent.get('confidence', 0)*100:.1f}%
- Position Size: {rl_agent.get('position_size', 0)} shares
- Rationale: {rl_agent.get('rationale', 'No rationale available')}

FINAL RECOMMENDATION:
- Action: {decision['strength']} {decision['recommendation']}
- Confidence: {decision['confidence']*100:.1f}%
- Combined Score: {decision['combined_score']:+.2f}

RISK ASSESSMENT:
- Level: {decision['risk_assessment']['level']}
- Score: {decision['risk_assessment']['score']:.2f}
- Description: {decision['risk_assessment']['description']}

TARGET RANGE:
- Target Low: ${decision['target_range']['target_low']:.2f}
- Target High: ${decision['target_range']['target_high']:.2f}
- Stop Loss: ${decision['target_range']['stop_loss']:.2f}
- Take Profit: ${decision['target_range']['take_profit']:.2f}
"""
        return formatted
    
    def _answer_simple_question(self, user_message: str, ticker: str) -> Optional[str]:
        """Answer simple data questions without full analysis"""
        message_lower = user_message.lower()
        
        try:
            # Fetch basic data
            data_loader = DataLoader(ticker=ticker)
            data = data_loader.fetch_data()
            
            if data is None or len(data) == 0:
                return None
            
            current_price = float(data['Close'].iloc[-1])
            
            # Yesterday's price
            if any(word in message_lower for word in ['yesterday', 'last day', 'previous day']):
                if len(data) >= 2:
                    yesterday_price = float(data['Close'].iloc[-2])
                    change = current_price - yesterday_price
                    change_pct = (change / yesterday_price) * 100
                    return f"""Yesterday's trading data for {ticker}:

📊 **Yesterday's Close**: ${yesterday_price:.2f}
📊 **Today's Price**: ${current_price:.2f}
📈 **Change**: ${change:+.2f} ({change_pct:+.2f}%)

Would you like a full analysis of {ticker}?"""
            
            # Current price only
            if any(word in message_lower for word in ['current price', 'price now', 'trading at', 'what is the price']):
                return f"""Current price for {ticker}:

📊 **Price**: ${current_price:.2f}

Would you like a full investment analysis?"""
            
            # Volume
            if 'volume' in message_lower:
                volume = int(data['Volume'].iloc[-1])
                avg_volume = int(data['Volume'].tail(20).mean())
                return f"""Trading volume for {ticker}:

📊 **Today's Volume**: {volume:,}
📊 **20-day Avg Volume**: {avg_volume:,}

Would you like a full analysis?"""
                
        except Exception as e:
            return None
        
        return None
    
    def chat(self, user_message: str) -> str:
        """
        Process user message and generate response
        
        Args:
            user_message: User's question or request
            
        Returns:
            AI agent's response
        """
        # Extract ticker if mentioned
        ticker = self._extract_ticker(user_message)
        
        if ticker:
            # Try to answer simple questions first (faster)
            simple_answer = self._answer_simple_question(user_message, ticker)
            if simple_answer:
                return simple_answer
            # Perform analysis
            analysis = self.analyze_stock(ticker)
            
            if 'error' in analysis:
                return f"I encountered an error analyzing {ticker}: {analysis['error']}"
            
            # Format analysis for LLM
            analysis_text = self.format_analysis_for_llm(analysis)
            
            # Generate response using LLM
            if self.llm:
                try:
                    prompt = f"""{self.system_prompt}

Here is the comprehensive analysis I've performed:

{analysis_text}

User Question: {user_message}

Please provide a clear, conversational response that answers the user's question based on this analysis. 
Include your recommendation with reasoning, but always end with a disclaimer."""

                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    return response.content
                    
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    return self._generate_fallback_response(analysis)
            else:
                return self._generate_fallback_response(analysis)
        else:
            # General conversation without specific ticker
            if self.llm:
                try:
                    prompt = f"""{self.system_prompt}

User Question: {user_message}

Please provide a helpful response. If the user is asking about investing in a specific stock, 
ask them to provide the ticker symbol."""
                    
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    return response.content
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    return "I'm here to help you with stock analysis! Please provide a ticker symbol (e.g., AAPL, TSLA, MSFT) and I'll analyze it for you."
            else:
                return "I'm here to help you with stock analysis! Please provide a ticker symbol (e.g., AAPL, TSLA, MSFT) and I'll analyze it for you."
    
    def _extract_ticker(self, message: str) -> Optional[str]:
        """Extract stock ticker from user message"""
        import re
        
        # Common patterns for tickers
        message_upper = message.upper()
        
        # Company name to ticker mapping (check this FIRST before direct ticker matches)
        company_map = {
            'APPLE': 'AAPL',
            'TESLA': 'TSLA',
            'MICROSOFT': 'MSFT',
            'GOOGLE': 'GOOGL',
            'ALPHABET': 'GOOGL',
            'AMAZON': 'AMZN',
            'FACEBOOK': 'META',
            'META': 'META',
            'NVIDIA': 'NVDA',
            'NETFLIX': 'NFLX',
            'DISNEY': 'DIS',
            'BOEING': 'BA',
            'AMD': 'AMD',
            'INTEL': 'INTC',
            'WALMART': 'WMT',
            'JPMORGAN': 'JPM',
            'BANK OF AMERICA': 'BAC',
            'VISA': 'V',
            'MASTERCARD': 'MA'
        }
        
        # Check company names first (longer matches)
        for company, ticker in sorted(company_map.items(), key=lambda x: len(x[0]), reverse=True):
            if company in message_upper:
                return ticker
        
        # Direct ticker mentions
        common_tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 
                         'AMD', 'NFLX', 'DIS', 'BA', 'JPM', 'BAC', 'WMT', 'INTC', 'MA']
        
        for ticker in common_tickers:
            if ticker in message_upper:
                return ticker
        
        # Pattern matching for ticker-like strings (2-5 uppercase letters)
        # Only match if it looks like a ticker (all caps, isolated word)
        pattern = r'\b([A-Z]{2,5})\b'
        matches = re.findall(pattern, message.upper())
        if matches:
            # Filter out common words that aren't tickers
            excluded = ['I', 'IN', 'ON', 'AT', 'TO', 'FOR', 'THE', 'AND', 'OR', 'BUT', 'IS', 'ARE', 'WAS', 'WERE']
            for match in matches:
                if match not in excluded:
                    return match
        
        return None
    
    def chat_with_context(self, user_message: str, analysis: Dict[str, Any], data: pd.DataFrame) -> str:
        """
        Chat with pre-loaded analysis context
        
        Args:
            user_message: User's question
            analysis: Pre-loaded analysis data
            data: Historical price data with indicators
            
        Returns:
            LLM response using the context
        """
        if not self.llm:
            return f"""❌ **Groq API Key Not Found**

To enable intelligent chat responses, please:
1. Get a free API key from https://console.groq.com/
2. Add it to your .env file: `GROQ_API_KEY=gsk_your_key_here`
3. Restart the app

Without the API key, I cannot provide natural conversational responses."""
        
        ticker = analysis['ticker']
        
        # Format historical data summary with more details
        rsi_val = f"{data['RSI'].iloc[-1]:.2f}" if 'RSI' in data.columns else 'N/A'
        sma20_val = f"${data['SMA_20'].iloc[-1]:.2f}" if 'SMA_20' in data.columns else 'N/A'
        sma50_val = f"${data['SMA_50'].iloc[-1]:.2f}" if 'SMA_50' in data.columns else 'N/A'
        
        data_summary = f"""
HISTORICAL DATA (Last 7 Days):
{data.tail(7)[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()}

TECHNICAL INDICATORS (Current):
- RSI: {rsi_val}
- SMA 20: {sma20_val}
- SMA 50: {sma50_val}
- Current Price: ${analysis['current_price']:.2f}
"""
        
        # Format analysis context
        analysis_context = self.format_analysis_for_llm(analysis)
        
        prompt = f"""{self.system_prompt}

I have performed a comprehensive analysis of {ticker}. Here's all the data:

{data_summary}

{analysis_context}

The user is asking: "{user_message}"

IMPORTANT: 
- Directly answer their specific question using the ACTUAL DATA above
- If they ask about historical prices (highest, lowest, yesterday, last week) - LOOK AT THE HISTORICAL DATA TABLE
- If they ask about volume - LOOK AT THE VOLUME COLUMN
- Do NOT give generic recommendations unless they specifically ask for advice
- Be specific with numbers, dates, and values from the data
- Keep responses concise and natural

Respond naturally as if you're having a conversation with an investor."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            error_msg = str(e)
            return f"""❌ **Error connecting to Groq API**

Error: {error_msg}

Please check:
1. Your GROQ_API_KEY in .env file is correct
2. You have internet connection
3. Your Groq account has available credits

Get help at: https://console.groq.com/"""
    
    def _generate_fallback_response(self, analysis: Dict[str, Any]) -> str:
        """Generate response without LLM"""
        decision = analysis['decision']
        ticker = analysis['ticker']
        current_price = analysis['current_price']
        
        action = decision['recommendation']
        strength = decision['strength']
        confidence = decision['confidence'] * 100
        
        # Validate target range
        target_low = decision['target_range']['target_low']
        target_high = decision['target_range']['target_high']
        
        # If target range is identical or too close, use a reasonable spread
        if abs(target_high - target_low) < 0.01:
            spread = current_price * 0.03  # 3% spread
            target_low = current_price - spread
            target_high = current_price + spread
        
        response = f"""Based on my comprehensive AI analysis of {ticker}:

📊 **Current Price**: ${current_price:.2f}

🎯 **My Recommendation**: {strength} {action}
**Confidence**: {confidence:.1f}%

**Why?**
"""
        
        # Add reasoning based on components
        forecast = analysis['forecast']
        trend = forecast.get('trend', 'Sideways')
        trend_prob = forecast.get('trend_probability', 0.5)  # This is probability of UPWARD movement
        
        # Display the correct probability based on trend
        if trend == 'Upward':
            # Upward trend: show upward probability
            response += f"\n✅ The AI forecast predicts an upward trend with {trend_prob*100:.0f}% probability"
        elif trend == 'Downward':
            # Downward trend: show downward probability (inverse of upward)
            down_prob = (1 - trend_prob) * 100
            response += f"\n⚠️ The AI forecast predicts a downward trend with {down_prob:.0f}% probability"
        else:
            # Sideways: show it's neutral
            response += f"\n➡️ The AI forecast suggests a sideways trend (neutral movement)"
        
        sentiment = analysis['sentiment']
        if sentiment.get('overall_score', 0) > 0.1:
            response += f"\n✅ Market sentiment is positive ({sentiment.get('overall_label', 'Neutral')})"
        elif sentiment.get('overall_score', 0) < -0.1:
            response += f"\n⚠️ Market sentiment is negative ({sentiment.get('overall_label', 'Neutral')})"
        else:
            response += f"\n➡️ Market sentiment is neutral"
        
        rl_agent = analysis['rl_agent']
        response += f"\n🤖 RL Agent recommends: {rl_agent.get('action', 'HOLD')}"
        
        response += f"""

📈 **Target Range**: ${target_low:.2f} - ${target_high:.2f}
🛡️ **Risk Level**: {decision['risk_assessment']['level']}

⚠️ **Disclaimer**: This analysis is generated by AI for educational purposes only and does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.
"""
        
        return response


def main():
    """Main function for interactive chat"""
    print("🤖 AI Trading Agent - Conversational Mode")
    print("=" * 60)
    print("Ask me about any stock! Examples:")
    print("  • 'Should I invest in Apple?'")
    print("  • 'What's the outlook for Tesla?'")
    print("  • 'Is Microsoft a good buy right now?'")
    print("  • 'What's the risk level of NVDA?'")
    print("  • 'Compare Apple and Microsoft'")
    print("  • 'Explain what RSI means'")
    print("\nType 'help' for more options, 'quit' or 'exit' to end")
    print("=" * 60)
    
    agent = TradingChatAgent(use_timegan=True)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 Agent: Goodbye! Happy investing! 📈")
                break
            
            if user_input.lower() == 'help':
                print("\n🤖 Agent: Here's what I can help you with:\n")
                print("📊 **Stock Analysis Questions:**")
                print("  • 'Should I invest in [stock]?'")
                print("  • 'What's the outlook for [stock]?'")
                print("  • 'Is [stock] a good buy right now?'")
                print("  • 'What's the risk level of [stock]?'")
                print("  • 'Analyze AAPL' (or any ticker)")
                print("\n📈 **Market Insights:**")
                print("  • 'Compare Apple and Microsoft'")
                print("  • 'What's the current trend for [stock]?'")
                print("  • 'Should I sell my [stock] shares?'")
                print("\n📚 **Educational Questions:**")
                print("  • 'What does RSI mean?'")
                print("  • 'Explain moving averages'")
                print("  • 'What is a good P/E ratio?'")
                print("  • 'How does sentiment analysis work?'")
                print("\n💡 **Tips:**")
                print("  • I can recognize company names (Apple, Tesla, etc.)")
                print("  • Analysis is cached for 5 minutes for faster responses")
                print("  • First analysis may take 1-2 minutes (model training)")
                continue
            
            print("\n🤖 Agent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n🤖 Agent: Goodbye! Happy investing! 📈")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n🤖 Agent: I encountered an error: {e}")


if __name__ == "__main__":
    main()
