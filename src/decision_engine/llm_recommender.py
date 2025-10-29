"""
LLM-Based Recommendation Generator
Uses Large Language Models to generate trading recommendations
"""

import os
import logging
from typing import Dict, Optional
import json

logger = logging.getLogger(__name__)


class LLMRecommender:
    """
    LLM-based Trading Recommendation Generator
    Uses GPT/Claude to generate natural language recommendations
    """
    
    def __init__(self, config: dict):
        """
        Initialize LLM Recommender
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.provider = config.get('provider', 'groq')
        self.model_name = config.get('llm_model', 'llama-3.1-8b-instant')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 2000)
        
        # Try to initialize LLM client
        self.client = None
        self.use_mock = True
        
        try:
            # Try Groq first (fastest)
            if self.provider == 'groq':
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    from langchain_groq import ChatGroq
                    from langchain.schema import HumanMessage, SystemMessage
                    self.client = ChatGroq(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        groq_api_key=api_key
                    )
                    self.use_mock = False
                    self.client_type = 'groq'
                    logger.info(f"✅ LLM Recommender initialized with Groq ({self.model_name})")
                else:
                    logger.warning("No GROQ_API_KEY found, trying OpenAI...")
                    self.provider = 'openai'
            
            # Fallback to OpenAI
            if self.provider == 'openai' or (self.provider == 'groq' and self.use_mock):
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=api_key)
                    self.use_mock = False
                    self.client_type = 'openai'
                    self.model_name = config.get('openai_model', 'gpt-3.5-turbo')
                    logger.info(f"LLM Recommender initialized with OpenAI ({self.model_name})")
                else:
                    logger.warning("No API keys found, using mock LLM")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}. Using mock generation.")
    
    def generate_recommendation(self, 
                               ticker: str,
                               forecast_result: Dict,
                               sentiment_result: Dict,
                               rl_result: Dict,
                               current_price: float) -> Dict:
        """
        Generate comprehensive trading recommendation using LLM
        
        Args:
            ticker: Stock ticker
            forecast_result: Forecasting agent output
            sentiment_result: Sentiment agent output
            rl_result: RL agent output
            current_price: Current stock price
            
        Returns:
            Dictionary with LLM-generated recommendation
        """
        logger.info("Generating LLM-based recommendation...")
        
        # Create context for LLM
        context = self._create_context(
            ticker, forecast_result, sentiment_result, rl_result, current_price
        )
        
        if self.use_mock:
            return self._mock_generation(context)
        
        try:
            # Generate recommendation using LLM
            if hasattr(self, 'client_type') and self.client_type == 'groq':
                # Use LangChain Groq
                from langchain.schema import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=self._get_system_prompt()),
                    HumanMessage(content=context)
                ]
                response = self.client.invoke(messages)
                recommendation_text = response.content
            else:
                # Use OpenAI
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": context
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                recommendation_text = response.choices[0].message.content
            
            # Parse recommendation
            parsed = self._parse_recommendation(recommendation_text)
            
            logger.info(f"LLM generated: {parsed['action']} recommendation")
            return parsed
            
        except Exception as e:
            logger.error(f"Error generating LLM recommendation: {e}")
            return self._mock_generation(context)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an expert financial analyst and trading advisor with deep knowledge of:
- Technical analysis and chart patterns
- Market sentiment analysis
- Risk management and portfolio optimization
- Machine learning predictions in finance

Your task is to analyze the provided data and generate a clear, actionable trading recommendation.

Output your recommendation in the following JSON format:
{
    "action": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation of your recommendation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks": ["risk1", "risk2"],
    "target_price": price value,
    "stop_loss": price value,
    "time_horizon": "short-term" or "medium-term" or "long-term",
    "position_size": "small" or "medium" or "large"
}

Be specific, data-driven, and consider both opportunities and risks."""
    
    def _create_context(self, ticker: str, forecast_result: Dict,
                       sentiment_result: Dict, rl_result: Dict,
                       current_price: float) -> str:
        """Create context for LLM"""
        
        context = f"""Analyze the following data for {ticker} and provide a trading recommendation:

CURRENT STATUS:
- Ticker: {ticker}
- Current Price: ${current_price:.2f}

FORECASTING ANALYSIS:
- Trend: {forecast_result.get('trend', 'Unknown')}
- Trend Probability: {forecast_result.get('trend_probability', 0):.1%}
- Forecast Confidence: {forecast_result.get('confidence', 0):.1%}
- Generation Method: {forecast_result.get('generation_method', 'LSTM')}
"""
        
        if forecast_result.get('mean_forecast'):
            expected_price = forecast_result['mean_forecast'][-1]
            change = ((expected_price - current_price) / current_price) * 100
            context += f"- Expected Price ({forecast_result.get('forecast_days', 5)} days): ${expected_price:.2f} ({change:+.1f}%)\n"
        
        context += f"""
SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_result.get('overall_label', 'Neutral')}
- Sentiment Score: {sentiment_result.get('overall_score', 0):+.2f}
- Confidence: {sentiment_result.get('confidence', 0):.1%}
- Articles Analyzed: {sentiment_result.get('article_count', 0)}
- Summary: {sentiment_result.get('summary', 'No summary available')}

RL TRADING AGENT:
- Recommended Action: {rl_result.get('action', 'HOLD')}
- Confidence: {rl_result.get('confidence', 0):.1%}
- Position Size: {rl_result.get('position_size', 0)} shares
- Rationale: {rl_result.get('rationale', 'No rationale available')}

Based on this comprehensive analysis, provide your expert trading recommendation."""
        
        return context
    
    def _parse_recommendation(self, text: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate and normalize
                return {
                    'action': parsed.get('action', 'HOLD').upper(),
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'reasoning': parsed.get('reasoning', text),
                    'key_factors': parsed.get('key_factors', []),
                    'risks': parsed.get('risks', []),
                    'target_price': float(parsed.get('target_price', 0)),
                    'stop_loss': float(parsed.get('stop_loss', 0)),
                    'time_horizon': parsed.get('time_horizon', 'medium-term'),
                    'position_size': parsed.get('position_size', 'medium'),
                    'full_text': text,
                    'generation_method': 'LLM'
                }
        except Exception as e:
            logger.warning(f"Could not parse JSON from LLM response: {e}")
        
        # Fallback: extract action from text
        text_upper = text.upper()
        if 'STRONG BUY' in text_upper or 'STRONGLY RECOMMEND BUY' in text_upper:
            action = 'BUY'
            confidence = 0.8
        elif 'BUY' in text_upper:
            action = 'BUY'
            confidence = 0.6
        elif 'STRONG SELL' in text_upper or 'STRONGLY RECOMMEND SELL' in text_upper:
            action = 'SELL'
            confidence = 0.8
        elif 'SELL' in text_upper:
            action = 'SELL'
            confidence = 0.6
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': text,
            'key_factors': [],
            'risks': [],
            'target_price': 0,
            'stop_loss': 0,
            'time_horizon': 'medium-term',
            'position_size': 'medium',
            'full_text': text,
            'generation_method': 'LLM'
        }
    
    def _mock_generation(self, context: str) -> Dict:
        """Generate mock recommendation when LLM unavailable"""
        logger.info("Using mock LLM generation")
        
        # Simple rule-based generation
        if 'Upward' in context and 'Bullish' in context:
            action = 'BUY'
            confidence = 0.75
            reasoning = "Multiple positive signals: upward price trend forecast and bullish market sentiment suggest a buying opportunity."
            key_factors = ["Positive price forecast", "Bullish sentiment", "Technical indicators support uptrend"]
            risks = ["Market volatility", "Unexpected news events"]
        elif 'Downward' in context and 'Bearish' in context:
            action = 'SELL'
            confidence = 0.75
            reasoning = "Multiple negative signals: downward price trend forecast and bearish market sentiment suggest reducing exposure."
            key_factors = ["Negative price forecast", "Bearish sentiment", "Technical indicators show weakness"]
            risks = ["Potential reversal", "Missing recovery opportunity"]
        else:
            action = 'HOLD'
            confidence = 0.6
            reasoning = "Mixed signals from different indicators suggest maintaining current position and monitoring for clearer trends."
            key_factors = ["Mixed technical signals", "Neutral sentiment", "Uncertain trend direction"]
            risks = ["Opportunity cost", "Market direction unclear"]
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'key_factors': key_factors,
            'risks': risks,
            'target_price': 0,
            'stop_loss': 0,
            'time_horizon': 'medium-term',
            'position_size': 'medium',
            'full_text': reasoning,
            'generation_method': 'Mock'
        }
    
    def generate_detailed_report(self, recommendation: Dict, ticker: str) -> str:
        """Generate detailed markdown report"""
        
        report = f"""# Trading Recommendation Report for {ticker}

## Executive Summary
**Recommendation:** {recommendation['action']}  
**Confidence Level:** {recommendation['confidence']:.0%}  
**Time Horizon:** {recommendation['time_horizon']}  
**Suggested Position Size:** {recommendation['position_size']}

## Analysis

{recommendation['reasoning']}

## Key Factors
"""
        
        for i, factor in enumerate(recommendation.get('key_factors', []), 1):
            report += f"{i}. {factor}\n"
        
        report += "\n## Risk Considerations\n"
        for i, risk in enumerate(recommendation.get('risks', []), 1):
            report += f"{i}. {risk}\n"
        
        if recommendation.get('target_price', 0) > 0:
            report += f"\n## Price Targets\n"
            report += f"- **Target Price:** ${recommendation['target_price']:.2f}\n"
            report += f"- **Stop Loss:** ${recommendation['stop_loss']:.2f}\n"
        
        report += "\n---\n*Generated by LLM-powered AI Trading System*\n"
        
        return report
