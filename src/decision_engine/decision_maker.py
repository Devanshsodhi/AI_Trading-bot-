"""
Decision Engine
Combines outputs from all agents to generate final recommendations
"""

import numpy as np
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Final Decision Engine
    Combines forecasting, sentiment, and RL agent outputs
    """
    
    def __init__(self, config: dict):
        """
        Initialize Decision Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.sentiment_weight = config.get('sentiment_weight', 0.3)
        self.forecast_weight = config.get('forecast_weight', 0.4)
        self.rl_weight = config.get('rl_weight', 0.3)
        
        logger.info("Decision Engine initialized")
    
    def make_decision(self, 
                     forecast_result: Dict,
                     sentiment_result: Dict,
                     rl_result: Dict,
                     ticker: str,
                     current_price: float) -> Dict:
        """
        Generate final trading recommendation
        
        Args:
            forecast_result: Output from forecasting agent
            sentiment_result: Output from sentiment agent
            rl_result: Output from RL trading agent
            ticker: Stock ticker
            current_price: Current stock price
            
        Returns:
            Final recommendation dictionary
        """
        logger.info("Generating final recommendation...")
        
        # Extract key metrics
        forecast_score = self._score_forecast(forecast_result)
        sentiment_score = sentiment_result.get('overall_score', 0)
        rl_action = rl_result.get('action', 'HOLD')
        
        # Convert RL action to score
        rl_score = self._action_to_score(rl_action)
        
        # Calculate weighted combined score
        combined_score = (
            self.forecast_weight * forecast_score +
            self.sentiment_weight * sentiment_score +
            self.rl_weight * rl_score
        )
        
        # Determine final action
        if combined_score > 0.3:
            final_action = 'BUY'
            action_strength = 'Strong' if combined_score > 0.6 else 'Moderate'
        elif combined_score < -0.3:
            final_action = 'SELL'
            action_strength = 'Strong' if combined_score < -0.6 else 'Moderate'
        else:
            final_action = 'HOLD'
            action_strength = 'Neutral'
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            forecast_result, sentiment_result, rl_result, combined_score
        )
        
        # Generate detailed explanation
        explanation = self._generate_explanation(
            ticker, current_price, final_action, action_strength,
            forecast_result, sentiment_result, rl_result, combined_score, confidence
        )
        
        # Generate risk assessment
        risk_assessment = self._assess_risk(
            forecast_result, sentiment_result, combined_score
        )
        
        # Calculate target price range
        target_range = self._calculate_target_range(
            current_price, forecast_result, combined_score
        )
        
        result = {
            'recommendation': final_action,
            'strength': action_strength,
            'confidence': float(confidence),
            'combined_score': float(combined_score),
            'explanation': explanation,
            'risk_assessment': risk_assessment,
            'target_range': target_range,
            'position_size': rl_result.get('position_size', 0),
            'timestamp': datetime.now().isoformat(),
            'components': {
                'forecast_score': float(forecast_score),
                'sentiment_score': float(sentiment_score),
                'rl_score': float(rl_score),
            }
        }
        
        logger.info(f"Final recommendation: {action_strength} {final_action} (confidence: {confidence:.0%})")
        return result
    
    def _score_forecast(self, forecast_result: Dict) -> float:
        """Convert forecast to a score between -1 and 1"""
        trend = forecast_result.get('trend', 'Sideways')
        trend_probability = forecast_result.get('trend_probability', 0.5)
        
        if trend == 'Upward':
            score = (trend_probability - 0.5) * 2  # Map [0.5, 1] to [0, 1]
        elif trend == 'Downward':
            score = -(trend_probability - 0.5) * 2  # Map [0.5, 1] to [0, -1]
        else:
            score = 0
        
        return score
    
    def _action_to_score(self, action: str) -> float:
        """Convert RL action to score"""
        action_scores = {
            'BUY': 1.0,
            'HOLD': 0.0,
            'SELL': -1.0,
        }
        return action_scores.get(action, 0.0)
    
    def _calculate_confidence(self, forecast_result: Dict, sentiment_result: Dict,
                             rl_result: Dict, combined_score: float) -> float:
        """Calculate overall confidence in the recommendation"""
        confidences = []
        
        # Forecast confidence
        forecast_conf = forecast_result.get('confidence', 0.5)
        confidences.append(forecast_conf)
        
        # Sentiment confidence
        sentiment_conf = sentiment_result.get('confidence', 0.5)
        confidences.append(sentiment_conf)
        
        # RL confidence
        rl_conf = rl_result.get('confidence', 0.5)
        confidences.append(rl_conf)
        
        # Agreement bonus: if all agents agree, boost confidence
        forecast_score = self._score_forecast(forecast_result)
        sentiment_score = sentiment_result.get('overall_score', 0)
        rl_score = self._action_to_score(rl_result.get('action', 'HOLD'))
        
        scores = [forecast_score, sentiment_score, rl_score]
        agreement = 1 - np.std(scores) / 2  # Higher agreement = lower std
        
        # Combined confidence
        base_confidence = np.mean(confidences)
        final_confidence = (base_confidence + agreement) / 2
        
        return min(final_confidence, 0.99)
    
    def _generate_explanation(self, ticker: str, current_price: float,
                             action: str, strength: str,
                             forecast_result: Dict, sentiment_result: Dict,
                             rl_result: Dict, combined_score: float,
                             confidence: float) -> str:
        """Generate human-readable explanation"""
        
        explanation_parts = []
        
        # Header
        explanation_parts.append(
            f"## Recommendation: {strength} {action}\n"
            f"**Confidence:** {confidence:.0%} | **Score:** {combined_score:+.2f}\n"
        )
        
        # Current status
        explanation_parts.append(
            f"### Current Status\n"
            f"**{ticker}** is trading at **${current_price:.2f}**.\n"
        )
        
        # Forecast analysis
        trend = forecast_result.get('trend', 'Unknown')
        trend_prob = forecast_result.get('trend_probability', 0)
        forecast_days = forecast_result.get('forecast_days', 5)
        mean_forecast = forecast_result.get('mean_forecast', [])
        
        if mean_forecast:
            expected_price = mean_forecast[-1]
            price_change = ((expected_price - current_price) / current_price) * 100
            
            explanation_parts.append(
                f"### 📈 Forecast Analysis\n"
                f"- **Trend:** {trend} with {trend_prob:.0%} probability\n"
                f"- **{forecast_days}-day forecast:** ${expected_price:.2f} ({price_change:+.1f}%)\n"
                f"- **Confidence:** {forecast_result.get('confidence', 0):.0%}\n"
            )
        
        # Sentiment analysis
        sentiment_label = sentiment_result.get('overall_label', 'Neutral')
        sentiment_score = sentiment_result.get('overall_score', 0)
        article_count = sentiment_result.get('article_count', 0)
        
        explanation_parts.append(
            f"### 💭 Sentiment Analysis\n"
            f"- **Overall sentiment:** {sentiment_label} ({sentiment_score:+.2f})\n"
            f"- **Sources analyzed:** {article_count} articles\n"
            f"- **Summary:** {sentiment_result.get('summary', 'No summary available')}\n"
        )
        
        # RL agent recommendation
        rl_action = rl_result.get('action', 'HOLD')
        rl_confidence = rl_result.get('confidence', 0)
        position_size = rl_result.get('position_size', 0)
        
        explanation_parts.append(
            f"### 🤖 RL Agent Analysis\n"
            f"- **Recommended action:** {rl_action}\n"
            f"- **Confidence:** {rl_confidence:.0%}\n"
            f"- **Suggested position:** {position_size} shares\n"
            f"- **Rationale:** {rl_result.get('rationale', 'No rationale available')}\n"
        )
        
        # Final advice
        if action == 'BUY':
            advice = (
                f"### ✅ Final Advice\n"
                f"The analysis suggests a **buying opportunity** for {ticker}. "
                f"Consider entering a position with appropriate risk management. "
                f"Set stop-loss orders to protect against downside risk."
            )
        elif action == 'SELL':
            advice = (
                f"### ⚠️ Final Advice\n"
                f"The analysis suggests **reducing exposure** to {ticker}. "
                f"Consider taking profits or cutting losses. "
                f"Monitor market conditions for re-entry opportunities."
            )
        else:
            advice = (
                f"### ⏸️ Final Advice\n"
                f"The analysis suggests **maintaining current position** in {ticker}. "
                f"Wait for clearer signals before making significant changes. "
                f"Continue monitoring market conditions."
            )
        
        explanation_parts.append(advice)
        
        # Disclaimer
        explanation_parts.append(
            "\n---\n"
            "*This recommendation is generated by AI and is for informational purposes only. "
            "It does not constitute financial advice. Always conduct your own research and "
            "consult with a qualified financial advisor before making investment decisions.*"
        )
        
        return '\n'.join(explanation_parts)
    
    def _assess_risk(self, forecast_result: Dict, sentiment_result: Dict,
                    combined_score: float) -> Dict:
        """Assess risk level of the recommendation"""
        
        # Calculate volatility from forecast
        std_forecast = forecast_result.get('std_forecast', [0])
        mean_forecast = forecast_result.get('mean_forecast', [100])
        
        if mean_forecast:
            avg_volatility = np.mean(np.array(std_forecast) / np.array(mean_forecast))
        else:
            avg_volatility = 0.1
        
        # Sentiment consistency
        sentiment_dist = sentiment_result.get('sentiment_distribution', {})
        sentiment_consistency = max(sentiment_dist.values()) if sentiment_dist else 0.5
        
        # Calculate risk score (0 = low risk, 1 = high risk)
        risk_score = (
            0.5 * avg_volatility +
            0.3 * (1 - sentiment_consistency) +
            0.2 * abs(combined_score)
        )
        
        # Classify risk level
        if risk_score < 0.3:
            risk_level = 'Low'
            risk_description = 'Market conditions are relatively stable with consistent signals.'
        elif risk_score < 0.6:
            risk_level = 'Moderate'
            risk_description = 'Some uncertainty in market conditions. Use appropriate position sizing.'
        else:
            risk_level = 'High'
            risk_description = 'Significant uncertainty or volatility. Consider reducing position size.'
        
        return {
            'level': risk_level,
            'score': float(risk_score),
            'description': risk_description,
        }
    
    def _calculate_target_range(self, current_price: float,
                               forecast_result: Dict, combined_score: float) -> Dict:
        """Calculate target price range"""
        # Extract forecasting results - use MEAN forecast, not bounds
        mean_forecast = forecast_result.get('mean_forecast', [current_price])
        lower_bound = forecast_result.get('lower_bound', [current_price])
        upper_bound = forecast_result.get('upper_bound', [current_price])
        
        # Calculate expected future price
        expected_price = mean_forecast[-1] if mean_forecast and len(mean_forecast) > 0 else current_price
        
        # Calculate realistic target range based on historical volatility
        std_forecast = forecast_result.get('std_forecast', [current_price * 0.02])
        avg_vol = sum(std_forecast) / len(std_forecast) if std_forecast and len(std_forecast) > 0 else current_price * 0.03
        
        # Target range = expected ± volatility
        target_low = expected_price - avg_vol
        target_high = expected_price + avg_vol
        
        # Ensure targets are sensible relative to current price
        # Don't let targets be too close to current (min 2% movement)
        min_move = current_price * 0.02
        if abs(expected_price - current_price) < min_move:
            if combined_score > 0:  # Bullish
                target_low = current_price - min_move
                target_high = current_price + min_move * 2
            elif combined_score < 0:  # Bearish
                target_low = current_price - min_move * 2
                target_high = current_price + min_move
            else:  # Neutral
                target_low = current_price - min_move
                target_high = current_price + min_move
        
        # Calculate stop loss and take profit levels - MUST BE LOGICAL
        if combined_score > 0:  # Bullish (BUY)
            stop_loss = current_price * 0.95  # 5% below current (cut losses if drops)
            take_profit = max(target_high, current_price * 1.05)  # At least 5% upside
        elif combined_score < 0:  # Bearish (SELL)
            stop_loss = current_price * 1.05  # If it goes UP 5% after selling, you missed out
            take_profit = min(target_low, current_price * 0.95)  # Expect at least 5% down
        else:  # Neutral (HOLD)
            stop_loss = current_price * 0.95  # 5% downside protection
            take_profit = current_price * 1.05  # 5% upside target
        
        return {
            'current': float(current_price),
            'target_low': float(target_low),
            'target_high': float(target_high),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
        }
    
    def generate_summary(self, decision: Dict) -> str:
        """Generate a concise summary of the decision"""
        
        action = decision['recommendation']
        strength = decision['strength']
        confidence = decision['confidence']
        
        summary = (
            f"{strength} {action} recommendation with {confidence:.0%} confidence. "
            f"Risk level: {decision['risk_assessment']['level']}. "
        )
        
        if action != 'HOLD':
            target = decision['target_range']
            summary += (
                f"Target range: ${target['target_low']:.2f} - ${target['target_high']:.2f}. "
                f"Stop loss: ${target['stop_loss']:.2f}."
            )
        
        return summary
