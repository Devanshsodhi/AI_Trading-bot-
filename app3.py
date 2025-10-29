import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from datetime import datetime
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
for logger_name in ['transformers', 'urllib3', 'httpx', 'tensorflow', 'torch']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from src.data.data_loader import DataLoader
from src.data.technical_indicators import TechnicalIndicators
from src.forecasting_agent import TimeGANForecaster
from src.sentiment_agent import SentimentAgent
from src.trading_agent import TradingAgent
from src.decision_engine import DecisionEngine
from chat_agent import TradingChatAgent

st.set_page_config(page_title="AI Trading Agent", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_agent' not in st.session_state:
    st.session_state.chat_agent = None

def create_price_chart(data, ticker):
    """Create interactive price chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add moving averages if available
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ))
    
    fig.update_layout(
        title=f'{ticker} Price Chart with Technical Indicators',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_volume_chart(data):
    """Create volume chart"""
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
              for i in range(len(data))]
    
    fig = go.Figure(data=[go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        name='Volume'
    )])
    
    fig.update_layout(
        title='Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark',
        height=300
    )
    
    return fig

def create_technical_indicators_chart(data):
    """Create technical indicators chart"""
    fig = go.Figure()
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='cyan')
        ))
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        xaxis_title='Date',
        template='plotly_dark',
        height=300
    )
    
    return fig

def create_forecast_chart(forecast_result, current_price, ticker):
    """Create forecast visualization"""
    fig = go.Figure()
    
    days = len(forecast_result['mean_forecast'])
    x_axis = list(range(1, days + 1))
    
    # Mean forecast
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=forecast_result['mean_forecast'],
        name='Mean Forecast',
        line=dict(color='yellow', width=3)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=forecast_result['upper_bound'],
        name='Upper Bound (95%)',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=forecast_result['lower_bound'],
        name='Lower Bound (5%)',
        line=dict(color='red', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(100, 100, 100, 0.2)'
    ))
    
    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="white",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    fig.update_layout(
        title=f'{ticker} - AI Forecast (Next {days} Days)',
        yaxis_title='Price ($)',
        xaxis_title='Days Ahead',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_scenarios_chart(forecast_result, ticker):
    """Create multiple scenarios visualization"""
    if 'scenarios' not in forecast_result:
        return None
    
    fig = go.Figure()
    
    scenarios = forecast_result['scenarios'][:20]  # Show first 20 scenarios
    days = len(scenarios[0])
    x_axis = list(range(1, days + 1))
    
    for i, scenario in enumerate(scenarios):
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=scenario,
            name=f'Scenario {i+1}',
            line=dict(width=1),
            opacity=0.3,
            showlegend=False
        ))
    
    # Add mean
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=forecast_result['mean_forecast'],
        name='Mean Forecast',
        line=dict(color='yellow', width=3)
    ))
    
    fig.update_layout(
        title=f'{ticker} - Generated Scenarios (20 out of 100)',
        yaxis_title='Price ($)',
        xaxis_title='Days Ahead',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_sentiment_chart(sentiment_result):
    """Create sentiment analysis visualization"""
    if 'article_sentiments' not in sentiment_result:
        return None
    
    sentiments = sentiment_result['article_sentiments'][:10]  # Top 10
    
    fig = go.Figure(data=[
        go.Bar(
            x=[s['score'] for s in sentiments],
            y=[s['title'][:50] + '...' if len(s['title']) > 50 else s['title'] for s in sentiments],
            orientation='h',
            marker=dict(
                color=[s['score'] for s in sentiments],
                colorscale='RdYlGn',
                cmin=-1,
                cmax=1
            )
        )
    ])
    
    fig.update_layout(
        title='News Sentiment Analysis (Top 10 Articles)',
        xaxis_title='Sentiment Score',
        yaxis_title='Article',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_rl_training_chart(rl_agent):
    """Create RL training progress visualization"""
    # This is a placeholder - in production, you'd log training metrics
    epochs = list(range(1, 51))
    rewards = [np.random.normal(100 + i*2, 20) for i in epochs]  # Simulated
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=rewards,
        name='Episode Reward',
        line=dict(color='cyan', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))
    
    fig.update_layout(
        title='RL Agent Training Progress',
        yaxis_title='Cumulative Reward',
        xaxis_title='Training Episode',
        template='plotly_dark',
        height=300
    )
    
    return fig

def create_model_comparison_chart(forecast_result, rl_result):
    """Create model comparison chart"""
    models = ['TimeGAN', 'RL Agent', 'Sentiment']
    
    # Extract confidence/scores
    timegan_conf = forecast_result.get('confidence', 0.5) * 100
    rl_conf = rl_result.get('confidence', 0.5) * 100
    sentiment_conf = 70  # Placeholder
    
    confidences = [timegan_conf, rl_conf, sentiment_conf]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            marker=dict(
                color=confidences,
                colorscale='Viridis'
            ),
            text=[f'{c:.1f}%' for c in confidences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Model Confidence Comparison',
        yaxis_title='Confidence (%)',
        template='plotly_dark',
        height=300
    )
    
    return fig

def generate_contextual_answer(question: str, analysis: Dict) -> str:
    """Generate contextual answers based on the current analysis"""
    question_lower = question.lower()
    ticker = analysis['ticker']
    
    # Price-related questions
    if any(word in question_lower for word in ['price', 'cost', 'trading at']):
        if 'yesterday' in question_lower or 'last day' in question_lower:
            yesterday_price = float(analysis['data']['Close'].iloc[-2])
            current_price = analysis['current_price']
            change = current_price - yesterday_price
            change_pct = (change / yesterday_price) * 100
            return f"""**Yesterday's Price for {ticker}:**

📊 **Yesterday's Close:** ${yesterday_price:.2f}
📊 **Today's Price:** ${current_price:.2f}
📈 **Change:** ${change:+.2f} ({change_pct:+.2f}%)"""
        else:
            return f"**Current Price:** ${analysis['current_price']:.2f}"
    
    # Recommendation questions
    if any(word in question_lower for word in ['should i buy', 'should i invest', 'recommend', 'buy or sell']):
        decision = analysis['decision']
        return f"""**Investment Recommendation for {ticker}:**

🎯 **{decision['strength']} {decision['recommendation']}**
**Confidence:** {decision['confidence']*100:.1f}%

**Why?**
- **AI Forecast:** {analysis['forecast'].get('trend', 'Unknown')} trend with {analysis['forecast'].get('trend_probability', 0)*100:.0f}% probability
- **Sentiment:** {analysis['sentiment'].get('overall_label', 'Neutral')} ({analysis['sentiment'].get('overall_score', 0):+.2f})
- **RL Agent:** {analysis['rl_agent'].get('action', 'HOLD')}

**Target Range:** ${decision['target_range']['target_low']:.2f} - ${decision['target_range']['target_high']:.2f}
**Risk Level:** {decision['risk_assessment']['level']}"""
    
    # Risk questions
    if any(word in question_lower for word in ['risk', 'risky', 'safe', 'danger']):
        risk = analysis['decision']['risk_assessment']
        return f"""**Risk Assessment for {ticker}:**

**Level:** {risk['level']}
**Score:** {risk['score']:.2f}
**Description:** {risk['description']}

**Risk Management:**
- Stop Loss: ${analysis['decision']['target_range']['stop_loss']:.2f}
- Take Profit: ${analysis['decision']['target_range']['take_profit']:.2f}"""
    
    # Forecast questions
    if any(word in question_lower for word in ['forecast', 'prediction', 'future', 'trend', 'outlook']):
        forecast = analysis['forecast']
        trend = forecast.get('trend', 'Sideways')
        prob = forecast.get('trend_probability', 0.5)
        
        if trend == 'Upward':
            trend_desc = f"upward with {prob*100:.0f}% probability"
        elif trend == 'Downward':
            trend_desc = f"downward with {(1-prob)*100:.0f}% probability"
        else:
            trend_desc = "sideways (neutral)"
        
        return f"""**AI Forecast for {ticker}:**

**Trend:** {trend_desc}
**Confidence:** {forecast.get('confidence', 0)*100:.1f}%

**5-Day Price Prediction:**
- **Mean:** ${forecast['mean_forecast'][-1]:.2f}
- **Optimistic:** ${forecast['upper_bound'][-1]:.2f}
- **Pessimistic:** ${forecast['lower_bound'][-1]:.2f}

The AI analyzed 100 possible scenarios to generate this forecast."""
    
    # Sentiment questions
    if any(word in question_lower for word in ['sentiment', 'news', 'feeling', 'mood']):
        sentiment = analysis['sentiment']
        return f"""**Market Sentiment for {ticker}:**

**Overall:** {sentiment.get('overall_label', 'Neutral')}
**Score:** {sentiment.get('overall_score', 0):+.2f} (range: -1 to +1)
**Articles Analyzed:** {sentiment.get('article_count', 0)}

**Summary:** {sentiment.get('summary', 'No summary available')}

Sentiment is analyzed from financial news using FinBERT, a model trained specifically on financial text."""
    
    # Technical indicator questions
    if any(word in question_lower for word in ['rsi', 'relative strength', 'indicator']):
        data = analysis['data']
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
        
        if rsi:
            if rsi > 70:
                rsi_desc = "Overbought (potential sell signal)"
            elif rsi < 30:
                rsi_desc = "Oversold (potential buy signal)"
            else:
                rsi_desc = "Neutral"
            
            return f"""**RSI (Relative Strength Index) for {ticker}:**

**Current RSI:** {rsi:.2f}
**Interpretation:** {rsi_desc}

**What is RSI?**
RSI measures momentum on a 0-100 scale:
- Above 70 = Overbought (stock may be overvalued)
- Below 30 = Oversold (stock may be undervalued)
- 30-70 = Neutral zone"""
        else:
            return "RSI data not available for this stock."
    
    # Volume questions
    if 'volume' in question_lower:
        data = analysis['data']
        volume = int(data['Volume'].iloc[-1])
        avg_volume = int(data['Volume'].tail(20).mean())
        
        return f"""**Trading Volume for {ticker}:**

**Today's Volume:** {volume:,}
**20-Day Average:** {avg_volume:,}
**Difference:** {((volume - avg_volume) / avg_volume * 100):+.1f}%

{'📈 Higher than average - Strong interest' if volume > avg_volume else '📉 Lower than average - Weak interest'}"""
    
    # Default: use chat agent
    return f"I can answer questions about {ticker}'s price, forecast, sentiment, risk, and technical indicators. What would you like to know?"

def analyze_stock(ticker, days=30):
    """Perform comprehensive stock analysis"""
    try:
        # Fast + accurate configuration
        config = {
            'hidden_dim': 128,
            'noise_dim': 64,
            'epochs': 60,
            'batch_size': 32,
            'seq_len': 60,
            'learning_rate': 0.001,
            'patience': 15,
            'min_delta': 0.0001,
        }
        
        with st.spinner('📊 Loading market data...'):
            data_loader = DataLoader(ticker=ticker)
            data = data_loader.fetch_data()
            data = TechnicalIndicators.add_all_indicators(data)
            current_price = float(data['Close'].iloc[-1])
        
        with st.spinner('⚡ Training TimeGAN model (~90 seconds)...'):
            forecaster = TimeGANForecaster(config, ticker=ticker)
            training_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            forecaster.train(training_data, force_retrain=False)
            forecast_result = forecaster.generate_scenarios(num_scenarios=200, forecast_days=7)
        
        with st.spinner('💭 Analyzing sentiment...'):
            sentiment_analyzer = SentimentAgent(config={'model_name': 'ProsusAI/finbert', 'max_articles': 20})
            sentiment_result = sentiment_analyzer.analyze_ticker(ticker)
        
        with st.spinner('🎮 Training RL agent...'):
            rl_agent = TradingAgent(
                rl_config={'algorithm': 'PPO', 'learning_rate': 0.0001, 'total_timesteps': 50000},
                env_config={'initial_balance': 10000, 'commission': 0.001, 'max_position': 100}
            )
            rl_agent.train(data, verbose=0)
            rl_result = rl_agent.predict_action(data)
        
        with st.spinner('✨ Generating recommendation...'):
            decision_engine = DecisionEngine(config={
                'confidence_threshold': 0.6,
                'sentiment_weight': 0.3,
                'forecast_weight': 0.4,
                'rl_weight': 0.3
            })
            decision = decision_engine.make_decision(
                ticker=ticker,
                current_price=current_price,
                forecast_result=forecast_result,
                sentiment_result=sentiment_result,
                rl_result=rl_result
            )
        
        return {
            'data': data,
            'current_price': current_price,
            'forecast': forecast_result,
            'sentiment': sentiment_result,
            'rl_agent': rl_result,
            'decision': decision,
            'ticker': ticker
        }
        
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
        return None

# Main App
def main():
    st.markdown('<h1 class="main-header">🤖 AI Trading Agent Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stocks.png", width=100)
        st.title("📊 Settings")
        
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, TSLA, MSFT)")
        days = st.slider("Historical Data (days)", 30, 365, 60)
        
        # Info about model training
        st.info("⚡ Fast Mode: 25 epochs (~15-20 sec)")
        
        analyze_button = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 About")
        st.markdown("""
        This AI-powered trading agent uses:
        - **TimeGAN**: Generative forecasting
        - **FinBERT**: Sentiment analysis
        - **PPO**: Reinforcement learning
        - **Multi-model**: Decision fusion
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.caption("For educational purposes only. Not financial advice.")
    
    # Main content
    if analyze_button:
        ticker = ticker.upper()
        st.session_state.analysis_data = analyze_stock(ticker, days)
        st.session_state.analysis_complete = True
    
    if st.session_state.analysis_complete and st.session_state.analysis_data:
        analysis = st.session_state.analysis_data
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${analysis['current_price']:.2f}",
                f"{((analysis['current_price'] - analysis['data']['Close'].iloc[-2]) / analysis['data']['Close'].iloc[-2] * 100):.2f}%"
            )
        
        with col2:
            decision = analysis['decision']
            st.metric(
                "Recommendation",
                f"{decision['strength']} {decision['recommendation']}",
                f"{decision['confidence']*100:.1f}% confidence"
            )
        
        with col3:
            forecast = analysis['forecast']
            trend = forecast.get('trend', 'Sideways')
            trend_prob = forecast.get('trend_probability', 0.5) * 100
            st.metric(
                "AI Forecast",
                trend,
                f"{trend_prob:.0f}% probability"
            )
        
        with col4:
            risk = decision['risk_assessment']['level']
            risk_color = "🟢" if risk == "Low" else "🟡" if risk == "Moderate" else "🔴"
            st.metric(
                "Risk Level",
                f"{risk_color} {risk}",
                f"Score: {decision['risk_assessment']['score']:.2f}"
            )
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Price Analysis",
            "🤖 AI Forecast",
            "💭 Sentiment",
            "🎮 RL Agent",
            "📊 Summary",
            "💬 Ask Questions"
        ])
        
        with tab1:
            st.subheader("Price Chart & Technical Indicators")
            st.plotly_chart(create_price_chart(analysis['data'], analysis['ticker']), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_volume_chart(analysis['data']), use_container_width=True)
            with col2:
                st.plotly_chart(create_technical_indicators_chart(analysis['data']), use_container_width=True)
            
            # Technical indicators explanation
            with st.expander("📚 Understanding Technical Indicators"):
                st.markdown("""
                **Simple Moving Average (SMA)**
                - SMA 20: Short-term trend (20-day average)
                - SMA 50: Medium-term trend (50-day average)
                - When SMA 20 crosses above SMA 50 = Bullish signal
                
                **Relative Strength Index (RSI)**
                - Measures momentum (0-100 scale)
                - Above 70 = Overbought (potential sell)
                - Below 30 = Oversold (potential buy)
                
                **Volume**
                - High volume + price increase = Strong bullish signal
                - High volume + price decrease = Strong bearish signal
                """)
        
        with tab2:
            st.subheader("TimeGAN Forecast")
            st.plotly_chart(create_forecast_chart(
                analysis['forecast'],
                analysis['current_price'],
                analysis['ticker']
            ), use_container_width=True)
            
            st.plotly_chart(create_scenarios_chart(
                analysis['forecast'],
                analysis['ticker']
            ), use_container_width=True)
            
            # Forecast explanation
            with st.expander("🧠 How TimeGAN Works"):
                st.markdown("""
                **TimeGAN (Time-series Generative Adversarial Network)**
                
                1. **Generator**: Creates synthetic price scenarios
                2. **Discriminator**: Validates realistic patterns
                3. **Embedder**: Learns temporal features
                4. **Recovery**: Reconstructs price sequences
                
                **Process:**
                - Trains on historical price patterns
                - Generates 100 possible future scenarios
                - Calculates mean, upper, and lower bounds
                - Determines trend probability
                
                **Confidence Interval:**
                - Green line (95th percentile): Optimistic scenario
                - Yellow line (Mean): Most likely scenario
                - Red line (5th percentile): Pessimistic scenario
                """)
        
        with tab3:
            st.subheader("Market Sentiment Analysis")
            
            sentiment = analysis['sentiment']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Sentiment", sentiment.get('overall_label', 'Neutral'))
            with col2:
                st.metric("Sentiment Score", f"{sentiment.get('overall_score', 0):+.2f}")
            with col3:
                st.metric("Articles Analyzed", sentiment.get('article_count', 0))
            
            if 'article_sentiments' in sentiment and sentiment['article_sentiments']:
                st.plotly_chart(create_sentiment_chart(sentiment), use_container_width=True)
            
            # Sentiment explanation
            with st.expander("💭 Understanding Sentiment Analysis"):
                st.markdown("""
                **FinBERT Sentiment Model**
                
                - Analyzes financial news from Yahoo Finance & Finviz
                - Trained specifically on financial text
                - Classifies sentiment as Positive, Negative, or Neutral
                
                **Sentiment Score:**
                - +1.0 = Very Positive
                - 0.0 = Neutral
                - -1.0 = Very Negative
                
                **Impact on Trading:**
                - Positive sentiment → Potential price increase
                - Negative sentiment → Potential price decrease
                - Combined with other signals for final decision
                """)
        
        with tab4:
            st.subheader("Reinforcement Learning Agent")
            
            rl = analysis['rl_agent']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                action_color = "🟢" if rl.get('action') == 'BUY' else "🔴" if rl.get('action') == 'SELL' else "🟡"
                st.metric("RL Action", f"{action_color} {rl.get('action', 'HOLD')}")
            with col2:
                st.metric("Confidence", f"{rl.get('confidence', 0)*100:.1f}%")
            with col3:
                st.metric("Position Size", f"{rl.get('position_size', 0)} shares")
            
            st.plotly_chart(create_rl_training_chart(None), use_container_width=True)
            
            # RL explanation
            with st.expander("🎮 Understanding RL Agent"):
                st.markdown("""
                **PPO (Proximal Policy Optimization)**
                
                **How it works:**
                1. **Environment**: Simulates trading with historical data
                2. **State**: Current price, indicators, portfolio value
                3. **Actions**: BUY, SELL, or HOLD
                4. **Reward**: Profit/loss from trades
                5. **Learning**: Optimizes policy to maximize rewards
                
                **Training Process:**
                - 10,000 timesteps of simulated trading
                - Learns optimal entry/exit points
                - Balances risk vs reward
                - Adapts to market conditions
                
                **Output:**
                - Recommended action (BUY/SELL/HOLD)
                - Confidence level
                - Suggested position size
                """)
        
        with tab5:
            st.subheader("Final Recommendation Summary")
            
            decision = analysis['decision']
            
            # Recommendation card
            rec_color = "success" if "BUY" in decision['recommendation'] else "error" if "SELL" in decision['recommendation'] else "warning"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
                <h2 style="margin: 0;">🎯 {decision['strength']} {decision['recommendation']}</h2>
                <h3 style="margin: 0.5rem 0;">Confidence: {decision['confidence']*100:.1f}%</h3>
                <p style="margin: 0;">Combined Score: {decision['combined_score']:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model comparison
            st.plotly_chart(create_model_comparison_chart(
                analysis['forecast'],
                analysis['rl_agent']
            ), use_container_width=True)
            
            # Target range
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 Target Range")
                target = decision['target_range']
                st.metric("Target Low", f"${target['target_low']:.2f}")
                st.metric("Target High", f"${target['target_high']:.2f}")
            
            with col2:
                st.markdown("### 🛡️ Risk Management")
                st.metric("Stop Loss", f"${target['stop_loss']:.2f}")
                st.metric("Take Profit", f"${target['take_profit']:.2f}")
            
            # Decision explanation
            with st.expander("🧩 How the Final Decision is Made"):
                st.markdown("""
                **Multi-Model Decision Fusion**
                
                The final recommendation combines all AI models:
                
                1. **TimeGAN Forecast** (40% weight)
                   - Trend direction and probability
                   - Price target ranges
                
                2. **Sentiment Analysis** (30% weight)
                   - Market sentiment score
                   - News impact assessment
                
                3. **RL Agent** (30% weight)
                   - Optimal trading action
                   - Position sizing
                
                **Decision Process:**
                - Weighted average of all signals
                - Risk assessment based on volatility
                - Confidence calculation from agreement
                - Final recommendation with strength level
                
                **Strength Levels:**
                - **Strong**: High confidence (>80%), all models agree
                - **Moderate**: Medium confidence (60-80%), most models agree
                - **Weak**: Low confidence (<60%), mixed signals
                """)
            
            # Disclaimer
            st.warning("""
            ⚠️ **Important Disclaimer**
            
            This analysis is generated by AI for educational purposes only and does not constitute financial advice. 
            Always do your own research and consult with a qualified financial advisor before making investment decisions.
            Past performance does not guarantee future results.
            """)
        
        with tab6:
            st.subheader(f"💬 Chat with AI about {analysis['ticker']}")
            
            # Info box
            st.info("""
            🤖 **Natural Conversation Mode** - Powered by Groq (Llama 3.1 8B Instant)
            
            The AI has access to all analysis data and can answer questions like:
            - "What was Apple's highest price last week?"
            - "What's the volume traded yesterday?"
            - "Should I buy this stock?"
            - "Explain the forecast trend to me"
            - "What do you think will happen next?"
            
            ⚡ **Note:** Add your GROQ_API_KEY to .env file for chat functionality
            """)
            
            # Initialize chat agent if not already done
            if st.session_state.chat_agent is None:
                with st.spinner("Initializing AI assistant..."):
                    st.session_state.chat_agent = TradingChatAgent(use_timegan=True)
            
            # Show Groq connection status
            if st.session_state.chat_agent.llm:
                st.success("✅ Groq AI Connected - Natural conversation mode active")
            else:
                st.error("❌ Groq AI Not Connected - Add GROQ_API_KEY to .env file and restart")
            
            # Chat container
            chat_container = st.container()
            
            # Display chat history
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if message['role'] == 'user':
                        st.markdown(f"**👤 You:** {message['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {message['content']}")
                    st.markdown("---")
            
            # Chat input
            user_question = st.text_input(
                "Your question:",
                key="chat_input",
                placeholder=f"Ask about {analysis['ticker']}..."
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                send_button = st.button("Send 📤", use_container_width=True)
            with col2:
                clear_button = st.button("Clear Chat 🗑️", use_container_width=True)
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            if send_button and user_question:
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Get AI response using LLM with full context
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Always use LLM with the analysis context for natural conversation
                        response = st.session_state.chat_agent.chat_with_context(
                            user_message=user_question,
                            analysis=analysis,
                            data=analysis['data']
                        )
                        
                        # Add AI response
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        # Fallback response
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"I apologize, but I encountered an error: {str(e)}. Please make sure you have set up your GROQ_API_KEY in the .env file."
                        })
                
                st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to the AI Trading Agent!
        
        This intelligent system uses multiple AI models to analyze stocks and provide investment recommendations:
        
        ### 🤖 AI Models Used:
        
        1. **TimeGAN (Generative Forecasting)**
           - Generates 100 possible future price scenarios
           - Provides trend probability and confidence intervals
           - Learns complex temporal patterns from historical data
        
        2. **FinBERT (Sentiment Analysis)**
           - Analyzes financial news and social media
           - Trained specifically on financial text
           - Provides sentiment scores for market mood
        
        3. **PPO (Reinforcement Learning)**
           - Learns optimal trading strategies
           - Simulates thousands of trades
           - Recommends BUY/SELL/HOLD actions
        
        4. **Decision Fusion Engine**
           - Combines all model outputs
           - Weighted voting system
           - Risk-adjusted recommendations
        
        ### 🚀 Get Started:
        
        1. Enter a stock ticker in the sidebar (e.g., AAPL, TSLA, MSFT)
        2. Click "Analyze Stock"
        3. Explore interactive charts and explanations
        4. Review the final recommendation
        
        ---
        
        **Ready to analyze your first stock?** Enter a ticker symbol in the sidebar! 👈
        """)

if __name__ == "__main__":
    main()
