# 🤖 Generative AI Trading System

>A sophisticated multi-agent trading system powered by Generative AI, combining TimeGAN, Large Language Models, Reinforcement Learning, and Deep Learning for intelligent stock market analysis.

## 📋 Table of Contents

- [Why This Is a Generative AI Project](#-why-this-is-a-generative-ai-project)
- [System Architecture](#-system-architecture)
- [How Forecasting Works](#-how-forecasting-works)
- [How AI Agents Work](#-how-ai-agents-work)
- [Installation & Usage](#-installation--usage)
- [Technical Stack](#-technical-stack)

---

## 🌟 Why This Is a Generative AI Project

This project is fundamentally a **Generative AI system** - it doesn't just predict or classify, it **generates** new content. Here's why:

### 1. **TimeGAN: Generative Adversarial Network for Time Series**

**Primary Generative Component:** At the core is **TimeGAN**, a GAN that generates synthetic future price scenarios.

**How It's Generative:**
- Takes **random noise** as input (not historical data)
- **Generates 200 completely new** price sequences that have never existed
- Uses adversarial training: Generator creates fake data to fool Discriminator
- Outputs **probabilistic futures**, not deterministic predictions

```python
# Traditional: Predicts ONE future
prediction = model.predict(historical_data)

# TimeGAN: GENERATES 200 possible futures from noise
noise = torch.randn(200, 64)  # Random vectors
scenarios = generator(noise)   # Creates NEW sequences
```

**Why This Matters:** Traditional forecasting extrapolates patterns. TimeGAN **generates** entirely new, realistic market scenarios by learning the underlying data distribution - the hallmark of generative AI.

### 2. **Large Language Model: Natural Language Generation**

Uses **Llama 3.1 8B** (via Groq) to **generate** human-readable trading insights:
- Synthesizes recommendations from structured data into natural language
- Creates contextual explanations that don't exist in a database
- Generates personalized responses to user questions

```python
# Generates NEW text, not retrieval
llm_response = llm.generate(f"Explain {ticker} forecast: {data}")
# Creates unique, contextual natural language
```

### 3. **Reinforcement Learning: Policy Generation**

PPO agent **generates** trading policies through trial-and-error:
- Creates action sequences never seen during training
- Generates adaptive strategies for unseen market conditions
- Synthesizes optimal behavior from reward signals

### 4. **Strategy Evolution: Genetic Algorithm Generation**

Uses evolutionary algorithms to **generate** novel trading strategies:
- Creates population of random strategies
- Mutates and recombines successful approaches
- Generates entirely new strategy combinations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   INPUT: Stock Ticker (e.g., AAPL)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┬────────────────────┐
         ▼                                ▼                    ▼
┌─────────────────┐           ┌──────────────────┐   ┌─────────────────┐
│ Market Data     │           │ News Scraping    │   │ Technical       │
│ (yfinance)      │           │ (BeautifulSoup)  │   │ Indicators (TA) │
│ 2 years OHLCV   │           │ Financial news   │   │ RSI,MACD,BB,SMA │
└────────┬────────┘           └────────┬─────────┘   └────────┬────────┘
         │                             │                      │
         └──────────────┬──────────────┴──────────────┬───────┘
                        ▼                             ▼
         ┌──────────────────────────────────────────────────────────┐
         │           GENERATIVE AI AGENTS LAYER                     │
         │                                                            │
         │  ┌────────────────────────────────────────────────────┐  │
         │  │ 🔮 TIMEGAN FORECASTER (Primary Generative AI)      │  │
         │  │                                                     │  │
         │  │  Generator (LSTM) ◄─GAN Training─► Discriminator   │  │
         │  │  • 128 hidden dim, 64 noise dim                    │  │
         │  │  • Trained on 60-day OHLCV sequences               │  │
         │  │  • Generates 200 unique 7-day price scenarios      │  │
         │  │  • Output: Mean, confidence bands, trend prob      │  │
         │  └────────────────────────────────────────────────────┘  │
         │                                                            │
         │  ┌────────────────────────────────────────────────────┐  │
         │  │ 💭 FINBERT SENTIMENT ANALYZER                      │  │
         │  │  • BERT transformer fine-tuned on financial text   │  │
         │  │  • Analyzes up to 20 news articles                 │  │
         │  │  • Outputs: positive/negative/neutral scores       │  │
         │  └────────────────────────────────────────────────────┘  │
         │                                                            │
         │  ┌────────────────────────────────────────────────────┐  │
         │  │ 🎮 PPO REINFORCEMENT LEARNING AGENT                │  │
         │  │  • Proximal Policy Optimization                    │  │
         │  │  • 200,000 timesteps training                      │  │
         │  │  • Learns BUY/SELL/HOLD policy                     │  │
         │  └────────────────────────────────────────────────────┘  │
         │                                                            │
         │  ┌────────────────────────────────────────────────────┐  │
         │  │ 🧠 LLM REASONER (Secondary Generative AI)          │  │
         │  │  • Llama 3.1 8B via Groq                           │  │
         │  │  • Generates natural language recommendations      │  │
         │  │  • Conversational AI chat interface                │  │
         │  └────────────────────────────────────────────────────┘  │
         └────────────────────────┬───────────────────────────────┘
                                  ▼
         ┌──────────────────────────────────────────────────────────┐
         │              DECISION FUSION ENGINE                      │
         │  Combined Score = 40% TimeGAN + 30% Sentiment + 30% RL  │
         │                                                           │
         │  Thresholds:                                             │
         │  • > 0.6  → Strong BUY      • > -0.3 → HOLD             │
         │  • > 0.3  → Moderate BUY    • > -0.6 → Moderate SELL    │
         │  • < -0.6 → Strong SELL                                  │
         └────────────────────────┬─────────────────────────────────┘
                                  ▼
         ┌──────────────────────────────────────────────────────────┐
         │         STREAMLIT WEB UI + AI CHAT INTERFACE            │
         │  • Interactive price charts  • TimeGAN scenario plots   │
         │  • Technical indicators      • LLM-powered chatbot      │
         │  • Sentiment visualization   • Real-time recommendations│
         └──────────────────────────────────────────────────────────┘
```

---

## 🔮 How Forecasting Works

### TimeGAN Forecasting Pipeline

#### 1. **Data Preparation**
```python
# Fetch 2 years of historical data
data = yfinance.download('AAPL', period='2y')
# Add technical indicators
data = add_indicators(data)  # RSI, MACD, Bollinger Bands, etc.
# Normalize to [0,1]
normalized = MinMaxScaler().fit_transform(data)
# Create 60-day sequences
sequences = create_sequences(normalized, seq_len=60)
```

#### 2. **Adversarial Training**
```python
for epoch in range(100):
    # Train Discriminator to distinguish real vs fake
    real_data = historical_sequences
    fake_data = generator(random_noise)
    d_loss = BCE(discriminator(real_data), 1) + BCE(discriminator(fake_data), 0)
    
    # Train Generator to fool Discriminator
    fake_data = generator(random_noise)
    g_loss = BCE(discriminator(fake_data), 1)  # Want D to output 1
```

**Key Innovation:** Unlike standard prediction models, the generator never sees historical data directly during generation - it learns to create realistic patterns from pure noise.

#### 3. **Scenario Generation**
```python
scenarios = []
for i in range(200):
    # Generate from random noise
    noise = torch.randn(1, 64)
    raw_forecast = generator(noise, forecast_days=7)
    
    # Apply realistic constraints
    # - Max 5% daily change
    # - Consider historical volatility
    # - Maintain trend consistency
    constrained_forecast = apply_constraints(raw_forecast)
    scenarios.append(constrained_forecast)
```

#### 4. **Statistical Aggregation**
```python
mean_forecast = np.mean(scenarios, axis=0)          # Expected path
std_forecast = np.std(scenarios, axis=0)            # Uncertainty
lower_bound = np.percentile(scenarios, 5, axis=0)   # 5th percentile
upper_bound = np.percentile(scenarios, 95, axis=0)  # 95th percentile

# Calculate trend probability
upward = sum(s[-1] > current_price for s in scenarios)
trend_probability = upward / 200  # % scenarios going up

# Confidence (inverse of relative volatility)
confidence = 1 - (std_forecast.mean() / mean_forecast.mean())
```
## 🤖 How AI Agents Work

### Multi-Agent Ensemble System

Each agent specializes in one aspect of market analysis. They work **independently** and their outputs are **fused** for the final decision.

#### Agent 1: TimeGAN Forecaster
- **Input:** 60-day OHLCV sequences
- **Process:** Generates 200 scenarios from random noise
- **Output:** 
  - Mean forecast (expected future prices)
  - Confidence interval (90% band)
  - Trend probability (% scenarios going up)
  - Confidence score (0-1)

#### Agent 2: FinBERT Sentiment Analyzer
- **Input:** Recent financial news articles
- **Process:** 
  1. Web scraping for latest news
  2. BERT tokenization
  3. Sentiment classification (pos/neg/neutral)
  4. Aggregation across articles
- **Output:**
  - Overall sentiment score (-1 to +1)
  - Individual article sentiments
  - News count and confidence

#### Agent 3: PPO Trading Agent
- **Input:** Market state (prices, indicators, forecast, sentiment)
- **Process:**
  1. Observes current state
  2. Evaluates action values (BUY/SELL/HOLD)
  3. Selects action based on learned policy
- **Output:**
  - Recommended action
  - Action confidence
  - Expected value estimate

#### Agent 4: LLM Reasoning Agent
- **Input:** All agent outputs + market context
- **Process:**
  1. Constructs detailed prompt with all data
  2. Llama 3.1 generates natural language
  3. Parses and structures response
- **Output:**
  - Natural language recommendation
  - Reasoning explanation
  - Risk assessment

### Decision Fusion Algorithm

```python
def fuse_decisions(timegan, sentiment, rl_agent):
    # Normalize all scores to [-1, +1]
    
    # TimeGAN contribution
    timegan_score = (2 * timegan['trend_prob'] - 1) * timegan['confidence']
    
    # Sentiment contribution
    sentiment_score = sentiment['overall_score']
    
    # RL contribution
    if rl_agent['action'] == 'BUY':
        rl_score = rl_agent['confidence']
    elif rl_agent['action'] == 'SELL':
        rl_score = -rl_agent['confidence']
    else:
        rl_score = 0
    
    # Weighted ensemble (40-30-30)
    combined = 0.4*timegan_score + 0.3*sentiment_score + 0.3*rl_score
    
    # Map to recommendation
    if combined > 0.6:    return "Strong BUY", confidence
    elif combined > 0.3:  return "Moderate BUY", confidence
    elif combined > -0.3: return "HOLD", confidence
    elif combined > -0.6: return "Moderate SELL", confidence
    else:                 return "Strong SELL", confidence
```

**Why Ensemble?** Each agent has strengths and weaknesses:
- TimeGAN: Great at patterns, but no news awareness
- FinBERT: Captures sentiment, but no price context
- RL Agent: Learns strategies, but can overfit

Combining them creates a **robust, well-rounded recommendation**.

**Features:**
- Enter stock ticker (e.g., AAPL, TSLA, MSFT)
- View AI-generated forecasts with confidence bands
- See sentiment analysis from news
- Get BUY/SELL/HOLD recommendations
- Chat with AI about the analysis

## 🔧 Tech Stack

### Core AI/ML
- **PyTorch 2.0+**: Deep learning framework for TimeGAN
- **Transformers (Hugging Face)**: FinBERT sentiment analysis
- **Stable-Baselines3**: PPO reinforcement learning
- **LangChain**: LLM orchestration
- **Groq**: Ultra-fast LLM inference

### Data & Analysis
- **yfinance**: Real-time market data
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Preprocessing and scaling
- **TA-Lib**: Technical indicators

### Visualization & UI
- **Streamlit**: Interactive web application
- **Plotly**: Interactive charts
- **Matplotlib**: Static visualizations

### Other
- **gymnasium**: RL environment interface
- **DEAP**: Genetic algorithms for strategy generation
- **BeautifulSoup**: Web scraping for news

---

## 📊 Project Structure

```
GENAI/
├── src/
│   ├── forecasting_agent/
│   │   ├── timegan_forecaster_persistent.py  # TimeGAN implementation
│   │   └── realistic_constraints.py          # Forecast constraints
│   ├── sentiment_agent/
│   │   └── sentiment_analyzer.py             # FinBERT sentiment
│   ├── trading_agent/
│   │   ├── rl_trader.py                      # PPO agent
│   │   └── trading_env.py                    # RL environment
│   ├── decision_engine/
│   │   └── ensemble_decision.py              # Fusion logic
│   └── data/
│       ├── data_loader.py                    # Market data fetching
│       └── technical_indicators.py           # Indicator calculations
├── models/                                   # Saved model checkpoints
├── streamlit_app.py                          # Web UI
├── main_generative.py                        # CLI interface
├── chat_agent.py                             # Conversational AI
├── config.py                                 # Configuration parameters
├── requirements.txt                          # Dependencies
└── README.md                                 # This file
```

---

## 🎯 Key Configuration Parameters

Located in `config.py`:

```python
# TimeGAN - Critical for accuracy
TIMEGAN_CONFIG = {
    'hidden_dim': 128,      # Network capacity
    'noise_dim': 64,        # Latent space size
    'epochs': 100,          # Training duration
    'seq_len': 60,          # Input context (60 days)
    'num_scenarios': 200,   # Generated futures
    'learning_rate': 0.0005,
}

# RL Agent - For optimal trading
RL_CONFIG = {
    'total_timesteps': 200000,  # Training steps
    'learning_rate': 0.0001,    # Policy learning rate
}

# Decision Weights - Customize importance
DECISION_CONFIG = {
    'forecast_weight': 0.4,   # 40% TimeGAN
    'sentiment_weight': 0.3,  # 30% FinBERT
    'rl_weight': 0.3,         # 30% PPO
}
```

---

## 📈 Performance Notes

### First Run
- **TimeGAN training**: 2-3 minutes (one-time per stock)
- **RL training**: 30-60 seconds
- **Total analysis**: ~3-5 minutes

### Subsequent Runs
- Models are cached automatically
- Analysis completes in **10-20 seconds**
- Cache cleared if stock data changes significantly

### Accuracy Tips
1. Use at least 2 years of historical data
2. TimeGAN performs best with 100+ epochs
3. RL agent needs 50k+ timesteps for stable policies
4. Ensemble decision is more reliable than individual agents

---

## 🎓 Why This Architecture?

### Generative AI Benefits
1. **Uncertainty Quantification**: 200 scenarios = probability distribution
2. **Black Swan Events**: Generator can create unlikely but possible scenarios
3. **Market Dynamics**: Learns underlying distributions, not just patterns
4. **Explainability**: LLM translates technical analysis to natural language

### Multi-Agent Benefits
1. **Robustness**: No single point of failure
2. **Complementary**: Each agent captures different market aspects
3. **Adaptability**: Can add/remove agents without redesign
4. **Transparency**: Individual agent outputs are visible

### Production-Ready Features
- Model persistence (training cached)
- Error handling and fallbacks
- Configurable parameters
- Logging and monitoring
- Real-time data integration

---

## 📚 Learn More

### Key Papers
- **TimeGAN**: "Time-series Generative Adversarial Networks" (NeurIPS 2019)
- **FinBERT**: "FinBERT: Financial Sentiment Analysis with BERT" (2020)
- **PPO**: "Proximal Policy Optimization Algorithms" (2017)

### Related Concepts
- Generative Adversarial Networks (GANs)
- Transformer architectures (BERT)
- Reinforcement Learning (RL)
- Ensemble Learning
- Time Series Analysis

---

## 🤝 Contributing

This project demonstrates advanced GenAI concepts for trading. Potential improvements:
- Add more sophisticated ensemble methods
- Integrate additional data sources (options, volume patterns)
- Implement risk management module
- Add backtesting framework
- Support for cryptocurrency analysis

---

## ⚠️ Disclaimer

**This system is for educational and research purposes only.** It demonstrates generative AI techniques applied to financial markets. **NOT financial advice.** Always consult licensed financial advisors before making investment decisions.

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using Generative AI**
