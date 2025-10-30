"""
Sentiment Agent
FinBERT-based sentiment analysis from news, social media, Yahoo Finance, and Finviz
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SentimentAgent:

    def __init__(self, config: dict):
        """
        Initialize Sentiment Agent

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model_name', 'ProsusAI/finbert')
        self.max_articles = config.get('max_articles', 20)
        self.sentiment_threshold = config.get('sentiment_threshold', 0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load FinBERT model
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Will use mock sentiment analysis")
            self.model = None
            self.tokenizer = None

    def scrape_news(self, ticker: str) -> List[Dict]:
        """
        Scrape financial news for a given ticker from Yahoo Finance and Finviz
        """
        articles = []

        # --- Yahoo Finance ---
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                news_items = soup.find_all('h3', limit=self.max_articles)
                for item in news_items:
                    title = item.get_text().strip()
                    if title:
                        articles.append({
                            'title': title,
                            'text': title,
                            'source': 'Yahoo Finance',
                            'date': datetime.now().strftime('%Y-%m-%d')
                        })
        except Exception as e:
            logger.warning(f"Yahoo Finance scraping failed: {e}")

        # --- Finviz ---
        try:
            finviz_url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(finviz_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                news_table = soup.find('table', class_='fullview-news-outer')
                if news_table:
                    rows = news_table.find_all('tr')
                    for row in rows:
                        link = row.find('a')
                        date_text = row.find('td').text.strip() if row.find('td') else ''
                        if link and link.text:
                            title = link.text.strip()
                            # Extract date
                            try:
                                date_match = re.search(r'(\w{3}-\d{1,2}-\d{2})', date_text)
                                article_date = datetime.strptime(date_match.group(1), '%b-%d-%y').strftime('%Y-%m-%d') if date_match else datetime.now().strftime('%Y-%m-%d')
                            except:
                                article_date = datetime.now().strftime('%Y-%m-%d')

                            articles.append({
                                'title': title,
                                'text': title,
                                'source': 'Finviz',
                                'date': article_date
                            })
        except Exception as e:
            logger.warning(f"Finviz scraping failed: {e}")

        # Deduplicate titles
        seen_titles = set()
        unique_articles = []
        for a in articles:
            if a['title'] not in seen_titles:
                unique_articles.append(a)
                seen_titles.add(a['title'])

        # Use mock if too few articles
        if len(unique_articles) < 5:
            logger.info("Using mock news data")
            unique_articles = self._generate_mock_news(ticker)

        return unique_articles[:self.max_articles]

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if self.model is None or self.tokenizer is None:
            return self._mock_sentiment_analysis(text)

        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores = predictions[0].cpu().numpy()
            sentiment_dict = {'positive': float(scores[0]), 'negative': float(scores[1]), 'neutral': float(scores[2])}
            sentiment_score = sentiment_dict['positive'] - sentiment_dict['negative']
            sentiment_label = 'Bullish' if sentiment_score > self.sentiment_threshold else 'Bearish' if sentiment_score < -self.sentiment_threshold else 'Neutral'
            return {'score': sentiment_score, 'label': sentiment_label, 'scores': sentiment_dict}
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._mock_sentiment_analysis(text)


    def analyze_ticker(self, ticker: str) -> Dict:
        """Analyze sentiment for a ticker"""
        logger.info(f"Analyzing sentiment for {ticker}...")
        articles = self.scrape_news(ticker)
        if not articles:
            return {'overall_score': 0.0, 'overall_label': 'Neutral', 'confidence': 0.0, 'article_count': 0, 'summary': 'No news data available.', 'top_articles': []}

        sentiments = []
        analyzed_articles = []
        for article in articles:
            text = article['title'] + ' ' + article.get('text', '')
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment['score'])
            analyzed_articles.append({'title': article['title'], 'sentiment': sentiment['label'], 'score': sentiment['score'], 'date': article['date'], 'source': article['source']})

        overall_score = float(np.mean(sentiments))
        sentiment_std = np.std(sentiments)
        confidence = float(1 - min(sentiment_std, 1.0))
        overall_label = 'Bullish' if overall_score > 0.2 else 'Bearish' if overall_score < -0.2 else 'Neutral'
        summary = self._generate_summary(ticker, analyzed_articles, overall_label, overall_score)

        return {
            'overall_score': overall_score,
            'overall_label': overall_label,
            'confidence': confidence,
            'article_count': len(articles),
            'summary': summary,
            'top_articles': analyzed_articles[:5],
            'sentiment_distribution': {
                'bullish': sum(1 for s in sentiments if s > 0.2) / len(sentiments),
                'bearish': sum(1 for s in sentiments if s < -0.2) / len(sentiments),
                'neutral': sum(1 for s in sentiments if -0.2 <= s <= 0.2) / len(sentiments),
            }
        }

    def _generate_summary(self, ticker: str, articles: List[Dict], label: str, score: float) -> str:
        """Generate natural language summary of sentiment"""
        bullish = sum(1 for a in articles if a['sentiment'] == 'Bullish')
        bearish = sum(1 for a in articles if a['sentiment'] == 'Bearish')
        neutral = sum(1 for a in articles if a['sentiment'] == 'Neutral')
        summary_parts = []
        if label == 'Bullish':
            summary_parts.append(f"Market sentiment for {ticker} is predominantly positive.")
        elif label == 'Bearish':
            summary_parts.append(f"Market sentiment for {ticker} is predominantly negative.")
        else:
            summary_parts.append(f"Market sentiment for {ticker} is mixed.")
        summary_parts.append(f"Analysis of {len(articles)} articles shows {bullish} bullish, {bearish} bearish, and {neutral} neutral signals.")
        top_articles = articles[:3]
        if top_articles:
            themes = [a['title'][:50] + '...' if len(a['title']) > 50 else a['title'] for a in top_articles]
            summary_parts.append(f"Key themes include: {'; '.join(themes[:2])}.")
        return ' '.join(summary_parts)
