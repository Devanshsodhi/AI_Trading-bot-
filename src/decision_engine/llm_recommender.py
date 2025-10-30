"""
LLM-Based Trading Recommender
Generates trading advice using GPT/Claude or mock fallback
"""

import os, json, logging
from typing import Dict

log = logging.getLogger(__name__)

class LLMRecommender:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg.get('provider', 'groq')
        self.model = cfg.get('llm_model', 'llama-3.1-8b-instant')
        self.temp = cfg.get('temperature', 0.2)
        self.max_tokens = cfg.get('max_tokens', 2000)
        self.client = None
        self.use_mock = True

        try:
            if self.provider == 'groq' and os.getenv('GROQ_API_KEY'):
                from langchain_groq import ChatGroq
                self.client = ChatGroq(
                    model=self.model,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    groq_api_key=os.getenv('GROQ_API_KEY')
                )
                self.client_type = 'groq'
                self.use_mock = False
                log.info(f"Using Groq model {self.model}")
            elif os.getenv('OPENAI_API_KEY'):
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.client_type = 'openai'
                self.model = cfg.get('openai_model', 'gpt-3.5-turbo')
                self.use_mock = False
                log.info(f"Using OpenAI model {self.model}")
            else:
                log.warning("No LLM API key found, fallback to mock.")
        except Exception as e:
            log.warning(f"LLM init failed: {e}, using mock.")

    def generate(self, ticker: str, forecast: Dict, sentiment: Dict, rl: Dict, price: float) -> Dict:
        ctx = self._context(ticker, forecast, sentiment, rl, price)
        if self.use_mock: return self._mock(ctx)
        try:
            if self.client_type == 'groq':
                from langchain.schema import HumanMessage, SystemMessage
                msgs = [SystemMessage(content=self._prompt()), HumanMessage(content=ctx)]
                resp = self.client.invoke(msgs)
                txt = resp.content
            else:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self._prompt()},
                              {"role": "user", "content": ctx}],
                    temperature=self.temp, max_tokens=self.max_tokens
                )
                txt = r.choices[0].message.content
            return self._parse(txt)
        except Exception as e:
            log.error(f"LLM error: {e}")
            return self._mock(ctx)

    def _prompt(self):
        return """You are a trading expert. Analyze data and return JSON:
{
 "action": "BUY/SELL/HOLD",
 "confidence": 0.0-1.0,
 "reasoning": "why",
 "key_factors": [],
 "risks": [],
 "target_price": 0,
 "stop_loss": 0,
 "time_horizon": "short/medium/long",
 "position_size": "small/medium/large"
}"""

    def _context(self, t, f, s, r, p):
        c = f"""Analyze {t}:
Current: ${p:.2f}
Forecast: {f.get('trend','?')} ({f.get('trend_probability',0):.1%})
Sentiment: {s.get('overall_label','Neutral')} ({s.get('overall_score',0):+.2f})
RL Agent: {r.get('action','HOLD')} ({r.get('confidence',0):.1%})
"""
        if f.get('mean_forecast'):
            exp = f['mean_forecast'][-1]
            chg = ((exp - p) / p) * 100
            c += f"Expected {f.get('forecast_days',5)}d price: ${exp:.2f} ({chg:+.1f}%)\n"
        return c

    def _parse(self, txt: str) -> Dict:
        try:
            j = txt[txt.find('{'):txt.rfind('}')+1]
            d = json.loads(j)
            return {
                'action': d.get('action','HOLD').upper(),
                'confidence': float(d.get('confidence',0.5)),
                'reasoning': d.get('reasoning', txt),
                'key_factors': d.get('key_factors', []),
                'risks': d.get('risks', []),
                'target_price': float(d.get('target_price', 0)),
                'stop_loss': float(d.get('stop_loss', 0)),
                'time_horizon': d.get('time_horizon', 'medium-term'),
                'position_size': d.get('position_size', 'medium'),
                'full_text': txt,
                'generation_method': 'LLM'
            }
        except:
            txtU = txt.upper()
            if 'BUY' in txtU: act, conf = 'BUY', 0.7
            elif 'SELL' in txtU: act, conf = 'SELL', 0.7
            else: act, conf = 'HOLD', 0.5
            return {'action': act, 'confidence': conf, 'reasoning': txt,
                    'key_factors': [], 'risks': [], 'target_price': 0,
                    'stop_loss': 0, 'time_horizon': 'medium-term',
                    'position_size': 'medium', 'full_text': txt, 'generation_method': 'LLM'}

    def _mock(self, ctx: str) -> Dict:
        if 'Upward' in ctx and 'Bullish' in ctx:
            a, conf = 'BUY', 0.75
            reason = "Upward forecast + bullish sentiment → buy."
            factors = ["Positive trend", "Bullish mood"]
            risks = ["Volatility", "News shocks"]
        elif 'Downward' in ctx and 'Bearish' in ctx:
            a, conf = 'SELL', 0.75
            reason = "Downward trend + bearish mood → sell."
            factors = ["Negative trend", "Bearish sentiment"]
            risks = ["Reversal", "Missed rebound"]
        else:
            a, conf = 'HOLD', 0.6
            reason = "Mixed signals → hold."
            factors, risks = ["Neutral outlook"], ["Unclear direction"]
        return {'action': a, 'confidence': conf, 'reasoning': reason,
                'key_factors': factors, 'risks': risks, 'target_price': 0,
                'stop_loss': 0, 'time_horizon': 'medium-term',
                'position_size': 'medium', 'full_text': reason, 'generation_method': 'Mock'}

    def report(self, rec: Dict, ticker: str) -> str:
        out = f"# {ticker} Recommendation\n\n**Action:** {rec['action']} ({rec['confidence']:.0%})\n\n{rec['reasoning']}\n\n"
        if rec.get('key_factors'): out += "## Key Factors\n" + "\n".join(f"- {x}" for x in rec['key_factors']) + "\n"
        if rec.get('risks'): out += "\n## Risks\n" + "\n".join(f"- {x}" for x in rec['risks']) + "\n"
        if rec.get('target_price', 0) > 0:
            out += f"\n## Price Targets\nTarget: ${rec['target_price']:.2f}\nStop Loss: ${rec['stop_loss']:.2f}\n"
        return out + "\n---\n*Generated by LLM System*\n"
