"""
Decision Engine - combines all agent outputs to form final recommendation
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DecisionEngine:
    def __init__(self, config: dict):
        self.config = config
        self.conf_thresh = config.get('confidence_threshold', 0.6)
        self.w_sent = config.get('sentiment_weight', 0.3)
        self.w_forecast = config.get('forecast_weight', 0.4)
        self.w_rl = config.get('rl_weight', 0.3)
        logger.info("Decision Engine ready")

    def make_decision(self, forecast, sentiment, rl, ticker, price):
        """Combine all agent outputs"""
        logger.info("Making final decision...")
        f_score = self._score_forecast(forecast)
        s_score = sentiment.get('overall_score', 0)
        rl_score = self._action_to_score(rl.get('action', 'HOLD'))

        score = (self.w_forecast * f_score +
                 self.w_sent * s_score +
                 self.w_rl * rl_score)

        if score > 0.3:
            action = 'BUY'; strength = 'Strong' if score > 0.6 else 'Moderate'
        elif score < -0.3:
            action = 'SELL'; strength = 'Strong' if score < -0.6 else 'Moderate'
        else:
            action = 'HOLD'; strength = 'Neutral'

        conf = self._calc_conf(forecast, sentiment, rl, score)
        risk = self._risk(forecast, sentiment, score)
        target = self._targets(price, forecast, score)
        explain = self._explain(ticker, price, action, strength, forecast, sentiment, rl, score, conf)

        return {
            'recommendation': action,
            'strength': strength,
            'confidence': float(conf),
            'combined_score': float(score),
            'explanation': explain,
            'risk_assessment': risk,
            'target_range': target,
            'position_size': rl.get('position_size', 0),
            'timestamp': datetime.now().isoformat(),
            'components': {
                'forecast_score': float(f_score),
                'sentiment_score': float(s_score),
                'rl_score': float(rl_score),
            }
        }

    def _score_forecast(self, f):
        t, p = f.get('trend', 'Sideways'), f.get('trend_probability', 0.5)
        if t == 'Upward': return (p - 0.5) * 2
        if t == 'Downward': return -(p - 0.5) * 2
        return 0

    def _action_to_score(self, a):
        return {'BUY': 1.0, 'HOLD': 0.0, 'SELL': -1.0}.get(a, 0.0)

    def _calc_conf(self, f, s, rl, score):
        confs = [f.get('confidence', .5), s.get('confidence', .5), rl.get('confidence', .5)]
        scores = [self._score_forecast(f), s.get('overall_score', 0), self._action_to_score(rl.get('action', 'HOLD'))]
        agree = 1 - np.std(scores) / 2
        return min((np.mean(confs) + agree) / 2, .99)

    def _explain(self, ticker, price, act, strength, f, s, rl, score, conf):
        parts = [f"## {strength} {act} ({conf:.0%}) | Score: {score:+.2f}",
                 f"\n{ticker} @ ${price:.2f}"]

        trend = f.get('trend', 'Unknown')
        prob = f.get('trend_probability', 0)
        days = f.get('forecast_days', 5)
        mean_f = f.get('mean_forecast', [])
        if mean_f:
            exp_p = mean_f[-1]
            change = ((exp_p - price) / price) * 100
            parts.append(f"\nForecast: {trend} ({prob:.0%}), {days}d -> ${exp_p:.2f} ({change:+.1f}%)")

        parts.append(f"\nSentiment: {s.get('overall_label','Neutral')} ({s.get('overall_score',0):+.2f}) | {s.get('article_count',0)} sources")
        parts.append(f"RL Agent: {rl.get('action','HOLD')} ({rl.get('confidence',0):.0%}) | {rl.get('rationale','-')}")

        if act == 'BUY':
            parts.append("Advice: Likely buying opportunity, consider entry with stop-loss protection.")
        elif act == 'SELL':
            parts.append("Advice: Possible overvaluation, consider trimming position.")
        else:
            parts.append("Advice: Hold position, monitor for clearer signals.")
        return "\n".join(parts)

    def _risk(self, f, s, score):
        std_f = f.get('std_forecast', [0])
        mean_f = f.get('mean_forecast', [100])
        vol = np.mean(np.array(std_f)/np.array(mean_f)) if mean_f else .1
        dist = s.get('sentiment_distribution', {})
        cons = max(dist.values()) if dist else .5
        risk_score = .5*vol + .3*(1-cons) + .2*abs(score)

        if risk_score < .3: lvl, desc = 'Low', 'Stable conditions.'
        elif risk_score < .6: lvl, desc = 'Moderate', 'Some uncertainty.'
        else: lvl, desc = 'High', 'Volatile or unclear conditions.'

        return {'level': lvl, 'score': float(risk_score), 'description': desc}

    def _targets(self, price, f, score):
        mean_f = f.get('mean_forecast', [price])
        std_f = f.get('std_forecast', [price*0.02])
        exp_p = mean_f[-1] if mean_f else price
        vol = np.mean(std_f) if std_f else price*0.03
        t_low, t_high = exp_p - vol, exp_p + vol
        min_mv = price*0.02
        if abs(exp_p - price) < min_mv:
            if score > 0: t_low, t_high = price - min_mv, price + 2*min_mv
            elif score < 0: t_low, t_high = price - 2*min_mv, price + min_mv
            else: t_low, t_high = price - min_mv, price + min_mv

        if score > 0:
            sl, tp = price*0.95, max(t_high, price*1.05)
        elif score < 0:
            sl, tp = price*1.05, min(t_low, price*0.95)
        else:
            sl, tp = price*0.95, price*1.05

        return {'current': float(price), 'target_low': float(t_low), 'target_high': float(t_high),
                'stop_loss': float(sl), 'take_profit': float(tp)}

    def summary(self, d):
        s = f"{d['strength']} {d['recommendation']} ({d['confidence']:.0%}) | Risk: {d['risk_assessment']['level']}"
        if d['recommendation'] != 'HOLD':
            t = d['target_range']
            s += f" | Target: ${t['target_low']:.2f}-{t['target_high']:.2f}, SL: ${t['stop_loss']:.2f}"
        return s
