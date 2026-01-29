#!/usr/bin/env python3
"""
BACKTEST ENGINE v14.0 - FULL MODE OPTIMIZED
Optimisé pour l'exécution complète avec performance améliorée
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Manager, cpu_count
import json
import time
import warnings
import pickle
import hashlib
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

class Config:
    # Actifs (réduits pour le FULL mode optimisé)
    CRYPTO_TICKERS = {'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD'}  # 2 cryptos au lieu de 3
    STOCK_TICKERS = ['AAPL', 'TSLA', 'NVDA']  # 3 actions au lieu de 6
    
    # Timeframes (réduits pour optimisation)
    TIMEFRAMES = {'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1wk'}  # 4 au lieu de 6
    
    LOOKBACK_PERIODS = {'H1': '729d', 'H4': '729d', 'D1': '1825d', 'W1': '3650d'}
    MIN_BARS = {'H1': 1000, 'H4': 800, 'D1': 500, 'W1': 200}
    WARMUP_BARS = 200
    
    # Paramètres optimisés pour FULL mode
    RSI_BUY_RANGE = list(range(20, 41, 3))  # 7 valeurs: 20, 23, 26, 29, 32, 35, 38
    EMA_PERIODS = [10, 20, 30, 50, 100, 150, 200]  # 7 périodes au lieu de 11
    BB_PERIODS = [15, 20, 30, 50]  # 4 périodes au lieu de 7
    BB_MULTIPLIERS = [1.5, 2.0, 2.5, 3.0]  # 4 multiplicateurs
    
    # Paramètres de risque
    INITIAL_CAPITAL = 10000.0
    RISK_PER_TRADE = 0.02
    MAX_POSITION_PCT = 0.15
    ATR_SL_MULTIPLIERS = [2.0, 2.5, 3.0]
    ATR_TP_MULTIPLIERS = [3.0, 3.5, 4.0]
    MAX_BARS_HOLD = [15, 30, 50]
    
    # Coûts
    COMMISSIONS = {'H1': 0.0003, 'H4': 0.0002, 'D1': 0.0001, 'W1': 0.0001}
    SLIPPAGE = {'H1': 0.0001, 'H4': 0.00005, 'D1': 0.00002, 'W1': 0.00001}
    
    # Filtres
    MIN_TRADES = 5
    MIN_VOLUME_RATIO = 0.1
    
    # Performance
    CPU_CORES = cpu_count()
    WORKERS = max(1, min(CPU_CORES - 1, 8))  # Maximum 8 workers
    MODE = 'FULL'  # Mode FULL activé
    
    # Cache et optimisation
    RATE_LIMIT_DELAY = 0.5  # Réduit pour accélérer
    MAX_TRADES_MEMORY = 100000
    USE_DISK_CACHE = True  # Cache disque activé
    CACHE_DIR = Path.cwd() / 'data_cache'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path.cwd() / f'backtest_results_{timestamp}'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def total_combinations(cls):
        """Calcule le nombre total de combinaisons à tester"""
        if cls.MODE == 'TEST':
            return 39
        else:
            strategies = 0
            # RSI combos
            strategies += len(cls.RSI_BUY_RANGE)
            # EMA combos (3 types)
            strategies += len(cls.EMA_PERIODS) * 3
            # BB combos (3 types)
            strategies += len(cls.BB_PERIODS) * len(cls.BB_MULTIPLIERS) * 3
            
            risk_configs = 0
            risk_configs += 1  # opposite_signal
            risk_configs += len(cls.ATR_SL_MULTIPLIERS) * len(cls.ATR_TP_MULTIPLIERS)
            risk_configs += len(cls.MAX_BARS_HOLD)
            
            return strategies * risk_configs

@dataclass
class Trade:
    trade_id: int
    ticker: str
    timeframe: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    position_size: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    is_winner: bool
    bars_held: int
    volume_at_entry: float
    atr_at_entry: float
    strategy_type: str
    strategy_params: Dict
    total_cost: float

@dataclass
class StrategyMetrics:
    ticker: str
    timeframe: str
    strategy_type: str
    strategy_params: Dict
    risk_params: Dict
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    total_return_usd: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    avg_drawdown_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: float
    avg_bars_held: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    exit_sl_pct: float
    exit_tp_pct: float
    exit_time_pct: float
    exit_signal_pct: float
    total_commission: float
    total_slippage: float
    total_costs: float
    composite_score: float
    trades_per_month: float
    avg_trade_duration_days: float
    recovery_factor: float
    stability_score: float

class DataDownloader:
    def __init__(self):
        self.last_request = 0
        self.cache = {}
        if Config.USE_DISK_CACHE:
            Config.CACHE_DIR.mkdir(exist_ok=True)
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < Config.RATE_LIMIT_DELAY:
            time.sleep(Config.RATE_LIMIT_DELAY - elapsed)
        self.last_request = time.time()
    
    def _get_cache_key(self, ticker: str, timeframe: str) -> str:
        """Génère une clé de cache unique"""
        return f"{ticker}_{timeframe}"
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Retourne le chemin du cache disque"""
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return Config.CACHE_DIR / f"{cache_hash}.pkl"
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Charge depuis le cache disque"""
        if not Config.USE_DISK_CACHE:
            return None
        
        cache_path = self._get_disk_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # Vérifier si le cache est récent (moins de 7 jours)
                    if 'timestamp' in data:
                        cache_age = datetime.now() - data['timestamp']
                        if cache_age.days < 7:
                            print(f"  💾 Loaded from cache: {cache_key}")
                            return data['df']
            except Exception:
                pass
        return None
    
    def _save_to_disk_cache(self, cache_key: str, df: pd.DataFrame):
        """Sauvegarde dans le cache disque"""
        if not Config.USE_DISK_CACHE:
            return
        
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            cache_data = {
                'df': df,
                'timestamp': datetime.now(),
                'key': cache_key
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass
    
    def fetch_single(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        cache_key = self._get_cache_key(ticker, timeframe)
        
        # 1. Vérifier le cache mémoire
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 2. Vérifier le cache disque
        cached_df = self._load_from_disk_cache(cache_key)
        if cached_df is not None:
            self.cache[cache_key] = cached_df
            return cached_df
        
        # 3. Télécharger depuis yfinance
        yf_ticker = Config.CRYPTO_TICKERS.get(ticker, ticker)
        yf_interval = Config.TIMEFRAMES[timeframe]
        period = Config.LOOKBACK_PERIODS[timeframe]
        
        for attempt in range(3):
            try:
                self._rate_limit()
                print(f"  📥 {yf_ticker} {timeframe}...", end=' ', flush=True)
                
                is_stock = ticker not in Config.CRYPTO_TICKERS
                data = yf.download(yf_ticker, period=period, interval=yf_interval,
                                 progress=False, auto_adjust=is_stock, threads=False)
                
                if data.empty:
                    print("❌ Empty")
                    continue
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data.columns = [str(c).lower().strip() for c in data.columns]
                required = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required):
                    print(f"❌ Missing columns")
                    continue
                
                data = data[required].copy()
                
                # Conversion en numérique
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                data = data.replace([np.inf, -np.inf], np.nan)
                data = data.dropna()
                
                min_bars = Config.MIN_BARS.get(timeframe, 500)
                if len(data) < min_bars:
                    print(f"❌ Insufficient ({len(data)} < {min_bars})")
                    continue
                
                data['volume'] = data['volume'].replace(0, np.nan)
                data['volume'] = data['volume'].fillna(data['volume'].median())
                data['is_weekend'] = data.index.weekday >= 5
                
                print(f"✅ {len(data)} bars")
                
                # Sauvegarder dans les caches
                self.cache[cache_key] = data
                self._save_to_disk_cache(cache_key, data)
                
                return data
                
            except Exception as e:
                print(f"❌ Error {attempt+1}/3: {str(e)[:50]}")
                if attempt < 2:
                    time.sleep(2)
        
        return None

class Indicators:
    @staticmethod
    def safe_series(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return Indicators.safe_series(rsi.fillna(50))
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        if len(data) < period:
            return pd.Series(data.iloc[-1] if len(data) > 0 else 0, index=data.index)
        return data.ewm(span=period, adjust=False, min_periods=period).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int, std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if len(data) < period:
            mid = pd.Series(data.iloc[-1] if len(data) > 0 else 0, index=data.index)
            return mid, mid, mid
        sma = data.rolling(period, min_periods=period).mean()
        std_dev = data.rolling(period, min_periods=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return (Indicators.safe_series(upper), Indicators.safe_series(sma), Indicators.safe_series(lower))
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        return Indicators.safe_series(atr)

def generate_signals(df: pd.DataFrame, strategy_config: Dict) -> Tuple[pd.Series, pd.Series]:
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)
    strategy_type = strategy_config['type']
    params = strategy_config['params']
    
    try:
        if strategy_type == 'RSI_CROSS':
            rsi_value = params['rsi_value']
            rsi = Indicators.rsi(df['close'], 14)
            buy = (rsi >= rsi_value) & (rsi.shift(1) < rsi_value)
        
        elif strategy_type == 'EMA_TOUCH':
            period = params['ema_period']
            if len(df) >= period:
                ema = Indicators.ema(df['close'], period)
                buy = (df['low'] <= ema) & (df['close'] > ema)
        
        elif strategy_type == 'EMA_CROSSOVER':
            period = params['ema_period']
            if len(df) >= period:
                ema = Indicators.ema(df['close'], period)
                buy = (df['close'] > ema) & (df['close'].shift(1) <= ema.shift(1))
        
        elif strategy_type == 'EMA_REVERSION':
            period = params['ema_period']
            if len(df) >= period:
                ema = Indicators.ema(df['close'], period)
                deviation = ((df['close'] - ema) / ema * 100).abs()
                buy = deviation > 2.0
        
        elif strategy_type == 'BB_TOUCH':
            period = params['bb_period']
            mult = params['bb_multiplier']
            if len(df) >= period:
                upper, mid, lower = Indicators.bollinger_bands(df['close'], period, mult)
                buy = (df['low'] <= lower) & (df['close'] > lower)
        
        elif strategy_type == 'BB_BREAKOUT':
            period = params['bb_period']
            mult = params['bb_multiplier']
            if len(df) >= period:
                upper, mid, lower = Indicators.bollinger_bands(df['close'], period, mult)
                buy = df['close'] < lower
                sell = df['close'] > upper
        
        elif strategy_type == 'BB_REENTRY':
            period = params['bb_period']
            mult = params['bb_multiplier']
            if len(df) >= period:
                upper, mid, lower = Indicators.bollinger_bands(df['close'], period, mult)
                was_outside = (df['close'].shift(1) > upper.shift(1)) | (df['close'].shift(1) < lower.shift(1))
                is_inside = (df['close'] <= upper) & (df['close'] >= lower)
                buy = was_outside & is_inside
    except Exception:
        pass
    
    return buy.fillna(False), sell.fillna(False)

def run_backtest(ticker: str, timeframe: str, df: pd.DataFrame, strategy_config: Dict, risk_config: Dict) -> Tuple[List[Trade], List[float]]:
    trades = []
    capital = Config.INITIAL_CAPITAL
    equity_curve = [capital]
    position = None
    in_position = False
    trade_id = 0
    
    commission_rate = Config.COMMISSIONS.get(timeframe, 0.001)
    slippage_rate = Config.SLIPPAGE.get(timeframe, 0.001)
    atr_series = Indicators.atr(df, 14)
    volume_sma = df['volume'].rolling(20, min_periods=10).mean()
    buy_signals_raw, sell_signals_raw = generate_signals(df, strategy_config)
    
    # Convertir en arrays numpy pour performance
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    dates = df.index
    atr_values = atr_series.values
    buy_arr = buy_signals_raw.values
    sell_arr = sell_signals_raw.values
    
    start_idx = max(Config.WARMUP_BARS, 200)
    
    for i in range(start_idx, len(df)):
        if capital <= 0:
            equity_curve.append(0)
            break
        
        if in_position and position:
            sl_effective = position['sl'] * (1 - slippage_rate)
            tp_effective = position['tp'] * (1 + slippage_rate)
            exit_triggered = False
            exit_price = 0
            exit_reason = "UNKNOWN"
            
            if lows[i] <= sl_effective:
                exit_price = sl_effective
                exit_reason = "SL"
                exit_triggered = True
            elif highs[i] >= tp_effective:
                exit_price = tp_effective
                exit_reason = "TP"
                exit_triggered = True
            elif risk_config['exit_type'] == 'time_based':
                bars_held = i - position['entry_idx']
                if bars_held >= position['max_bars']:
                    exit_price = opens[i] * (1 - slippage_rate)
                    exit_reason = "TIME"
                    exit_triggered = True
            elif risk_config['exit_type'] == 'opposite_signal':
                if i > 0 and sell_arr[i-1]:
                    exit_price = opens[i] * (1 - slippage_rate)
                    exit_reason = "SIGNAL"
                    exit_triggered = True
            
            if exit_triggered:
                entry_final = position['entry_price']
                gross_pnl = (exit_price - entry_final) * position['size']
                commission = (entry_final + exit_price) * position['size'] * commission_rate
                net_pnl = gross_pnl - commission
                capital += net_pnl
                slippage_cost = (entry_final * slippage_rate + exit_price * slippage_rate) * position['size']
                total_cost = commission + slippage_cost
                
                trade = Trade(trade_id=trade_id, ticker=ticker, timeframe=timeframe,
                            entry_date=str(position['entry_date']), exit_date=str(dates[i]),
                            entry_price=entry_final, exit_price=exit_price,
                            sl_price=position['sl'], tp_price=position['tp'],
                            position_size=position['size'], pnl_usd=round(net_pnl, 2),
                            pnl_pct=round((net_pnl / (entry_final * position['size'])) * 100, 2),
                            exit_reason=exit_reason, is_winner=net_pnl > 0,
                            bars_held=i - position['entry_idx'], volume_at_entry=position['volume'],
                            atr_at_entry=position['atr'], strategy_type=strategy_config['type'],
                            strategy_params=strategy_config['params'], total_cost=round(total_cost, 2))
                
                trades.append(trade)
                trade_id += 1
                in_position = False
                position = None
        
        if not in_position:
            if i > 0 and buy_arr[i-1]:
                vol_threshold = volume_sma.iloc[i] * Config.MIN_VOLUME_RATIO
                if pd.notna(vol_threshold) and volumes[i] < vol_threshold:
                    equity_curve.append(capital)
                    continue
                
                atr_val = atr_values[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    equity_curve.append(capital)
                    continue
                
                entry_price = closes[i] * (1 + slippage_rate)
                
                if risk_config['exit_type'] == 'atr_fixed':
                    sl_mult = risk_config.get('sl_multiplier', 2.0)
                    tp_mult = risk_config.get('tp_multiplier', 3.0)
                    sl = entry_price - (atr_val * sl_mult)
                    tp = entry_price + (atr_val * tp_mult)
                    max_bars = 999999
                elif risk_config['exit_type'] == 'time_based':
                    sl = entry_price * 0.95
                    tp = entry_price * 1.10
                    max_bars = risk_config.get('max_bars', 20)
                else:
                    sl = entry_price * 0.98
                    tp = entry_price * 1.05
                    max_bars = 999999
                
                if sl <= 0 or tp <= entry_price or sl >= entry_price:
                    equity_curve.append(capital)
                    continue
                
                risk_per_share = entry_price - sl
                if risk_per_share <= 0:
                    equity_curve.append(capital)
                    continue
                
                risk_amount = capital * Config.RISK_PER_TRADE
                size = risk_amount / risk_per_share
                max_size = (capital * Config.MAX_POSITION_PCT) / entry_price
                size = min(size, max_size)
                
                if size <= 0 or size * entry_price > capital:
                    equity_curve.append(capital)
                    continue
                
                position = {'entry_idx': i, 'entry_date': dates[i], 'entry_price': entry_price,
                          'sl': sl, 'tp': tp, 'size': size, 'volume': volumes[i],
                          'atr': atr_val, 'max_bars': max_bars}
                in_position = True
        
        current_equity = capital
        if in_position and position:
            unrealized = (closes[i] - position['entry_price']) * position['size']
            current_equity = capital + unrealized
        equity_curve.append(current_equity)
    
    return trades, equity_curve

def calculate_metrics(ticker: str, timeframe: str, trades: List[Trade], equity_curve: List[float],
                     strategy_config: Dict, risk_config: Dict) -> Optional[StrategyMetrics]:
    if not trades or len(trades) < Config.MIN_TRADES:
        return None
    
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]
    total = len(trades)
    win_rate = (len(winners) / total * 100) if total > 0 else 0
    total_wins = sum(t.pnl_usd for t in winners) if winners else 0
    total_losses = abs(sum(t.pnl_usd for t in losers)) if losers else 0.01
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    final = equity_curve[-1]
    total_return_usd = final - Config.INITIAL_CAPITAL
    total_return_pct = (total_return_usd / Config.INITIAL_CAPITAL) * 100
    avg_win = (sum(t.pnl_pct for t in winners) / len(winners)) if winners else 0
    avg_loss = (sum(t.pnl_pct for t in losers) / len(losers)) if losers else 0
    win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0
    
    peak = Config.INITIAL_CAPITAL
    max_dd_pct = 0
    max_dd_usd = 0
    drawdowns = []
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd_usd = peak - equity
        dd_pct = (dd_usd / peak * 100) if peak > 0 else 0
        drawdowns.append(dd_pct)
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_usd = dd_usd
    
    avg_dd = np.mean([d for d in drawdowns if d > 0]) if any(d > 0 for d in drawdowns) else 0
    recovery = abs(total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
    
    consec_wins = 0
    consec_losses = 0
    max_consec_wins = 0
    max_consec_losses = 0
    for trade in trades:
        if trade.is_winner:
            consec_wins += 1
            consec_losses = 0
            max_consec_wins = max(max_consec_wins, consec_wins)
        else:
            consec_losses += 1
            consec_wins = 0
            max_consec_losses = max(max_consec_losses, consec_losses)
    
    returns = pd.Series(equity_curve).pct_change().dropna()
    bars_per_year = {'H1': 252*6.5, 'H4': 252*1.625, 'D1': 252, 'W1': 52}
    annual_factor = bars_per_year.get(timeframe, 252)
    
    sharpe = 0
    if len(returns) > 0 and returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(annual_factor)
    
    sortino = 0
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() != 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(annual_factor)
    
    calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
    
    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    exit_sl = (exit_counts.get('SL', 0) / total * 100)
    exit_tp = (exit_counts.get('TP', 0) / total * 100)
    exit_time = (exit_counts.get('TIME', 0) / total * 100)
    exit_signal = (exit_counts.get('SIGNAL', 0) / total * 100)
    
    total_commission = sum(t.total_cost for t in trades)
    total_slippage = total_commission * 0.5
    total_costs = total_commission + total_slippage
    
    pf_norm = min(profit_factor / 3.0, 1.0)
    dd_norm = max(1 - (max_dd_pct / 100), 0)
    wr_norm = win_rate / 100
    trades_norm = min(np.log10(total + 1) / 3.0, 1.0)
    composite = pf_norm * dd_norm * wr_norm * trades_norm * 100
    
    duration_days = (pd.to_datetime(trades[-1].exit_date) - pd.to_datetime(trades[0].entry_date)).days
    trades_per_month = (total / max(duration_days / 30, 1)) if duration_days > 0 else 0
    avg_duration = np.mean([t.bars_held for t in trades])
    bars_to_days = {'H1': 1/6.5, 'H4': 1/1.625, 'D1': 1, 'W1': 7}
    avg_trade_days = avg_duration * bars_to_days.get(timeframe, 1)
    stability = 1 / (1 + returns.std()) if len(returns) > 0 and returns.std() > 0 else 0
    
    return StrategyMetrics(ticker=ticker, timeframe=timeframe, strategy_type=strategy_config['type'],
                          strategy_params=strategy_config['params'], risk_params=risk_config,
                          total_trades=total, win_rate=round(win_rate, 2), profit_factor=round(profit_factor, 2),
                          total_return_pct=round(total_return_pct, 2), total_return_usd=round(total_return_usd, 2),
                          max_drawdown_pct=round(max_dd_pct, 2), max_drawdown_usd=round(max_dd_usd, 2),
                          avg_drawdown_pct=round(avg_dd, 2), avg_win_pct=round(avg_win, 2),
                          avg_loss_pct=round(avg_loss, 2), win_loss_ratio=round(win_loss_ratio, 2),
                          avg_bars_held=round(avg_duration, 1), sharpe_ratio=round(sharpe, 2),
                          sortino_ratio=round(sortino, 2), calmar_ratio=round(calmar, 2),
                          max_consecutive_wins=max_consec_wins, max_consecutive_losses=max_consec_losses,
                          exit_sl_pct=round(exit_sl, 1), exit_tp_pct=round(exit_tp, 1),
                          exit_time_pct=round(exit_time, 1), exit_signal_pct=round(exit_signal, 1),
                          total_commission=round(total_commission, 2), total_slippage=round(total_slippage, 2),
                          total_costs=round(total_costs, 2), composite_score=round(composite, 2),
                          trades_per_month=round(trades_per_month, 2), avg_trade_duration_days=round(avg_trade_days, 1),
                          recovery_factor=round(recovery, 2), stability_score=round(stability, 3))

def generate_combinations() -> List[Tuple[Dict, Dict]]:
    strategies = []
    risk_configs = []
    
    if Config.MODE == 'TEST':
        for rsi_val in [25, 30, 35]:
            strategies.append({'type': 'RSI_CROSS', 'params': {'rsi_value': rsi_val}})
        for period in [20, 50, 100]:
            for ema_type in ['TOUCH', 'CROSSOVER']:
                strategies.append({'type': f'EMA_{ema_type}', 'params': {'ema_period': period}})
        for period in [20]:
            for mult in [2.0, 2.5]:
                for bb_type in ['TOUCH', 'BREAKOUT']:
                    strategies.append({'type': f'BB_{bb_type}', 'params': {'bb_period': period, 'bb_multiplier': mult}})
        risk_configs = [{'exit_type': 'opposite_signal'},
                       {'exit_type': 'atr_fixed', 'sl_multiplier': 2.0, 'tp_multiplier': 3.0},
                       {'exit_type': 'time_based', 'max_bars': 20}]
    else:
        # FULL MODE - paramètres complets mais optimisés
        for rsi_val in Config.RSI_BUY_RANGE:
            strategies.append({'type': 'RSI_CROSS', 'params': {'rsi_value': rsi_val}})
        
        for period in Config.EMA_PERIODS:
            for ema_type in ['TOUCH', 'CROSSOVER', 'REVERSION']:
                strategies.append({'type': f'EMA_{ema_type}', 'params': {'ema_period': period}})
        
        for period in Config.BB_PERIODS:
            for mult in Config.BB_MULTIPLIERS:
                for bb_type in ['TOUCH', 'BREAKOUT', 'REENTRY']:
                    strategies.append({'type': f'BB_{bb_type}', 'params': {'bb_period': period, 'bb_multiplier': mult}})
        
        risk_configs.append({'exit_type': 'opposite_signal'})
        
        for sl in Config.ATR_SL_MULTIPLIERS:
            for tp in Config.ATR_TP_MULTIPLIERS:
                if tp > sl:
                    risk_configs.append({'exit_type': 'atr_fixed', 'sl_multiplier': sl, 'tp_multiplier': tp})
        
        for max_bars in Config.MAX_BARS_HOLD:
            risk_configs.append({'exit_type': 'time_based', 'max_bars': max_bars})
    
    combos = []
    for strat in strategies:
        for risk in risk_configs:
            combos.append((strat, risk))
    
    print(f"  📊 Generated {len(combos)} strategy/risk combinations")
    return combos

def backtest_worker(args):
    """Worker function for multiprocessing"""
    try:
        ticker, timeframe, strategy_config, risk_config, df_data = args
        
        # Reconstruire le DataFrame
        df = pd.DataFrame(
            data=df_data['data'],
            index=pd.to_datetime(df_data['index']),
            columns=df_data['columns']
        )
        
        trades, equity = run_backtest(ticker, timeframe, df, strategy_config, risk_config)
        
        if len(trades) < Config.MIN_TRADES:
            return None
        
        metrics = calculate_metrics(ticker, timeframe, trades, equity, strategy_config, risk_config)
        
        if metrics is None:
            return None
        
        return {'metrics': metrics, 'trades': trades}
        
    except Exception as e:
        return None

def optimize_timeframe_parallel(ticker: str, timeframe: str, df: pd.DataFrame) -> Tuple[List[StrategyMetrics], List[Trade]]:
    """Version parallélisée pour le mode FULL"""
    print(f"\n{'='*70}")
    print(f"🔍 {ticker} | {timeframe}")
    print('='*70)
    
    combos = generate_combinations()
    
    # Préparer les données pour le multiprocessing
    df_data = {
        'data': {col: df[col].tolist() for col in df.columns},
        'index': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'columns': df.columns.tolist()
    }
    
    # Créer les tâches
    tasks = [(ticker, timeframe, strat, risk, df_data) for strat, risk in combos]
    
    all_metrics = []
    all_trades = []
    
    print(f"  ⚡ Using {Config.WORKERS} parallel workers")
    print(f"  📈 Testing {len(tasks)} combinations...")
    
    start_time = time.time()
    
    # Utiliser multiprocessing pour le mode FULL
    with Pool(processes=Config.WORKERS) as pool:
        # Diviser en chunks pour mieux gérer la mémoire
        chunk_size = max(100, len(tasks) // (Config.WORKERS * 4))
        results = list(pool.imap(backtest_worker, tasks, chunksize=chunk_size))
    
    # Traiter les résultats
    for result in results:
        if result:
            all_metrics.append(result['metrics'])
            if result['trades']:
                all_trades.extend(result['trades'])
    
    elapsed = time.time() - start_time
    print(f"  ⏱️  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    # Trier par score
    all_metrics.sort(key=lambda x: x.composite_score, reverse=True)
    
    print(f"  ✅ {len(all_metrics)} valid strategies found")
    
    if all_metrics:
        print(f"\n  🏆 TOP 5:")
        for i, m in enumerate(all_metrics[:5], 1):
            trades_hour = (m.trades_per_month * 12) / (24 * 365) if m.avg_trade_duration_days > 0 else 0
            print(f"    {i}. {m.strategy_type:15} | Score={m.composite_score:5.1f} | "
                  f"PF={m.profit_factor:4.2f} | WR={m.win_rate:5.1f}% | "
                  f"Trades={m.total_trades:3d}")
    
    return all_metrics, all_trades

def save_results(all_metrics: List[StrategyMetrics], all_trades: List[Trade]):
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print('='*70)
    
    if not all_metrics:
        print("  ❌ No results to save")
        return
    
    # Sauvegarder les métriques
    df_metrics = pd.DataFrame([asdict(m) for m in all_metrics])
    df_metrics['strategy_params_str'] = df_metrics['strategy_params'].apply(str)
    df_metrics['risk_params_str'] = df_metrics['risk_params'].apply(str)
    
    metrics_file = Config.OUTPUT_DIR / 'all_metrics.csv'
    df_metrics.to_csv(metrics_file, index=False)
    print(f"  ✅ Metrics: {len(df_metrics)} strategies ({metrics_file.name})")
    
    # Top 200 stratégies
    top_200 = df_metrics.nlargest(min(200, len(df_metrics)), 'composite_score')
    top_file = Config.OUTPUT_DIR / 'top_strategies.csv'
    top_200.to_csv(top_file, index=False)
    print(f"  ✅ Top strategies: {top_file.name}")
    
    # Statistiques par timeframe
    summary = {}
    for tf in Config.TIMEFRAMES:
        tf_data = df_metrics[df_metrics['timeframe'] == tf]
        if len(tf_data) > 0:
            best = tf_data.iloc[0]
            summary[tf] = {
                'count': len(tf_data),
                'avg_composite': float(tf_data['composite_score'].mean()),
                'avg_profit_factor': float(tf_data['profit_factor'].mean()),
                'avg_win_rate': float(tf_data['win_rate'].mean()),
                'best_strategy': best['strategy_type'],
                'best_composite': float(best['composite_score']),
                'best_profit_factor': float(best['profit_factor'])
            }
    
    summary_file = Config.OUTPUT_DIR / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary: {summary_file.name}")
    
    # Sauvegarder les trades
    if all_trades:
        trades_file = Config.OUTPUT_DIR / 'all_trades.csv'
        df_trades = pd.DataFrame([asdict(t) for t in all_trades])
        df_trades.to_csv(trades_file, index=False)
        print(f"  ✅ Trades: {len(df_trades)} trades ({trades_file.name})")
    
    # Sauvegarder la configuration
    config_summary = {
        'timestamp': Config.timestamp,
        'total_combinations': Config.total_combinations(),
        'tickers': {
            'crypto': list(Config.CRYPTO_TICKERS.keys()),
            'stocks': Config.STOCK_TICKERS
        },
        'timeframes': list(Config.TIMEFRAMES.keys()),
        'mode': Config.MODE,
        'workers': Config.WORKERS
    }
    
    config_file = Config.OUTPUT_DIR / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config_summary, f, indent=2)
    print(f"  ✅ Configuration: {config_file.name}")

def main():
    print("\n╔" + "═"*76 + "╗")
    print("║" + " BACKTEST ENGINE v14.0 - FULL MODE OPTIMIZED ".center(76) + "║")
    print("╚" + "═"*76 + "╝\n")
    
    if not HAS_YFINANCE:
        print("❌ Please install: pip install yfinance pandas numpy")
        return
    
    print(f"⚙️  CONFIGURATION:")
    print(f"   Mode: {Config.MODE}")
    print(f"   Workers: {Config.WORKERS}")
    print(f"   Output: {Config.OUTPUT_DIR}")
    print(f"   Cache: {'Enabled' if Config.USE_DISK_CACHE else 'Disabled'}")
    
    total_actifs = len(Config.CRYPTO_TICKERS) + len(Config.STOCK_TICKERS)
    total_timeframes = len(Config.TIMEFRAMES)
    total_combos = Config.total_combinations()
    
    print(f"\n📊 SCOPE:")
    print(f"   Assets: {total_actifs} ({len(Config.CRYPTO_TICKERS)} crypto, {len(Config.STOCK_TICKERS)} stocks)")
    print(f"   Timeframes: {total_timeframes}")
    print(f"   Tests per asset/timeframe: ~{total_combos}")
    print(f"   Total tests: ~{total_actifs * total_timeframes * total_combos:,}")
    print(f"   Estimated time: 4-8 hours (with {Config.WORKERS} workers)")
    
    print(f"\n{'='*70}")
    print("📥 DOWNLOADING DATA")
    print('='*70)
    
    # Afficher le plan d'exécution
    print("\n📋 EXECUTION PLAN:")
    all_tickers = list(Config.CRYPTO_TICKERS.keys()) + Config.STOCK_TICKERS
    for ticker in all_tickers:
        for timeframe in Config.TIMEFRAMES:
            print(f"   • {ticker} | {timeframe}")
    
    downloader = DataDownloader()
    datasets = {}
    
    start_download = time.time()
    for ticker in all_tickers:
        for timeframe in Config.TIMEFRAMES:
            df = downloader.fetch_single(ticker, timeframe)
            if df is not None:
                datasets[(ticker, timeframe)] = df
    
    download_time = time.time() - start_download
    print(f"\n✅ {len(datasets)} datasets ready ({download_time:.1f}s)")
    
    if not datasets:
        print("\n❌ No data available")
        return
    
    print(f"\n{'='*70}")
    print("🔬 RUNNING BACKTESTS")
    print('='*70)
    
    total_start = time.time()
    all_metrics = []
    all_trades = []
    
    print(f"\n⚠️  WARNING: This will take approximately 4-8 hours")
    print("   Press Ctrl+C to stop at any time")
    print("   Results are saved incrementally\n")
    
    for i, ((ticker, timeframe), df) in enumerate(datasets.items(), 1):
        print(f"\n[{i}/{len(datasets)}] {ticker} | {timeframe}")
        print(f"   Data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        
        try:
            metrics, trades = optimize_timeframe_parallel(ticker, timeframe, df)
            all_metrics.extend(metrics)
            all_trades.extend(trades)
            
            # Sauvegarde incrémentale
            if metrics:
                temp_file = Config.OUTPUT_DIR / f'partial_{ticker}_{timeframe}.csv'
                pd.DataFrame([asdict(m) for m in metrics]).to_csv(temp_file, index=False)
                print(f"   💾 Partial results saved: {len(metrics)} strategies")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  INTERRUPTED BY USER")
            print("   Saving current results...")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_metrics:
        print("\n❌ No valid strategies found")
        return
    
    # Sauvegarde finale
    save_results(all_metrics, all_trades)
    
    total_time = time.time() - total_start
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    
    print(f"\n{'='*70}")
    print("✅ COMPLETED")
    print('='*70)
    print(f"   Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"   Strategies: {len(all_metrics)}")
    print(f"   Trades: {len(all_trades)}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   - Average strategies per dataset: {len(all_metrics)/len(datasets):.1f}")
    print(f"   - Average trades per dataset: {len(all_trades)/len(datasets):.1f}")
    if all_metrics:
        best_score = max(m.composite_score for m in all_metrics)
        avg_pf = np.mean([m.profit_factor for m in all_metrics])
        avg_wr = np.mean([m.win_rate for m in all_metrics])
        print(f"   - Best composite score: {best_score:.1f}")
        print(f"   - Average profit factor: {avg_pf:.2f}")
        print(f"   - Average win rate: {avg_wr:.1f}%")
    
    print(f"\n📁 Files created:")
    for file in Config.OUTPUT_DIR.glob('*'):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   • {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED")
        print("   Partial results have been saved")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
