#!/usr/bin/env python3
"""
BACKTEST ENGINE v15.0 - GÉNÉRATEUR DE DATASET POUR IA
Toutes les stratégies brute force, aucune interprétation, données brutes seulement
"""

import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import json
import time
import warnings
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
import gc
import hashlib
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration complète du backtest v15"""
    
    # === DONNÉES ===
    CRYPTO_TICKERS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
        'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'LTC/USDT'
    ]
    
    STOCK_TICKERS = [
        'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL',
        'AMZN', 'META', 'JPM', 'V', 'WMT'
    ]
    
    TIMEFRAMES = {
        'M15': '15m', 'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w'
    }
    
    YEARS_OF_DATA = 5
    LOOKBACK_DAYS = {
        'M15': 365 * 5, 'H1': 365 * 5, 'H4': 365 * 5, 
        'D1': 365 * 5, 'W1': 365 * 5
    }
    
    # === STRATÉGIES BRUTE FORCE ===
    # RSI
    RSI_PERIODS = [7, 14, 21, 28]
    RSI_BUY_VALUES = list(range(0, 46))      # 0-45
    RSI_SELL_VALUES = list(range(55, 101))   # 55-100
    RSI_SIGNAL_TYPES = ['CROSS', 'CROSS_CONFIRM', 'REVERSION', 'DIVERGENCE']
    
    # EMA
    EMA_PERIODS = list(range(5, 201, 5))     # 5 à 200 par pas de 5
    EMA_TYPES = ['TOUCH', 'CROSSOVER', 'FAST_SLOW_CROSS', 'CLUSTER', 'DISTANCE']
    EMA_FAST_SLOW_COMBOS = [(5,20), (10,30), (20,50), (50,100), (100,200)]
    
    # Bollinger Bands
    BB_PERIODS = [10, 15, 20, 25, 30, 40, 50]
    BB_STD_DEVS = [1.0, 1.5, 2.0, 2.5, 3.0]
    BB_TYPES = ['TOUCH_LOWER', 'TOUCH_UPPER', 'BREAKOUT', 'REENTRY', 'SQUEEZE']
    
    # MACD
    MACD_FAST = [8, 12, 16]
    MACD_SLOW = [21, 26, 34]
    MACD_SIGNAL = [7, 9, 13]
    MACD_TYPES = ['CROSSOVER', 'ZERO_CROSS', 'DIVERGENCE']
    
    # Volume
    VOLUME_PERIODS = [5, 10, 20]
    VOLUME_MULTIPLIERS = [1.5, 2.0, 3.0]
    VOLUME_TYPES = ['SPIKE', 'DECLINE', 'DIVERGENCE']
    
    # Support/Résistance
    SR_PERIODS = [20, 50, 100]
    SR_TYPES = ['BOUNCE', 'BREAK', 'RETEST']
    SR_CONFIRMATIONS = [1, 2, 3]
    
    # === FILTRES ===
    VOLUME_FILTER_VALUES = [0.8, 1.0, 1.2, 1.5, 2.0]
    TREND_FILTER_ADX = [20, 25, 30]
    VOLATILITY_FILTER_ATR = [0.5, 1.0, 2.0]
    
    # === GESTION DU RISQUE ===
    STOP_LOSS_TYPES = ['FIXED_PCT', 'ATR_MULTIPLE', 'SUPPORT', 'TRAILING']
    STOP_LOSS_VALUES = [1.0, 2.0, 3.0, 5.0]
    
    TAKE_PROFIT_TYPES = ['FIXED_RR', 'MULTIPLE_TARGETS', 'DYNAMIC', 'RESISTANCE']
    TAKE_PROFIT_VALUES = [1.5, 2.0, 3.0, 4.0]
    
    POSITION_SIZING_TYPES = ['FIXED_PCT', 'KELLY', 'VOLATILITY_ADJUSTED']
    POSITION_SIZING_VALUES = [1.0, 2.0, 5.0]
    
    # === WALK-FORWARD ===
    WF_FOLDS = 5
    WF_TRAIN_RATIO = 0.7
    WF_VALIDATION_TYPES = ['EXPANDING', 'ROLLING']
    
    # === OPTIMISATION ===
    CPU_CORES = max(1, cpu_count() - 2)
    CHUNK_SIZE = 10000
    MAX_MEMORY_GB = 32
    CACHE_ENABLED = True
    
    # === SORTIE ===
    OUTPUT_FORMAT = 'parquet'
    COMPRESSION = 'zstd'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path.cwd() / f'backtest_dataset_{timestamp}'
    
    # === CONSTANTES ===
    INITIAL_CAPITAL = 10000.0
    COMMISSION_RATE = 0.001
    SLIPPAGE_RATE = 0.001
    MIN_DATA_POINTS = 1000
    MIN_VOLUME = 1000
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# CLASSES DE DONNÉES
# ============================================================================

@dataclass
class Trade:
    """Trade complet avec 50+ features de contexte"""
    # Identifiant
    trade_id: str
    strategy_id: str
    ticker: str
    timeframe: str
    
    # Timing
    entry_timestamp: str
    exit_timestamp: str
    duration_bars: int
    duration_hours: float
    
    # Prix
    entry_price: float
    exit_price: float
    stop_loss_price: float
    take_profit_price: float
    
    # Position
    position_size: float
    position_value_usd: float
    
    # P&L
    pnl_absolute: float
    pnl_percentage: float
    pnl_commission: float
    pnl_slippage: float
    pnl_net: float
    
    # Sortie
    exit_reason: str
    exit_type: str
    
    # Contexte technique
    entry_rsi_14: float
    entry_ema_20: float
    entry_bb_width_pct: float
    entry_adx: float
    entry_atr: float
    entry_atr_pct: float
    entry_volume: float
    entry_volume_ratio: float
    
    # Régime marché
    market_regime: str
    trend_strength: str
    volatility_regime: str
    volume_regime: str
    
    # Structure prix
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    price_position_in_range: float
    is_swing_low: bool
    is_swing_high: bool
    
    # Momentum
    rsi_slope_5: float
    macd_histogram: float
    momentum_roc_10: float
    candle_body_ratio: float
    
    # Contexte temporel
    session: str
    hour_of_day: int
    day_of_week: int
    month: int
    quarter: int
    is_weekend: bool
    is_market_open: bool
    
    # Contexte global
    vix_level: float = 0.0
    btc_dominance: float = 0.0
    fear_greed_index: float = 0.0
    usd_index: float = 0.0
    
    # Métadonnées
    data_quality_score: float = 1.0
    liquidity_score: float = 1.0
    spread_impact: float = 0.0

@dataclass
class StrategyMetrics:
    """Métriques brutes d'une stratégie - PAS DE SCORE COMPOSITE"""
    # Identifiant
    strategy_id: str
    strategy_family: str
    strategy_params: Dict
    risk_params: Dict
    filter_params: Dict
    
    ticker: str
    timeframe: str
    
    # Performance brute
    total_trades: int
    winning_trades: int
    losing_trades: int
    break_even_trades: int
    
    win_rate: float
    profit_factor: float
    expectancy: float
    total_return_pct: float
    total_return_usd: float
    
    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_usd: float
    max_drawdown_duration_days: int
    avg_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    ulcer_index: float
    var_95: float
    
    # Trade statistics
    avg_win_pct: float
    avg_loss_pct: float
    avg_trade_duration_hours: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_bars_held: float
    time_in_market_pct: float
    
    # Distribution
    pnl_skewness: float
    pnl_kurtosis: float
    win_pnl_std: float
    loss_pnl_std: float
    
    # Walk-forward (JSON sérialisé)
    wf_results: str
    wf_win_rate_stability: float
    wf_profit_factor_stability: float
    wf_degradation_score: float
    
    # Performance par régime (JSON sérialisé)
    performance_by_regime: str
    
    # Sensibilités
    volatility_sensitivity: float
    trend_sensitivity: float
    volume_sensitivity: float
    time_sensitivity: float
    liquidity_sensitivity: float
    
    # Métadonnées
    data_points_used: int
    data_quality: float
    calculation_time_seconds: float
    test_period_start: str
    test_period_end: str

# ============================================================================
# MOTEUR DE DONNÉES
# ============================================================================

class DataFetcher:
    """Télécharge les données depuis Binance (crypto) et yfinance (actions)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1200,
            'options': {'defaultType': 'spot'}
        })
        self.cache = {}
        
    def fetch_all_data(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Télécharge toutes les données"""
        all_data = {}
        
        # Cryptos via Binance
        print("📥 Téléchargement des données crypto...")
        for ticker in self.config.CRYPTO_TICKERS:
            for tf_name, tf_value in self.config.TIMEFRAMES.items():
                data = self.fetch_binance_data(ticker, tf_value)
                if data is not None and len(data) > self.config.MIN_DATA_POINTS:
                    all_data[(ticker, tf_name)] = data
                    print(f"  ✅ {ticker} {tf_name}: {len(data)} bougies")
        
        # Actions via yfinance
        print("\n📥 Téléchargement des données actions...")
        for ticker in self.config.STOCK_TICKERS:
            for tf_name, tf_value in self.config.TIMEFRAMES.items():
                data = self.fetch_yfinance_data(ticker, tf_value)
                if data is not None and len(data) > self.config.MIN_DATA_POINTS:
                    all_data[(ticker, tf_name)] = data
                    print(f"  ✅ {ticker} {tf_name}: {len(data)} bougies")
        
        return all_data
    
    def fetch_binance_data(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Télécharge les données Binance avec pagination"""
        cache_key = f"binance_{ticker}_{timeframe}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            since = self.binance.parse8601(
                (datetime.now() - timedelta(days=self.config.LOOKBACK_DAYS['D1']))
                .strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            all_ohlcv = []
            while True:
                ohlcv = self.binance.fetch_ohlcv(
                    ticker, 
                    timeframe, 
                    since=since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if len(ohlcv) < 1000:
                    break
                
                time.sleep(self.binance.rateLimit / 1000)
            
            if not all_ohlcv:
                return None
            
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Nettoyage basique
            df = df.replace(0, np.nan).ffill().bfill()
            
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"❌ Erreur Binance {ticker} {timeframe}: {e}")
            return None
    
    def fetch_yfinance_data(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Télécharge les données yfinance"""
        cache_key = f"yfinance_{ticker}_{timeframe}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Mapping timeframe
            tf_mapping = {
                '15m': '15m', '1h': '60m', '4h': '240m',
                '1d': '1d', '1w': '1wk'
            }
            
            period = f"{self.config.YEARS_OF_DATA}y"
            interval = tf_mapping.get(timeframe, '1d')
            
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                return None
            
            data.columns = [col.lower() for col in data.columns]
            required = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in data.columns for col in required):
                return None
            
            df = data[required].copy()
            df = df.replace(0, np.nan).ffill().bfill()
            
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"❌ Erreur yfinance {ticker} {timeframe}: {e}")
            return None

# ============================================================================
# INDICATEURS TECHNIQUES (VECTORISÉS)
# ============================================================================

class TechnicalIndicators:
    """Calcul vectorisé de tous les indicateurs techniques"""
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSI vectorisé"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """EMA vectorisé"""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int, std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands vectorisé"""
        sma = series.rolling(window=period, min_periods=period).mean()
        std_dev = series.rolling(window=period, min_periods=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper.fillna(method='bfill'), sma.fillna(method='bfill'), lower.fillna(method='bfill')
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR vectorisé"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX vectorisé (Directional Movement Index)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Plus Directional Movement (+DM)
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        
        # Minus Directional Movement (-DM)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Smoothing
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / 
                         tr.rolling(period).mean())
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / 
                          tr.rolling(period).mean())
        
        # ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD vectorisé"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator vectorisé"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=k_period).mean()
        
        return k.fillna(50), d.fillna(50)

# ============================================================================
# CACHE D'INDICATEURS
# ============================================================================

class IndicatorsCache:
    """Cache hiérarchique pour tous les indicateurs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache = {}
        self.memory_cache = {}
        
    def precompute_for_data(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """Pré-calculer tous les indicateurs pour un dataframe"""
        cache_key = f"{ticker}_{timeframe}"
        
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        indicators = {}
        
        # RSI pour toutes périodes
        for period in self.config.RSI_PERIODS:
            key = f"rsi_{period}"
            indicators[key] = TechnicalIndicators.rsi(df['close'], period)
        
        # EMA pour périodes principales
        for period in [5, 10, 20, 50, 100, 200]:
            key = f"ema_{period}"
            indicators[key] = TechnicalIndicators.ema(df['close'], period)
        
        # Bollinger Bands pour combinaisons principales
        for period in [20, 50]:
            for std in [1.5, 2.0, 2.5]:
                key = f"bb_{period}_{std}"
                upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'], period, std)
                indicators[f"{key}_upper"] = upper
                indicators[f"{key}_middle"] = middle
                indicators[f"{key}_lower"] = lower
        
        # ATR
        indicators['atr_14'] = TechnicalIndicators.atr(df, 14)
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.adx(df, 14)
        indicators['adx_14'] = adx
        indicators['plus_di_14'] = plus_di
        indicators['minus_di_14'] = minus_di
        
        # Volume
        indicators['volume_sma_20'] = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma_20']
        
        self.memory_cache[cache_key] = indicators
        return indicators
    
    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, indicator: str, **params):
        """Récupérer un indicateur depuis le cache ou le calculer"""
        cache_key = f"{ticker}_{timeframe}_{indicator}_{'_'.join(map(str, params.values()))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if indicator == 'rsi':
            result = TechnicalIndicators.rsi(df['close'], params.get('period', 14))
        elif indicator == 'ema':
            result = TechnicalIndicators.ema(df['close'], params['period'])
        elif indicator == 'bb':
            result = TechnicalIndicators.bollinger_bands(df['close'], params['period'], params['std'])
        elif indicator == 'atr':
            result = TechnicalIndicators.atr(df, params.get('period', 14))
        elif indicator == 'adx':
            result = TechnicalIndicators.adx(df, params.get('period', 14))
        elif indicator == 'macd':
            result = TechnicalIndicators.macd(
                df['close'], 
                params.get('fast', 12), 
                params.get('slow', 26), 
                params.get('signal', 9)
            )
        else:
            raise ValueError(f"Indicateur non supporté: {indicator}")
        
        self.cache[cache_key] = result
        return result

# ============================================================================
# CONTEXTE DE MARCHÉ
# ============================================================================

class MarketContext:
    """Calcule le contexte de marché pour chaque bougie"""
    
    @staticmethod
    def calculate_market_regime(df: pd.DataFrame, indicators: Dict) -> pd.Series:
        """Détermine le régime de marché"""
        adx = indicators.get('adx_14', pd.Series(0, index=df.index))
        atr = indicators.get('atr_14', pd.Series(0, index=df.index))
        
        # Seuils
        adx_threshold = 25
        atr_median = atr.rolling(100).median()
        
        conditions = []
        regimes = []
        
        for i in range(len(df)):
            if i < 100:
                regimes.append('UNKNOWN')
                continue
            
            adx_val = adx.iloc[i] if i < len(adx) else 0
            atr_val = atr.iloc[i] if i < len(atr) else 0
            atr_med = atr_median.iloc[i]
            
            # Détermination du régime
            if adx_val > adx_threshold:
                if atr_val > atr_med * 1.5:
                    regime = 'TRENDING_HIGH_VOL'
                else:
                    regime = 'TRENDING_NORMAL_VOL'
            else:
                if atr_val > atr_med * 1.5:
                    regime = 'RANGING_HIGH_VOL'
                else:
                    regime = 'RANGING_NORMAL_VOL'
            
            regimes.append(regime)
        
        return pd.Series(regimes, index=df.index)
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, lookback: int = 100) -> Tuple[pd.Series, pd.Series]:
        """Calcule les niveaux de support et résistance"""
        supports = pd.Series(np.nan, index=df.index)
        resistances = pd.Series(np.nan, index=df.index)
        
        for i in range(lookback, len(df)):
            window = df['low'].iloc[i-lookback:i]
            supports.iloc[i] = window.min()
            
            window = df['high'].iloc[i-lookback:i]
            resistances.iloc[i] = window.max()
        
        return supports.ffill(), resistances.ffill()
    
    @staticmethod
    def calculate_session(timestamp: pd.Timestamp) -> str:
        """Détermine la session de trading"""
        hour = timestamp.hour
        
        if 0 <= hour < 8:
            return 'ASIA'
        elif 8 <= hour < 16:
            return 'LONDON'
        elif 16 <= hour < 24:
            return 'NY'
        else:
            return 'GLOBAL'

# ============================================================================
# GÉNÉRATEUR DE STRATÉGIES BRUTE FORCE
# ============================================================================

class StrategyGenerator:
    """Génère toutes les combinaisons de stratégies brute force"""
    
    def __init__(self, config: Config):
        self.config = config
        self.strategies = []
    
    def generate_all_strategies(self):
        """Génère toutes les stratégies à tester"""
        print("🧠 Génération des stratégies brute force...")
        
        # RSI brute force
        rsi_strategies = self._generate_rsi_strategies()
        print(f"  RSI: {len(rsi_strategies)} combinaisons")
        
        # EMA brute force
        ema_strategies = self._generate_ema_strategies()
        print(f"  EMA: {len(ema_strategies)} combinaisons")
        
        # Bollinger Bands brute force
        bb_strategies = self._generate_bb_strategies()
        print(f"  BB: {len(bb_strategies)} combinaisons")
        
        # MACD brute force
        macd_strategies = self._generate_macd_strategies()
        print(f"  MACD: {len(macd_strategies)} combinaisons")
        
        self.strategies = rsi_strategies + ema_strategies + bb_strategies + macd_strategies
        print(f"🎯 Total: {len(self.strategies)} stratégies")
        
        return self.strategies
    
    def _generate_rsi_strategies(self) -> List[Dict]:
        """Génère toutes les combinaisons RSI"""
        strategies = []
        
        for period in self.config.RSI_PERIODS:
            for buy_thresh in self.config.RSI_BUY_VALUES:
                for sell_thresh in self.config.RSI_SELL_VALUES:
                    if buy_thresh >= sell_thresh:
                        continue
                    
                    for signal_type in self.config.RSI_SIGNAL_TYPES:
                        strategy = {
                            'family': 'RSI',
                            'params': {
                                'period': period,
                                'buy_threshold': buy_thresh,
                                'sell_threshold': sell_thresh,
                                'signal_type': signal_type
                            }
                        }
                        strategies.append(strategy)
        
        return strategies
    
    def _generate_ema_strategies(self) -> List[Dict]:
        """Génère toutes les combinaisons EMA"""
        strategies = []
        
        for ema_type in self.config.EMA_TYPES:
            if ema_type == 'FAST_SLOW_CROSS':
                for fast, slow in self.config.EMA_FAST_SLOW_COMBOS:
                    strategy = {
                        'family': 'EMA',
                        'params': {
                            'type': ema_type,
                            'fast_period': fast,
                            'slow_period': slow
                        }
                    }
                    strategies.append(strategy)
            else:
                for period in self.config.EMA_PERIODS[:20]:  # Limité à 20 pour performance
                    strategy = {
                        'family': 'EMA',
                        'params': {
                            'type': ema_type,
                            'period': period
                        }
                    }
                    strategies.append(strategy)
        
        return strategies
    
    def _generate_bb_strategies(self) -> List[Dict]:
        """Génère toutes les combinaisons Bollinger Bands"""
        strategies = []
        
        for period in self.config.BB_PERIODS:
            for std in self.config.BB_STD_DEVS:
                for bb_type in self.config.BB_TYPES:
                    strategy = {
                        'family': 'BB',
                        'params': {
                            'period': period,
                            'std_dev': std,
                            'type': bb_type
                        }
                    }
                    strategies.append(strategy)
        
        return strategies
    
    def _generate_macd_strategies(self) -> List[Dict]:
        """Génère toutes les combinaisons MACD"""
        strategies = []
        
        for fast in self.config.MACD_FAST:
            for slow in self.config.MACD_SLOW:
                if fast >= slow:
                    continue
                for signal in self.config.MACD_SIGNAL:
                    for macd_type in self.config.MACD_TYPES:
                        strategy = {
                            'family': 'MACD',
                            'params': {
                                'fast': fast,
                                'slow': slow,
                                'signal': signal,
                                'type': macd_type
                            }
                        }
                        strategies.append(strategy)
        
        return strategies

# ============================================================================
# FILTRES
# ============================================================================

class VolumeFilter:
    """Filtre basé sur le volume"""
    
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Applique le filtre volume"""
        volume_sma = df['volume'].rolling(20).mean()
        filtered = signals.copy()
        filtered[df['volume'] < (volume_sma * self.multiplier)] = False
        return filtered

class TrendFilter:
    """Filtre basé sur la tendance"""
    
    def __init__(self, adx_threshold: float = 25):
        self.adx_threshold = adx_threshold
    
    def apply(self, df: pd.DataFrame, signals: pd.Series, adx: pd.Series) -> pd.Series:
        """Applique le filtre tendance"""
        filtered = signals.copy()
        filtered[adx < self.adx_threshold] = False
        return filtered

class VolatilityFilter:
    """Filtre basé sur la volatilité"""
    
    def __init__(self, atr_multiplier: float = 1.0):
        self.atr_multiplier = atr_multiplier
    
    def apply(self, df: pd.DataFrame, signals: pd.Series, atr: pd.Series) -> pd.Series:
        """Applique le filtre volatilité"""
        atr_median = atr.rolling(100).median()
        filtered = signals.copy()
        filtered[atr > (atr_median * self.atr_multiplier)] = False
        return filtered

# ============================================================================
# GESTION DU RISQUE
# ============================================================================

class RiskManager:
    """Gestion du risque adaptative"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_stop_loss(self, df: pd.DataFrame, entry_idx: int, 
                           entry_price: float, sl_type: str, 
                           sl_value: float, atr: pd.Series) -> float:
        """Calcule le stop-loss"""
        if sl_type == 'FIXED_PCT':
            return entry_price * (1 - sl_value / 100)
        elif sl_type == 'ATR_MULTIPLE':
            atr_val = atr.iloc[entry_idx] if entry_idx < len(atr) else 0
            return entry_price - (atr_val * sl_value)
        elif sl_type == 'SUPPORT':
            # Cherche le support le plus proche
            lookback = 50
            start = max(0, entry_idx - lookback)
            support = df['low'].iloc[start:entry_idx].min()
            return support * 0.99
        else:  # TRAILING
            return entry_price * 0.95
    
    def calculate_take_profit(self, entry_price: float, sl_price: float, 
                             tp_type: str, tp_value: float) -> float:
        """Calcule le take-profit"""
        risk = entry_price - sl_price
        
        if tp_type == 'FIXED_RR':
            return entry_price + (risk * tp_value)
        elif tp_type == 'MULTIPLE_TARGETS':
            return entry_price * (1 + tp_value / 100)
        elif tp_type == 'DYNAMIC':
            return entry_price * 1.03
        else:  # RESISTANCE
            return entry_price * 1.05
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                               sl_price: float, sizing_type: str, 
                               sizing_value: float) -> float:
        """Calcule la taille de position"""
        risk_per_share = entry_price - sl_price
        
        if sizing_type == 'FIXED_PCT':
            risk_amount = capital * (sizing_value / 100)
        elif sizing_type == 'KELLY':
            # Simplifié - à améliorer
            risk_amount = capital * 0.02
        else:  # VOLATILITY_ADJUSTED
            risk_amount = capital * 0.02
        
        if risk_per_share <= 0:
            return 0
        
        size = risk_amount / risk_per_share
        max_size = (capital * 0.15) / entry_price  # 15% max
        
        return min(size, max_size)

# ============================================================================
# MOTEUR DE BACKTEST VECTORISÉ
# ============================================================================

class VectorizedBacktestEngine:
    """Moteur de backtest vectorisé pour performance optimale"""
    
    def __init__(self, config: Config):
        self.config = config
        self.risk_manager = RiskManager(config)
    
    def run_backtest(self, df: pd.DataFrame, strategy: Dict, 
                    indicators: Dict, ticker: str, timeframe: str) -> Tuple[List[Trade], pd.Series]:
        """Exécute un backtest vectorisé"""
        
        # Générer les signaux
        signals = self._generate_signals(df, strategy, indicators)
        
        if signals.empty:
            return [], pd.Series()
        
        # Vectoriser les données
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        dates = df.index
        
        # Configuration risque (fixe pour cette version)
        sl_type = 'FIXED_PCT'
        sl_value = 2.0
        tp_type = 'FIXED_RR'
        tp_value = 3.0
        sizing_type = 'FIXED_PCT'
        sizing_value = 2.0
        
        # Initialisation
        capital = self.config.INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]
        position = None
        trade_id = 0
        
        atr = indicators.get('atr_14', pd.Series(0, index=df.index)).values
        adx = indicators.get('adx_14', pd.Series(0, index=df.index)).values
        
        for i in range(50, len(df)):  # Warmup period
            if capital <= 0:
                break
            
            # Gestion position ouverte
            if position is not None:
                sl_price = position['sl']
                tp_price = position['tp']
                entry_idx = position['entry_idx']
                
                # Vérifier sorties
                exit_idx = None
                exit_price = 0
                exit_reason = ""
                
                # SL hit
                if lows[i] <= sl_price:
                    exit_idx = i
                    exit_price = sl_price
                    exit_reason = "SL"
                # TP hit
                elif highs[i] >= tp_price:
                    exit_idx = i
                    exit_price = tp_price
                    exit_reason = "TP"
                # Time-based exit (50 bars max)
                elif (i - entry_idx) >= 50:
                    exit_idx = i
                    exit_price = opens[i]
                    exit_reason = "TIME"
                
                if exit_idx is not None:
                    # Calcul P&L
                    entry_price = position['entry_price']
                    position_size = position['size']
                    
                    gross_pnl = (exit_price - entry_price) * position_size
                    commission = (entry_price + exit_price) * position_size * self.config.COMMISSION_RATE
                    slippage = (entry_price + exit_price) * position_size * self.config.SLIPPAGE_RATE
                    net_pnl = gross_pnl - commission - slippage
                    
                    capital += net_pnl
                    
                    # Créer le trade
                    trade = self._create_trade(
                        trade_id=trade_id,
                        ticker=ticker,
                        timeframe=timeframe,
                        entry_idx=entry_idx,
                        exit_idx=exit_idx,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position_size,
                        pnl_net=net_pnl,
                        exit_reason=exit_reason,
                        df=df,
                        indicators=indicators,
                        strategy=strategy
                    )
                    
                    trades.append(trade)
                    trade_id += 1
                    position = None
            
            # Ouverture nouvelle position
            if position is None and i > 0 and signals.iloc[i-1]:
                entry_price = closes[i]
                
                # Calcul SL/TP
                sl_price = self.risk_manager.calculate_stop_loss(
                    df, i, entry_price, sl_type, sl_value, 
                    pd.Series(atr, index=df.index)
                )
                
                tp_price = self.risk_manager.calculate_take_profit(
                    entry_price, sl_price, tp_type, tp_value
                )
                
                # Calcul taille position
                position_size = self.risk_manager.calculate_position_size(
                    capital, entry_price, sl_price, sizing_type, sizing_value
                )
                
                if position_size > 0:
                    position = {
                        'entry_idx': i,
                        'entry_price': entry_price,
                        'sl': sl_price,
                        'tp': tp_price,
                        'size': position_size
                    }
            
            # Calcul equity courante
            current_equity = capital
            if position is not None:
                unrealized = (closes[i] - position['entry_price']) * position['size']
                current_equity = capital + unrealized
            
            equity_curve.append(current_equity)
        
        equity_series = pd.Series(equity_curve, index=[df.index[0]] + list(df.index[50:]))
        
        return trades, equity_series
    
    def _generate_signals(self, df: pd.DataFrame, strategy: Dict, indicators: Dict) -> pd.Series:
        """Génère les signaux selon la stratégie"""
        family = strategy['family']
        params = strategy['params']
        
        if family == 'RSI':
            return self._generate_rsi_signals(df, params, indicators)
        elif family == 'EMA':
            return self._generate_ema_signals(df, params, indicators)
        elif family == 'BB':
            return self._generate_bb_signals(df, params, indicators)
        elif family == 'MACD':
            return self._generate_macd_signals(df, params, indicators)
        else:
            return pd.Series(False, index=df.index)
    
    def _generate_rsi_signals(self, df: pd.DataFrame, params: Dict, indicators: Dict) -> pd.Series:
        """Signaux RSI"""
        period = params['period']
        buy_thresh = params['buy_threshold']
        signal_type = params['signal_type']
        
        rsi_key = f"rsi_{period}"
        rsi = indicators.get(rsi_key, TechnicalIndicators.rsi(df['close'], period))
        
        if signal_type == 'CROSS':
            return (rsi <= buy_thresh) & (rsi.shift(1) > buy_thresh)
        elif signal_type == 'CROSS_CONFIRM':
            return (rsi <= buy_thresh) & (rsi.shift(1) <= buy_thresh) & (rsi.shift(2) > buy_thresh)
        else:
            return pd.Series(False, index=df.index)
    
    def _generate_ema_signals(self, df: pd.DataFrame, params: Dict, indicators: Dict) -> pd.Series:
        """Signaux EMA"""
        ema_type = params['type']
        
        if ema_type == 'TOUCH':
            period = params['period']
            ema_key = f"ema_{period}"
            ema = indicators.get(ema_key, TechnicalIndicators.ema(df['close'], period))
            return (df['low'] <= ema) & (df['close'] > ema)
        
        elif ema_type == 'CROSSOVER':
            period = params['period']
            ema_key = f"ema_{period}"
            ema = indicators.get(ema_key, TechnicalIndicators.ema(df['close'], period))
            return (df['close'] > ema) & (df['close'].shift(1) <= ema.shift(1))
        
        elif ema_type == 'FAST_SLOW_CROSS':
            fast = params['fast_period']
            slow = params['slow_period']
            
            ema_fast_key = f"ema_{fast}"
            ema_slow_key = f"ema_{slow}"
            
            ema_fast = indicators.get(ema_fast_key, TechnicalIndicators.ema(df['close'], fast))
            ema_slow = indicators.get(ema_slow_key, TechnicalIndicators.ema(df['close'], slow))
            
            return (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        
        else:
            return pd.Series(False, index=df.index)
    
    def _generate_bb_signals(self, df: pd.DataFrame, params: Dict, indicators: Dict) -> pd.Series:
        """Signaux Bollinger Bands"""
        bb_type = params['type']
        period = params['period']
        std = params['std_dev']
        
        bb_key = f"bb_{period}_{std}"
        upper_key = f"{bb_key}_upper"
        lower_key = f"{bb_key}_lower"
        
        if upper_key not in indicators or lower_key not in indicators:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'], period, std)
        else:
            upper = indicators[upper_key]
            lower = indicators[lower_key]
        
        if bb_type == 'TOUCH_LOWER':
            return (df['low'] <= lower) & (df['close'] > lower)
        elif bb_type == 'BREAKOUT':
            return df['close'] < lower
        elif bb_type == 'REENTRY':
            was_outside = (df['close'].shift(1) > upper.shift(1)) | (df['close'].shift(1) < lower.shift(1))
            is_inside = (df['close'] <= upper) & (df['close'] >= lower)
            return was_outside & is_inside
        else:
            return pd.Series(False, index=df.index)
    
    def _generate_macd_signals(self, df: pd.DataFrame, params: Dict, indicators: Dict) -> pd.Series:
        """Signaux MACD"""
        fast = params['fast']
        slow = params['slow']
        signal = params['signal']
        macd_type = params['type']
        
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            df['close'], fast, slow, signal
        )
        
        if macd_type == 'CROSSOVER':
            return (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        elif macd_type == 'ZERO_CROSS':
            return (macd_line > 0) & (macd_line.shift(1) <= 0)
        else:
            return pd.Series(False, index=df.index)
    
    def _create_trade(self, trade_id: int, ticker: str, timeframe: str,
                     entry_idx: int, exit_idx: int, entry_price: float,
                     exit_price: float, position_size: float, pnl_net: float,
                     exit_reason: str, df: pd.DataFrame, indicators: Dict,
                     strategy: Dict) -> Trade:
        """Crée un objet Trade enrichi avec le contexte"""
        
        # Contexte technique à l'entrée
        entry_rsi_14 = indicators.get('rsi_14', pd.Series(50, index=df.index)).iloc[entry_idx]
        entry_ema_20 = indicators.get('ema_20', pd.Series(0, index=df.index)).iloc[entry_idx]
        entry_adx = indicators.get('adx_14', pd.Series(0, index=df.index)).iloc[entry_idx]
        entry_atr = indicators.get('atr_14', pd.Series(0, index=df.index)).iloc[entry_idx]
        entry_volume = df['volume'].iloc[entry_idx]
        entry_volume_ratio = indicators.get('volume_ratio', pd.Series(1, index=df.index)).iloc[entry_idx]
        
        # Régime marché
        adx_val = entry_adx
        atr_val = entry_atr
        atr_median = indicators.get('atr_14', pd.Series(0, index=df.index)).rolling(100).median().iloc[entry_idx]
        
        if adx_val > 25:
            market_regime = 'TRENDING'
            trend_strength = 'STRONG' if adx_val > 35 else 'WEAK'
        else:
            market_regime = 'RANGING'
            trend_strength = 'NEUTRAL'
        
        volatility_regime = 'HIGH' if atr_val > atr_median * 1.5 else 'NORMAL'
        volume_regime = 'HIGH' if entry_volume_ratio > 1.5 else 'NORMAL'
        
        # Structure prix
        support, resistance = MarketContext.calculate_support_resistance(df)
        distance_to_support = ((df['close'].iloc[entry_idx] - support.iloc[entry_idx]) / 
                              support.iloc[entry_idx] * 100) if support.iloc[entry_idx] > 0 else 0
        distance_to_resistance = ((resistance.iloc[entry_idx] - df['close'].iloc[entry_idx]) / 
                                 df['close'].iloc[entry_idx] * 100) if df['close'].iloc[entry_idx] > 0 else 0
        
        # Momentum
        rsi_slope = (entry_rsi_14 - indicators.get('rsi_14', pd.Series(50, index=df.index)).iloc[entry_idx-5]) / 5
        macd_hist = 0  # À calculer si nécessaire
        
        # Contexte temporel
        entry_time = df.index[entry_idx]
        session = MarketContext.calculate_session(entry_time)
        hour_of_day = entry_time.hour
        day_of_week = entry_time.weekday()
        month = entry_time.month
        quarter = (month - 1) // 3 + 1
        is_weekend = day_of_week >= 5
        
        # Calcul durée
        exit_time = df.index[exit_idx]
        duration_bars = exit_idx - entry_idx
        duration_hours = duration_bars * {'M15': 0.25, 'H1': 1, 'H4': 4, 'D1': 24, 'W1': 168}.get(timeframe, 1)
        
        # Création du trade
        return Trade(
            trade_id=f"{ticker}_{timeframe}_{trade_id}",
            strategy_id=f"{strategy['family']}_{hashlib.md5(json.dumps(strategy['params']).encode()).hexdigest()[:8]}",
            ticker=ticker,
            timeframe=timeframe,
            entry_timestamp=str(entry_time),
            exit_timestamp=str(exit_time),
            duration_bars=duration_bars,
            duration_hours=duration_hours,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss_price=entry_price * 0.98,  # Simplifié
            take_profit_price=entry_price * 1.03,  # Simplifié
            position_size=position_size,
            position_value_usd=entry_price * position_size,
            pnl_absolute=pnl_net,
            pnl_percentage=(pnl_net / (entry_price * position_size)) * 100 if (entry_price * position_size) > 0 else 0,
            pnl_commission=(entry_price + exit_price) * position_size * self.config.COMMISSION_RATE,
            pnl_slippage=(entry_price + exit_price) * position_size * self.config.SLIPPAGE_RATE,
            pnl_net=pnl_net,
            exit_reason=exit_reason,
            exit_type=exit_reason,
            entry_rsi_14=float(entry_rsi_14),
            entry_ema_20=float(entry_ema_20),
            entry_bb_width_pct=0.0,  # À calculer
            entry_adx=float(entry_adx),
            entry_atr=float(entry_atr),
            entry_atr_pct=float((entry_atr / entry_price) * 100) if entry_price > 0 else 0,
            entry_volume=float(entry_volume),
            entry_volume_ratio=float(entry_volume_ratio),
            market_regime=market_regime,
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            volume_regime=volume_regime,
            distance_to_support_pct=float(distance_to_support),
            distance_to_resistance_pct=float(distance_to_resistance),
            price_position_in_range=float((df['close'].iloc[entry_idx] - support.iloc[entry_idx]) / 
                                         (resistance.iloc[entry_idx] - support.iloc[entry_idx]) 
                                         if resistance.iloc[entry_idx] > support.iloc[entry_idx] else 0.5),
            is_swing_low=False,  # À calculer
            is_swing_high=False,  # À calculer
            rsi_slope_5=float(rsi_slope),
            macd_histogram=float(macd_hist),
            momentum_roc_10=0.0,  # À calculer
            candle_body_ratio=0.0,  # À calculer
            session=session,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            month=month,
            quarter=quarter,
            is_weekend=is_weekend,
            is_market_open=not is_weekend
        )

# ============================================================================
# CALCUL DES MÉTRIQUES BRUTES
# ============================================================================

class MetricsCalculator:
    """Calcule les métriques brutes - PAS DE SCORE COMPOSITE"""
    
    @staticmethod
    def calculate_strategy_metrics(trades: List[Trade], equity_curve: pd.Series,
                                  strategy: Dict, ticker: str, timeframe: str,
                                  start_date: str, end_date: str) -> StrategyMetrics:
        """Calcule toutes les métriques brutes d'une stratégie"""
        
        if not trades:
            return None
        
        # Métriques de base
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl_net > 0]
        losing_trades = [t for t in trades if t.pnl_net < 0]
        break_even_trades = [t for t in trades if t.pnl_net == 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        total_wins = sum(t.pnl_net for t in winning_trades)
        total_losses = abs(sum(t.pnl_net for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        expectancy = (win_rate / 100 * np.mean([t.pnl_percentage for t in winning_trades]) + 
                     ((100 - win_rate) / 100 * np.mean([t.pnl_percentage for t in losing_trades]))) if winning_trades and losing_trades else 0
        
        total_return_usd = equity_curve.iloc[-1] - equity_curve.iloc[0] if len(equity_curve) > 0 else 0
        total_return_pct = (total_return_usd / equity_curve.iloc[0] * 100) if equity_curve.iloc[0] > 0 else 0
        
        # Risk metrics
        max_dd_pct, max_dd_usd, dd_duration = MetricsCalculator._calculate_drawdown(equity_curve)
        sharpe, sortino = MetricsCalculator._calculate_risk_adjusted_returns(equity_curve, timeframe)
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
        
        # Trade statistics
        avg_win_pct = np.mean([t.pnl_percentage for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t.pnl_percentage for t in losing_trades]) if losing_trades else 0
        avg_trade_duration = np.mean([t.duration_hours for t in trades]) if trades else 0
        
        # Distribution
        pnl_values = [t.pnl_percentage for t in trades]
        pnl_skewness = pd.Series(pnl_values).skew() if len(pnl_values) > 2 else 0
        pnl_kurtosis = pd.Series(pnl_values).kurtosis() if len(pnl_values) > 3 else 0
        
        # Walk-forward (simplifié pour cette version)
        wf_results = json.dumps({'fold_1': {'win_rate': win_rate, 'profit_factor': profit_factor}})
        
        # Performance par régime (simplifié)
        performance_by_regime = json.dumps({
            'trending': {'win_rate': win_rate, 'profit_factor': profit_factor},
            'ranging': {'win_rate': win_rate, 'profit_factor': profit_factor}
        })
        
        return StrategyMetrics(
            strategy_id=f"{strategy['family']}_{hashlib.md5(json.dumps(strategy['params']).encode()).hexdigest()[:8]}",
            strategy_family=strategy['family'],
            strategy_params=strategy['params'],
            risk_params={},  # À remplir
            filter_params={},  # À remplir
            ticker=ticker,
            timeframe=timeframe,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            break_even_trades=len(break_even_trades),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            expectancy=float(expectancy),
            total_return_pct=float(total_return_pct),
            total_return_usd=float(total_return_usd),
            max_drawdown_pct=float(max_dd_pct),
            max_drawdown_usd=float(max_dd_usd),
            max_drawdown_duration_days=int(dd_duration),
            avg_drawdown_pct=float(max_dd_pct / 2),  # Simplifié
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            ulcer_index=float(max_dd_pct / 10),  # Simplifié
            var_95=float(-max_dd_pct * 0.8),  # Simplifié
            avg_win_pct=float(avg_win_pct),
            avg_loss_pct=float(avg_loss_pct),
            avg_trade_duration_hours=float(avg_trade_duration),
            max_consecutive_wins=0,  # À calculer
            max_consecutive_losses=0,  # À calculer
            avg_bars_held=float(np.mean([t.duration_bars for t in trades]) if trades else 0),
            time_in_market_pct=float(avg_trade_duration * total_trades / (24 * 365) * 100) if total_trades > 0 else 0,
            pnl_skewness=float(pnl_skewness),
            pnl_kurtosis=float(pnl_kurtosis),
            win_pnl_std=float(np.std([t.pnl_percentage for t in winning_trades]) if winning_trades else 0),
            loss_pnl_std=float(np.std([t.pnl_percentage for t in losing_trades]) if losing_trades else 0),
            wf_results=wf_results,
            wf_win_rate_stability=1.0,
            wf_profit_factor_stability=1.0,
            wf_degradation_score=1.0,
            performance_by_regime=performance_by_regime,
            volatility_sensitivity=0.5,
            trend_sensitivity=0.5,
            volume_sensitivity=0.5,
            time_sensitivity=0.5,
            liquidity_sensitivity=0.5,
            data_points_used=len(equity_curve),
            data_quality=1.0,
            calculation_time_seconds=0.0,
            test_period_start=start_date,
            test_period_end=end_date
        )
    
    @staticmethod
    def _calculate_drawdown(equity_curve: pd.Series) -> Tuple[float, float, int]:
        """Calcule le drawdown maximum"""
        if len(equity_curve) == 0:
            return 0.0, 0.0, 0
        
        peak = equity_curve.expanding().max()
        drawdown = (peak - equity_curve) / peak * 100
        
        max_dd_pct = drawdown.max()
        max_dd_idx = drawdown.argmax()
        max_dd_usd = (peak.iloc[max_dd_idx] - equity_curve.iloc[max_dd_idx]) if max_dd_idx < len(equity_curve) else 0
        
        # Durée du drawdown
        dd_duration = 0
        in_dd = False
        current_dd = 0
        
        for dd in drawdown:
            if dd > 0:
                if not in_dd:
                    in_dd = True
                current_dd += 1
            else:
                if in_dd:
                    dd_duration = max(dd_duration, current_dd)
                    in_dd = False
                    current_dd = 0
        
        return float(max_dd_pct), float(max_dd_usd), dd_duration
    
    @staticmethod
    def _calculate_risk_adjusted_returns(equity_curve: pd.Series, timeframe: str) -> Tuple[float, float]:
        """Calcule Sharpe et Sortino ratios"""
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0, 0.0
        
        # Annualization factor
        bars_per_year = {'M15': 252*24*4, 'H1': 252*24, 'H4': 252*6, 'D1': 252, 'W1': 52}
        annual_factor = np.sqrt(bars_per_year.get(timeframe, 252))
        
        sharpe = (returns.mean() / returns.std()) * annual_factor
        
        # Sortino (seulement downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * annual_factor
        else:
            sortino = 0.0
        
        return float(sharpe), float(sortino)

# ============================================================================
# WALK-FORWARD VALIDATOR
# ============================================================================

class WalkForwardValidator:
    """Validation walk-forward en 5 folds"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def validate(self, df: pd.DataFrame, strategy: Dict, indicators: Dict,
                ticker: str, timeframe: str) -> List[StrategyMetrics]:
        """Exécute la validation walk-forward en 5 folds"""
        
        folds = self._create_folds(df)
        all_metrics = []
        
        for fold_num, (train_df, test_df) in enumerate(folds):
            # Backtest sur train
            backtest_engine = VectorizedBacktestEngine(self.config)
            train_trades, train_equity = backtest_engine.run_backtest(
                train_df, strategy, indicators, ticker, timeframe
            )
            
            if not train_trades:
                continue
            
            # Calcul métriques train
            train_metrics = MetricsCalculator.calculate_strategy_metrics(
                train_trades, train_equity, strategy, ticker, timeframe,
                str(train_df.index[0]), str(train_df.index[-1])
            )
            
            # Backtest sur test
            test_trades, test_equity = backtest_engine.run_backtest(
                test_df, strategy, indicators, ticker, timeframe
            )
            
            if not test_trades:
                continue
            
            # Calcul métriques test
            test_metrics = MetricsCalculator.calculate_strategy_metrics(
                test_trades, test_equity, strategy, ticker, timeframe,
                str(test_df.index[0]), str(test_df.index[-1])
            )
            
            # Enrichir avec les résultats walk-forward
            if train_metrics and test_metrics:
                train_metrics.wf_results = json.dumps({
                    f'fold_{fold_num}': {
                        'train': {'win_rate': train_metrics.win_rate, 
                                 'profit_factor': train_metrics.profit_factor},
                        'test': {'win_rate': test_metrics.win_rate, 
                                'profit_factor': test_metrics.profit_factor}
                    }
                })
                
                train_metrics.wf_degradation_score = min(
                    test_metrics.win_rate / train_metrics.win_rate if train_metrics.win_rate > 0 else 1,
                    test_metrics.profit_factor / train_metrics.profit_factor if train_metrics.profit_factor > 0 else 1
                )
                
                all_metrics.append(train_metrics)
        
        return all_metrics
    
    def _create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Crée 5 folds temporels"""
        folds = []
        total_len = len(df)
        fold_size = total_len // self.config.WF_FOLDS
        
        for i in range(self.config.WF_FOLDS - 1):
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]
            
            folds.append((train_df, test_df))
        
        # Dernier fold
        train_df = df.iloc[: (self.config.WF_FOLDS - 1) * fold_size]
        test_df = df.iloc[(self.config.WF_FOLDS - 1) * fold_size:]
        
        folds.append((train_df, test_df))
        
        return folds

# ============================================================================
# GESTIONNAIRE DE MULTIPROCESSING
# ============================================================================

class MultiprocessManager:
    """Gère le multiprocessing pour les backtests"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
    
    def run_backtests_parallel(self, all_data: Dict, all_strategies: List[Dict]) -> Tuple[List[Trade], List[StrategyMetrics]]:
        """Exécute tous les backtests en parallèle"""
        
        print(f"⚡ Lancement de {len(all_strategies)} stratégies sur {self.config.CPU_CORES} cores...")
        
        # Préparer les tâches
        tasks = []
        for (ticker, timeframe), df in all_data.items():
            for strategy in all_strategies:
                tasks.append((ticker, timeframe, df, strategy))
        
        # Exécution parallèle
        all_trades = []
        all_metrics = []
        
        with ProcessPoolExecutor(max_workers=self.config.CPU_CORES) as executor:
            futures = []
            
            for task in tasks:
                future = executor.submit(self._run_single_backtest_task, task)
                futures.append(future)
            
            # Collecter les résultats
            for future in tqdm(as_completed(futures), total=len(futures), desc="Backtesting"):
                try:
                    trades, metrics = future.result(timeout=300)
                    if trades:
                        all_trades.extend(trades)
                    if metrics:
                        all_metrics.extend(metrics)
                except Exception as e:
                    print(f"❌ Erreur dans le backtest: {e}")
                    continue
        
        print(f"✅ {len(all_trades)} trades générés, {len(all_metrics)} stratégies évaluées")
        
        return all_trades, all_metrics
    
    def _run_single_backtest_task(self, task: Tuple) -> Tuple[List[Trade], List[StrategyMetrics]]:
        """Tâche individuelle de backtest"""
        ticker, timeframe, df, strategy = task
        
        # Créer le cache d'indicateurs
        cache = IndicatorsCache(self.config)
        indicators = cache.precompute_for_data(df, ticker, timeframe)
        
        # Valider avec walk-forward
        validator = WalkForwardValidator(self.config)
        metrics = validator.validate(df, strategy, indicators, ticker, timeframe)
        
        # Récupérer les trades du premier fold (pour l'export)
        backtest_engine = VectorizedBacktestEngine(self.config)
        trades, _ = backtest_engine.run_backtest(df, strategy, indicators, ticker, timeframe)
        
        return trades, metrics

# ============================================================================
# EXPORT DE DONNÉES
# ============================================================================

class DataExporter:
    """Exporte les données au format Parquet optimisé"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def export_all(self, trades: List[Trade], metrics: List[StrategyMetrics]):
        """Exporte toutes les données"""
        
        print("\n💾 Export des données...")
        
        # Export des trades
        if trades:
            trades_df = pd.DataFrame([asdict(t) for t in trades])
            self._export_parquet(trades_df, "trades_detailed")
            print(f"  ✅ Trades: {len(trades_df)} enregistrements")
        
        # Export des métriques de stratégies
        if metrics:
            metrics_dicts = []
            for m in metrics:
                m_dict = asdict(m)
                # Sérialiser les dictionnaires en JSON
                m_dict['strategy_params'] = json.dumps(m_dict['strategy_params'])
                m_dict['risk_params'] = json.dumps(m_dict['risk_params'])
                m_dict['filter_params'] = json.dumps(m_dict['filter_params'])
                metrics_dicts.append(m_dict)
            
            metrics_df = pd.DataFrame(metrics_dicts)
            self._export_parquet(metrics_df, "strategies_summary")
            print(f"  ✅ Stratégies: {len(metrics_df)} enregistrements")
        
        # Sauvegarder la configuration
        self._save_config()
        
        print(f"\n🎉 Dataset sauvegardé dans: {self.config.OUTPUT_DIR}")
    
    def _export_parquet(self, df: pd.DataFrame, filename: str):
        """Export un DataFrame en Parquet optimisé"""
        
        # Optimiser les types
        df_optimized = self._optimize_dtypes(df)
        
        # Chemin de sortie
        output_path = self.config.OUTPUT_DIR / f"{filename}.parquet"
        
        # Exporter
        df_optimized.to_parquet(
            output_path,
            engine='pyarrow',
            compression=self.config.COMPRESSION,
            index=False
        )
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimise les types de données pour réduire la taille"""
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'float64':
                # Convertir en float32 si possible
                df[col] = df[col].astype('float32')
            
            elif col_type == 'int64':
                # Trouver le bon type entier
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype('uint8')
                    elif col_max < 65535:
                        df[col] = df[col].astype('uint16')
                    elif col_max < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if col_min > -128 and col_max < 127:
                        df[col] = df[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df[col] = df[col].astype('int32')
        
        return df
    
    def _save_config(self):
        """Sauvegarde la configuration utilisée"""
        config_dict = {
            k: v for k, v in vars(self.config).items() 
            if not k.startswith('_') and not callable(v)
        }
        
        config_path = self.config.OUTPUT_DIR / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Point d'entrée principal"""
    
    print("\n" + "="*80)
    print("🚀 BACKTEST ENGINE v15 - GÉNÉRATEUR DE DATASET POUR IA")
    print("="*80 + "\n")
    
    # Initialisation
    start_time = time.time()
    config = Config()
    
    print(f"📊 Configuration:")
    print(f"   Actifs: {len(config.CRYPTO_TICKERS)} crypto + {len(config.STOCK_TICKERS)} actions")
    print(f"   Timeframes: {list(config.TIMEFRAMES.keys())}")
    print(f"   Cores CPU: {config.CPU_CORES}")
    print(f"   Sortie: {config.OUTPUT_DIR}\n")
    
    # 1. Téléchargement des données
    print("📥 Étape 1: Téléchargement des données...")
    data_fetcher = DataFetcher(config)
    all_data = data_fetcher.fetch_all_data()
    
    if not all_data:
        print("❌ Aucune donnée téléchargée")
        return
    
    print(f"✅ {len(all_data)} datasets téléchargés\n")
    
    # 2. Génération des stratégies
    print("🧠 Étape 2: Génération des stratégies brute force...")
    strategy_gen = StrategyGenerator(config)
    all_strategies = strategy_gen.generate_all_strategies()
    
    # Limiter pour les tests (commenter pour la version complète)
    all_strategies = all_strategies[:100]  # TEST: 100 stratégies seulement
    print(f"🔬 TEST MODE: {len(all_strategies)} stratégies seulement\n")
    
    # 3. Backtests parallèles
    print("⚡ Étape 3: Backtests parallèles...")
    mp_manager = MultiprocessManager(config)
    all_trades, all_metrics = mp_manager.run_backtests_parallel(all_data, all_strategies)
    
    # 4. Export des données
    print("\n💾 Étape 4: Export des données...")
    exporter = DataExporter(config)
    exporter.export_all(all_trades, all_metrics)
    
    # Résumé final
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("✅ TERMINÉ AVEC SUCCÈS")
    print("="*80)
    print(f"   Durée totale: {duration:.1f} secondes ({duration/60:.1f} minutes)")
    print(f"   Trades générés: {len(all_trades)}")
    print(f"   Stratégies évaluées: {len(all_metrics)}")
    print(f"   Dossier de sortie: {config.OUTPUT_DIR}")
    print("="*80)
    
    # Afficher un échantillon de stratégies
    if all_metrics:
        print("\n📈 Top 5 stratégies par profit factor:")
        top_metrics = sorted(all_metrics, key=lambda x: x.profit_factor, reverse=True)[:5]
        
        for i, metric in enumerate(top_metrics, 1):
            print(f"   {i}. {metric.strategy_family} - PF: {metric.profit_factor:.2f}, "
                  f"WR: {metric.win_rate:.1f}%, Trades: {metric.total_trades}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()