📊 Backtest Engine v13.0

Système de backtesting automatisé pour stratégies de trading

🚀 Fonctionnalités

· ✅ Backtesting multi-timeframes (15min à 1 semaine)
· ✅ Multi-actifs (Cryptos + Actions)
· ✅ 7 stratégies intégrées (RSI, EMA, Bollinger Bands)
· ✅ Gestion des risques avancée (SL, TP, ATR)
· ✅ Calcul de métriques complètes (Sharpe, Sortino, Calmar)
· ✅ Export des résultats en CSV/JSON
· ✅ Téléchargement automatique des données (Yahoo Finance)

📦 Installation

1. Prérequis

· Python 3.8 ou supérieur
· pip (gestionnaire de packages Python)

2. Installation des dépendances

```bash
pip install -r requirements.txt
```

Si vous n'avez pas de fichier requirements.txt, installez manuellement :

```bash
pip install pandas numpy yfinance pyarrow
```

🎯 Utilisation

1. Exécution simple

```bash
python backtestv13.py
```

2. Structure du projet

```
bot-trading-ia/
├── backtestv13.py          # Script principal
├── requirements.txt        # Dépendances
├── backtest_results_*/     # Dossier de résultats (auto-créé)
│   ├── all_metrics_for_ai.csv
│   ├── top_500_strategies.csv
│   ├── all_trades.csv
│   └── summary.json
```

⚙️ Configuration

Le fichier contient une classe Config modifiable :

Actifs testés

```python
CRYPTO_TICKERS = {'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD', 'XLM/USD': 'XLM-USD'}
STOCK_TICKERS = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'ORCL', 'V']
```

Timeframes disponibles

· M15 (15 minutes)
· M30 (30 minutes)
· H1 (1 heure)
· H4 (4 heures)
· D1 (1 jour)
· W1 (1 semaine)

Stratégies intégrées

1. RSI_CROSS - Achat quand RSI traverse un seuil
2. EMA_TOUCH - Achat au toucher d'une EMA
3. EMA_CROSSOVER - Achat au croisement de prix avec EMA
4. EMA_REVERSION - Achat sur déviation de l'EMA
5. BB_TOUCH - Achat au toucher de la bande inférieure de Bollinger
6. BB_BREAKOUT - Achat/vente sur breakout des bandes
7. BB_REENTRY - Achat après sortie des bandes

📈 Métriques calculées

Pour chaque stratégie, le système calcule :

· Profit Factor et Win Rate
· Drawdown max et moyen
· Ratios de performance (Sharpe, Sortino, Calmar)
· Statistiques de trades (durée, fréquence)
· Score composite de performance

🎮 Mode TEST vs FULL

Mode TEST (défaut)

```python
MODE = 'TEST'
```

· Combinaisons réduites pour tests rapides
· 39 tests par timeframe
· Parfait pour le développement

Mode FULL

```python
MODE = 'FULL'
```

· Toutes les combinaisons de paramètres
· 1000+ tests par timeframe
· Pour l'optimisation complète

📊 Résultats

Fichiers générés

1. all_metrics_for_ai.csv - Toutes les métriques détaillées
2. top_500_strategies.csv - Top 500 stratégies triées par score
3. all_trades.csv - Historique complet de tous les trades
4. summary.json - Résumé par timeframe

Structure d'un trade

```python
{
  "trade_id": 1,
  "ticker": "BTC/USD",
  "timeframe": "H1",
  "entry_price": 50000.0,
  "exit_price": 51000.0,
  "pnl_pct": 2.0,
  "exit_reason": "TP",  # SL, TP, TIME, SIGNAL
  "strategy_type": "RSI_CROSS",
  "bars_held": 5
}
```

⚠️ Notes importantes

1. Données historiques

· Les données sont téléchargées depuis Yahoo Finance
· Limites : ~60 jours pour les intraday, ~5 ans pour le daily
· Volume minimal requis : 500 bougies (configurable)

2. Paramètres de risque

```python
INITIAL_CAPITAL = 10000.0      # Capital initial
RISK_PER_TRADE = 0.02          # 2% de risque par trade
MAX_POSITION_PCT = 0.15        # 15% max du capital
```

3. Commissions et slippage

Des frais réalistes sont appliqués selon le timeframe :

· M15 : 0.05% commission, 0.03% slippage
· D1 : 0.01% commission, 0.002% slippage

🔧 Dépannage

Erreur "No module named yfinance"

```bash
pip install yfinance
```

Erreur de données insuffisantes

· Vérifiez la connexion internet
· Augmentez MIN_BARS dans la configuration
· Testez avec un timeframe plus long

Performance lente

· Réduisez le nombre d'actifs/timeframes
· Passez en mode TEST
· Fermez d'autres applications

📝 Personnalisation

Ajouter un nouvel actif

```python
STOCK_TICKERS.append('GOOGL')  # Ajouter Google
```

Ajouter une stratégie

1. Ajoutez la logique dans generate_signals()
2. Ajoutez les paramètres dans generate_combinations()
3. Testez avec un timeframe spécifique

📄 Licence

Projet éducatif - Utilisation à vos propres risques

👨‍💻 Auteur

Développé pour la formation en trading algorithmique

---

⚠️ AVERTISSEMENT : Ce système est pour l'éducation et la recherche. Le trading comporte des risques de perte. Testez toujours vos stratégies avec un capital que vous pouvez vous permettre de perdre.
