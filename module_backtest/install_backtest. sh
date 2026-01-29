#!/bin/bash
# install_backtest.sh - Installation automatisée du backtest engine

echo "🚀 Installation du Backtest Engine v15..."

# Mise à jour pip
python -m pip install --upgrade pip

# Installation des dépendances de base
echo "📦 Installation des dépendances de base..."
pip install pandas numpy scipy statsmodels

# Installation des sources de données
echo "🌐 Installation des sources de données..."
pip install yfinance ccxt

# Installation de l'optimisation
echo "⚡ Installation des optimisations..."
pip install pyarrow tqdm numba joblib

# Installation optionnelle (commenter si problème)
echo "🔧 Installation des options avancées..."
pip install dask zstandard orjson

# Vérification de l'installation
echo "✅ Vérification de l'installation..."
python -c "import pandas, numpy, yfinance, ccxt, pyarrow, tqdm; print('Installation réussie!')"

echo "🎉 Installation terminée! Lancez: python backtest_v15.py"
