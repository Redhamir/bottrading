#!/usr/bin/env python3
"""
Vérifie que toutes les dépendances sont installées correctement
"""

import sys
import importlib

REQUIRED_PACKAGES = [
    ('pandas', '2.0.0'),
    ('numpy', '1.24.0'),
    ('yfinance', '0.2.33'),
    ('ccxt', '4.2.0'),
    ('pyarrow', '14.0.0'),
    ('tqdm', '4.65.0'),
    ('multiprocess', '0.70.15'),
]

OPTIONAL_PACKAGES = [
    ('numba', '0.57.0'),
    ('joblib', '1.3.0'),
    ('dask', '2023.8.0'),
    ('zstandard', '0.22.0'),
]

def check_package(package_name, min_version):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', '0.0.0')
        
        # Comparaison de version simple
        version_parts = list(map(int, version.split('.')[:3]))
        min_parts = list(map(int, min_version.split('.')[:3]))
        
        if version_parts >= min_parts:
            return True, version
        else:
            return False, f"{version} < {min_version}"
            
    except ImportError:
        return False, "Non installé"

print("🔍 Vérification de l'environnement Backtest v15")
print("="*50)

all_ok = True

# Packages requis
print("\n📦 Packages requis:")
for package, min_version in REQUIRED_PACKAGES:
    ok, status = check_package(package, min_version)
    status_symbol = "✅" if ok else "❌"
    print(f"  {status_symbol} {package:20} {status}")
    if not ok:
        all_ok = False

# Packages optionnels
print("\n🔧 Packages optionnels (recommandés):")
for package, min_version in OPTIONAL_PACKAGES:
    ok, status = check_package(package, min_version)
    status_symbol = "✓" if ok else "○"
    print(f"  {status_symbol} {package:20} {status}")

# Résumé
print("\n" + "="*50)
if all_ok:
    print("🎉 Environnement prêt pour le backtest!")
    print("   Lancez: python backtest_v15.py")
else:
    print("⚠️  Certains packages manquent ou sont obsolètes")
    print("   Exécutez: pip install -r requirements.txt")
    sys.exit(1)
