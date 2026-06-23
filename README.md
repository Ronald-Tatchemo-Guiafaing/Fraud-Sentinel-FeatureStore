# Fraud Sentinel — Feature Store (3 datasets CNP)

**Ronald Tatchemo Guiafaing** —  Feature Stores Engineering

Repository : code + figures EDA. Les datasets volumineux ne sont pas versionnés (voir ci-dessous).

## Trois datasets RÉELS (pas de données inventées)

| # | Institution | Lignes | Fraude % | Source |
|---|-------------|--------|----------|--------|
| 1 | ULB | 284 807 | 0,17 % | `creditcard.csv` (Kaggle ULB) |
| 2 | IEEE-CIS / Vesta | 590 540 | 3,50 % | HuggingFace (corpus Kaggle) |
| 3 | Sparkov | 1 852 394 | 0,52 % | Kaggle `kartik2112/fraud-detection` |

Chaque dataset : **Feature Engineering** + **Feature Store** (Parquet offline, SQLite online, metadata JSON).

## Installation

```bat
pip install -r requirements.txt
```

Placer `creditcard.csv` à la racine du projet (ou laisser le pipeline le télécharger si configuré). IEEE et Sparkov sont mis en cache au premier run.

## Lancer

```bat
python run_feature_store_pipeline.py
python generate_store_figures.py
run_dashboard.bat
python show_metadata.py
```

## Fichiers livrables (local)

- `Rapport_Fraud_Sentinel.pdf`
- `figures/` — graphiques par dataset (`d1_ulb_*`, `d2_ieee_*`, `d3_sparkov_*`)

## Code principal

- `fraud_sentinel/data_loaders.py` — charge les 3 vrais corpus
- `fraud_sentinel/pipeline.py` — pipeline complet FE + FSE
- `fraud_sentinel/ml_utils.py` — RF + MLP par dataset
- `generate_store_figures.py` — figures EDA pour les 3
