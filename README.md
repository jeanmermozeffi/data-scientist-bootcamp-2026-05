# Data Scientist Bootcamp — GOMYCODE Mai 2026

Matériel de cours et exercices pour la promotion data science de mai 2026.

## Structure du bootcamp

```
01-sql/
├── notions/              # Cours théoriques SQL (DDL, DML, DQL, DCL, TCL)
├── exercices/            # Exercices pratiques
├── solutions/            # Corrections
└── pratiques/projet-banque-sql/  # Projet complet FinBank

02-python/
├── 01-chapitre-introduction/  # Chapitres Python (Introduction → OOP)
├── notebooks/                 # Notebooks interactifs
└── TP/                        # 5 projets + TP final

03-data-science/
├── cours/                # NumPy, Pandas, Exploration, Visualisation, Web Scraping
├── notebooks/            # Notebooks interactifs
└── TP/                   # Data Cleaning (Employés) + web-scraping (page HTML/CSS d'exercice)

04-machine-learning/
├── cours/                # 01 Algorithmes ML · 02 Checkpoint 5 Iris · 03 Train/Test & Validation croisée
│                         # 04 Métriques d'évaluation · 05 Nettoyage & Feature Scaling · 06 Checkpoint 8 Preprocessing
├── notebooks/            # Notebooks interactifs
└── TP/                   # TP Loyer, Crédit, Churn, Facture (énoncés + guides + notebooks)

05-deep-learning/
├── cours/                # 01 Fondamentaux IA/ML/DL · 02 Checkpoint 9 Réseau de neurones
│   │                     # 03 Forward & Backward Propagation · 04 Concevoir et entraîner des CNN
│   └── images/           # Illustrations des cours 03 et 04
└── notebooks/            # Notebooks interactifs (Colab : activer le GPU pour le notebook 04)

data/                     # Jeux de données partagés (CSV)
```

## Démarrage

```bash
git clone https://github.com/jeanmermozeffi/data-scientist-bootcamp-2026-05.git
cd data-scientist-bootcamp-2026-05
```

Ouvrir les notebooks avec **DataSpell** ou **JupyterLab**.

> 🧠 **Deep Learning** : les notebooks du module `05-deep-learning` (notamment le 04, sur les CNN) utilisent
> **TensorFlow / Keras** : `pip install tensorflow`. Ils fonctionnent aussi directement sur **Google Colab**.
