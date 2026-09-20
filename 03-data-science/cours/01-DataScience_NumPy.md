# 🔢 Numerical and Data Analysis with NumPy — Cours Bootcamp Data Science

> **Module 2 — Data Science | Chapitre 1** | Prérequis : Module Python Pur (Chapitres 1 à 5)

---

## Table des matières

1. [What's NumPy ? — Introduction](#1-whats-numpy--introduction)
2. [Pourquoi NumPy et pas des listes Python ?](#2-pourquoi-numpy-et-pas-des-listes-python-)
3. [NumPy Arrays Creation — Créer des tableaux](#3-numpy-arrays-creation--créer-des-tableaux)
4. [Anatomie d'un tableau NumPy](#4-anatomie-dun-tableau-numpy)
5. [Types de données NumPy (dtype)](#5-types-de-données-numpy-dtype)
6. [Array Reshaping — Remodeler un tableau](#6-array-reshaping--remodeler-un-tableau)
7. [Array Indexing — Accéder aux éléments](#7-array-indexing--accéder-aux-éléments)
8. [Array Slicing — Découper un tableau](#8-array-slicing--découper-un-tableau)
9. [Indexation booléenne et Fancy Indexing](#9-indexation-booléenne-et-fancy-indexing)
10. [NumPy Random — Génération aléatoire](#10-numpy-random--génération-aléatoire)
11. [NumPy Addition/Soustraction et le Broadcasting](#11-numpy-additionsoustraction-et-le-broadcasting)
12. [Fonctions d'agrégation](#12-fonctions-dagrégation)
13. [Some NumPy Functions — Fonctions utiles](#13-some-numpy-functions--fonctions-utiles)
14. [Algèbre linéaire avec NumPy](#14-algèbre-linéaire-avec-numpy)
15. [Conclusion](#15-conclusion)
16. [✅ Point de contrôle — NumPy](#16--point-de-contrôle--numpy)

---

## 1. What's NumPy ? — Introduction

### 📖 Définition

**NumPy** (Numerical Python) est **la bibliothèque fondamentale** du calcul scientifique et numérique en Python. Elle introduit un nouvel objet central, le **`ndarray`** (N-dimensional array), et propose des milliers de fonctions optimisées pour manipuler des données numériques à grande vitesse.

> 💡 **Analogie** : Si Python pur est une **calculatrice de poche**, NumPy est une **calculatrice scientifique industrielle**. Les deux font des calculs, mais NumPy est conçu pour traiter des **millions de nombres simultanément**, à une vitesse que les listes Python classiques ne peuvent pas égaler.

### 1.1 Pourquoi NumPy est-il si important en Data Science ?

```
NUMPY EST LE SOCLE DE TOUT L'ÉCOSYSTÈME DATA SCIENCE PYTHON
│
├── 🐼 Pandas          → construit directement SUR NumPy
├── 📊 Matplotlib       → utilise des tableaux NumPy pour tracer des graphiques
├── 🤖 Scikit-learn     → attend des tableaux NumPy en entrée de ses modèles
├── 🧠 TensorFlow/PyTorch → leurs "tenseurs" s'inspirent directement des ndarray
└── 📐 SciPy            → calcul scientifique avancé, construit sur NumPy
```

> 🔑 **Comprendre NumPy en profondeur, c'est comprendre les fondations sur lesquelles reposent TOUS les autres outils du bootcamp.**

### 1.2 Installer et importer NumPy

```python
# Installation (une seule fois, dans un terminal ou une cellule Colab)
!pip install numpy

# Importation (convention universelle : alias "np")
import numpy as np

print(np.__version__)   # Vérifier la version installée
```

> 💡 **Convention universelle** : Tout le monde importe NumPy avec l'alias `np`. Vous verrez `np.array()`, `np.zeros()`, etc. dans absolument tous les codes Data Science en Python — c'est presque une signature du langage.

---

## 2. Pourquoi NumPy et pas des listes Python ?

### 2.1 La différence de performance

```python
import numpy as np
import time

# Liste Python classique
liste = list(range(1_000_000))

debut = time.time()
liste_carre = [x**2 for x in liste]
print(f"Liste Python : {time.time() - debut:.4f} secondes")

# Tableau NumPy
tableau = np.arange(1_000_000)

debut = time.time()
tableau_carre = tableau ** 2
print(f"Tableau NumPy : {time.time() - debut:.4f} secondes")

# NumPy est généralement 10 à 100 FOIS plus rapide !
```

### 2.2 Pourquoi cette différence ?

```
LISTE PYTHON                          TABLEAU NUMPY (ndarray)
─────────────────                     ─────────────────────────
[1, "texte", 3.14, True]              [1, 2, 3, 4, 5]
    ↓                                      ↓
Types MÉLANGÉS possibles              Type UNIQUE et FIXE (ex: tous des int64)
    ↓                                      ↓
Chaque élément = un objet Python      Données stockées en bloc CONTIGU
  séparé en mémoire, avec               en mémoire (comme un tableau en C)
  overhead (poids supplémentaire)         ↓
    ↓                                 Calculs VECTORISÉS
Boucles Python nécessaires              (opérations exécutées en code C
  (lentes)                              optimisé, pas en boucle Python)
```

> 💡 **Analogie** : Une liste Python, c'est comme une **file de casiers de tailles différentes**, chacun pouvant contenir n'importe quoi — il faut ouvrir chaque casier un par un pour voir ce qu'il contient. Un tableau NumPy, c'est comme une **rangée de casiers identiques, alignés et de même taille** — on peut traiter tous les casiers d'un coup, comme une chaîne de montage.

### 2.3 Vectorisation — Le concept clé

```python
notes = [14, 16, 12, 18, 10]

# ❌ Avec une liste Python — boucle nécessaire
notes_sur_100 = [n * 5 for n in notes]
print(notes_sur_100)   # [70, 80, 60, 90, 50]

# ✅ Avec NumPy — opération VECTORISÉE (pas de boucle explicite)
import numpy as np
notes_np = np.array([14, 16, 12, 18, 10])
notes_sur_100_np = notes_np * 5
print(notes_sur_100_np)   # [70 80 60 90 50]
```

> 🔑 **Vectorisation** = appliquer une opération à **tout un tableau en une seule instruction**, sans écrire de boucle `for` explicite. C'est plus rapide ET plus lisible.

---

## 3. NumPy Arrays Creation — Créer des tableaux

### 3.1 Depuis une liste Python — `np.array()`

```python
import numpy as np

# Tableau 1D (vecteur)
notes = np.array([14, 16, 12, 18, 15])
print(notes)          # [14 16 12 18 15]
print(type(notes))    # <class 'numpy.ndarray'>

# Tableau 2D (matrice) — depuis une liste de listes
ventes = np.array([
    [4, 8, 12],
    [6, 10, 15],
    [5, 7, 9]
])
print(ventes)
```

### 3.2 Tableaux pré-remplis

```python
# Tableau de zéros
zeros = np.zeros(5)
print(zeros)              # [0. 0. 0. 0. 0.]

zeros_2d = np.zeros((3, 4))     # 3 lignes, 4 colonnes
print(zeros_2d)

# Tableau de uns
uns = np.ones((2, 3))
print(uns)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Tableau rempli d'une valeur constante
constante = np.full((2, 2), 7)
print(constante)
# [[7 7]
#  [7 7]]

# Matrice identité (diagonale de 1, utile en algèbre linéaire)
identite = np.eye(3)
print(identite)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 3.3 Séquences de nombres

```python
# np.arange() — équivalent NumPy de range(), mais retourne un tableau
sequence = np.arange(0, 10, 2)    # début, fin (exclue), pas
print(sequence)   # [0 2 4 6 8]

sequence2 = np.arange(5)          # de 0 à 4
print(sequence2)  # [0 1 2 3 4]

# np.linspace() — génère N valeurs ÉQUIRÉPARTIES entre deux bornes (incluses)
lineaire = np.linspace(0, 1, 5)   # 5 valeurs entre 0 et 1
print(lineaire)   # [0.   0.25 0.5  0.75 1.  ]
```

**Différence arange vs linspace :**

| Fonction | Vous spécifiez | Exemple | Résultat |
|----------|-----------------|---------|----------|
| `np.arange(0, 10, 2)` | Le **pas** entre les valeurs | pas=2 | `[0, 2, 4, 6, 8]` |
| `np.linspace(0, 10, 5)` | Le **nombre** de valeurs voulues | 5 valeurs | `[0, 2.5, 5, 7.5, 10]` |

### 3.4 Tableaux aléatoires (aperçu — détaillé en section 10)

```python
# Tableau de nombres aléatoires entre 0 et 1
alea = np.random.rand(3)
print(alea)   # ex: [0.42 0.71 0.19]
```

### 3.5 Tableau récapitulatif des créations

| Fonction | Description | Exemple |
|----------|-------------|---------|
| `np.array(liste)` | Depuis une liste Python | `np.array([1,2,3])` |
| `np.zeros(n)` | Tableau de zéros | `np.zeros(5)` |
| `np.ones(n)` | Tableau de uns | `np.ones((2,3))` |
| `np.full(shape, val)` | Tableau rempli d'une valeur | `np.full((2,2), 7)` |
| `np.eye(n)` | Matrice identité | `np.eye(3)` |
| `np.arange(deb, fin, pas)` | Séquence par pas | `np.arange(0,10,2)` |
| `np.linspace(deb, fin, n)` | N valeurs équiréparties | `np.linspace(0,1,5)` |

---

