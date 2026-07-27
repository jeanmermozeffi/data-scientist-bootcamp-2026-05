# 🗂️ Data Structures in Python — Cours Bootcamp Data Science

> **Chapitre 3** | Prérequis : Chapitre 1 (Introduction) + Chapitre 2 (Bases)

---

## Table des matières

1. [Vue d'ensemble des structures de données](#1-vue-densemble-des-structures-de-données)
2. [List — Les listes](#2-list--les-listes)
3. [List Manipulation — Manipuler les listes](#3-list-manipulation--manipuler-les-listes)
4. [Dictionary — Les dictionnaires](#4-dictionary--les-dictionnaires)
5. [Dictionary Manipulation — Manipuler les dictionnaires](#5-dictionary-manipulation--manipuler-les-dictionnaires)
6. [Sets — Les ensembles](#6-sets--les-ensembles)
7. [Tuple — Les tuples](#7-tuple--les-tuples)
8. [Nested Structures — Structures imbriquées](#8-nested-structures--structures-imbriquées)
9. [Choisir la bonne structure](#9-choisir-la-bonne-structure)
10. [Conclusion](#10-conclusion)
11. [✅ Point de contrôle — Data Structures](#11--point-de-contrôle--data-structures)

---

## 1. Vue d'ensemble des structures de données

### 📖 Qu'est-ce qu'une structure de données ?

Une **structure de données** est une façon organisée de **stocker et regrouper plusieurs valeurs** dans une seule variable. Plutôt que de créer 30 variables pour 30 étudiants, on les regroupe dans une seule structure.

> 💡 **Analogie** : Imaginez que vous devez ranger des documents.
> - Une **liste** → un **classeur avec des pages numérotées** (ordre garanti)
> - Un **dictionnaire** → un **répertoire avec des étiquettes** (clé → valeur)
> - Un **ensemble** → un **sac de billes uniques** (pas de doublons)
> - Un **tuple** → une **archive scellée** (immuable, ne peut pas être modifiée)

### 1.1 Les 4 structures fondamentales

```
STRUCTURES DE DONNÉES PYTHON
│
├── list   [ ]   → Ordonnée, modifiable, doublons autorisés
├── dict   { }   → Paires clé:valeur, modifiable
├── set    { }   → Non ordonnée, unique, modifiable
└── tuple  ( )   → Ordonnée, NON modifiable (immuable)
```

### 1.2 Tableau comparatif

| Propriété | `list` | `dict` | `set` | `tuple` |
|-----------|--------|--------|-------|---------|
| **Syntaxe** | `[...]` | `{clé: val}` | `{...}` | `(...)` |
| **Ordonnée** | ✅ Oui | ✅ Oui (Python 3.7+) | ❌ Non | ✅ Oui |
| **Modifiable** | ✅ Oui | ✅ Oui | ✅ Oui | ❌ Non |
| **Doublons** | ✅ Oui | ❌ Clés uniques | ❌ Non | ✅ Oui |
| **Accès** | Par indice `[0]` | Par clé `["nom"]` | Itération | Par indice `[0]` |
| **Usage typique** | Collection ordonnée | Données structurées | Valeurs uniques | Données fixes |

---

## 2. List — Les listes

### 📖 Définition

Une **liste** est une collection **ordonnée** et **modifiable** d'éléments. C'est la structure de données la plus utilisée en Python.

> 💡 **Analogie** : Une liste Python, c'est comme une **liste de courses** : vous pouvez ajouter des articles, en retirer, les réorganiser, et accéder à chaque article par sa position.

### 2.1 Créer une liste

```python
# Liste vide
liste_vide = []
liste_vide2 = list()

# Liste de nombres
notes = [14, 16, 12, 18, 15]

# Liste de chaînes
etudiants = ["Alice", "Bob", "Claire", "David"]

# Liste mixte (différents types)
profil = ["Alice", 23, 15.75, True]

# Liste de listes (imbriquée)
matrice = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

print(notes)      # [14, 16, 12, 18, 15]
print(type(notes))# <class 'list'>
print(len(notes)) # 5
```

### 2.2 Accéder aux éléments — Indexation

```python
fruits = ["mangue", "papaye", "ananas", "banane", "orange"]
#           0          1         2         3         4
#          -5         -4        -3        -2        -1

# Accès par indice positif
print(fruits[0])   # mangue  (premier)
print(fruits[2])   # ananas
print(fruits[4])   # orange  (dernier)

# Accès par indice négatif (depuis la fin)
print(fruits[-1])  # orange  (dernier)
print(fruits[-2])  # banane
print(fruits[-5])  # mangue  (premier)
```

### 2.3 Slicing — Extraire une sous-liste

```python
notes = [14, 16, 12, 18, 15, 10, 19]
#         0   1   2   3   4   5   6

print(notes[1:4])    # [16, 12, 18]  (indices 1, 2, 3)
print(notes[:3])     # [14, 16, 12]  (du début jusqu'à 3 exclu)
print(notes[3:])     # [18, 15, 10, 19] (de 3 jusqu'à la fin)
print(notes[::2])    # [14, 12, 15, 19] (un sur deux)
print(notes[::-1])   # [19, 10, 15, 18, 12, 16, 14] (inversée)
```

### 2.4 Parcourir une liste

```python
etudiants = ["Alice", "Bob", "Claire"]

# Boucle for simple
for etudiant in etudiants:
    print(f"Bonjour {etudiant} !")

# Avec enumerate (indice + valeur)
for i, etudiant in enumerate(etudiants):
    print(f"{i+1}. {etudiant}")
# 1. Alice
# 2. Bob
# 3. Claire

# List comprehension (façon pythonique rapide)
majuscules = [e.upper() for e in etudiants]
print(majuscules)  # ['ALICE', 'BOB', 'CLAIRE']
```

---

## 3. List Manipulation — Manipuler les listes

### 3.1 Ajouter des éléments

```python
courses = ["riz", "huile", "sucre"]

# .append() — ajouter à la FIN
courses.append("sel")
print(courses)  # ['riz', 'huile', 'sucre', 'sel']

# .insert(indice, valeur) — insérer à une POSITION précise
courses.insert(1, "tomates")
print(courses)  # ['riz', 'tomates', 'huile', 'sucre', 'sel']

# .extend() — fusionner avec une autre liste
nouveaux = ["poivre", "cube maggi"]
courses.extend(nouveaux)
print(courses)  # ['riz', 'tomates', 'huile', 'sucre', 'sel', 'poivre', 'cube maggi']
```

### 3.2 Supprimer des éléments

```python
notes = [14, 16, 12, 14, 18, 12]

# .remove(valeur) — supprime la PREMIÈRE occurrence
notes.remove(14)
print(notes)  # [16, 12, 14, 18, 12]

# .pop(indice) — supprime et RETOURNE l'élément à l'indice donné
dernier = notes.pop()    # sans indice → supprime le dernier
print(dernier)  # 12
print(notes)    # [16, 12, 14, 18]

element = notes.pop(1)   # supprime l'indice 1
print(element)  # 12
print(notes)    # [16, 14, 18]

# del — supprimer par indice ou slice
del notes[0]
print(notes)    # [14, 18]

# .clear() — vider toute la liste
notes.clear()
print(notes)    # []
```

### 3.3 Modifier des éléments

```python
etudiants = ["Alice", "Bob", "Claire"]

# Modifier par indice
etudiants[1] = "Bobby"
print(etudiants)  # ['Alice', 'Bobby', 'Claire']

# Modifier un slice
etudiants[0:2] = ["Anna", "Bruno"]
print(etudiants)  # ['Anna', 'Bruno', 'Claire']
```

### 3.4 Trier une liste

```python
notes = [14, 18, 12, 16, 10, 15]

# .sort() — trier EN PLACE (modifie la liste originale)
notes.sort()
print(notes)              # [10, 12, 14, 15, 16, 18]

notes.sort(reverse=True)
print(notes)              # [18, 16, 15, 14, 12, 10]

# sorted() — retourne une NOUVELLE liste triée
original = [14, 18, 12, 16]
nouvelle = sorted(original)
print(original)  # [14, 18, 12, 16]  ← inchangé
print(nouvelle)  # [12, 14, 16, 18]  ← nouvelle liste

# Trier des chaînes (ordre alphabétique)
noms = ["Charlie", "Alice", "Bob"]
noms.sort()
print(noms)  # ['Alice', 'Bob', 'Charlie']
```

### 3.5 Méthodes utiles

```python
notes = [14, 16, 12, 18, 14, 15]

print(notes.count(14))   # 2   — nombre d'occurrences de 14
print(notes.index(18))   # 3   — indice de la première occurrence de 18
print(min(notes))        # 12  — valeur minimale
print(max(notes))        # 18  — valeur maximale
print(sum(notes))        # 89  — somme de tous les éléments
print(sum(notes)/len(notes))  # 14.83 — moyenne

notes.reverse()          # inverser la liste EN PLACE
print(notes)             # [15, 14, 18, 12, 16, 14]

# Vérifier l'appartenance
print(18 in notes)       # True
print(20 in notes)       # False
print(20 not in notes)   # True
```

### 3.6 List Comprehension — La syntaxe élégante

```python
# Syntaxe : [expression for element in iterable if condition]

# Carrés de 1 à 10
carres = [x**2 for x in range(1, 11)]
print(carres)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Notes supérieures à 14
notes = [14, 16, 12, 18, 10, 15]
bonnes_notes = [n for n in notes if n > 14]
print(bonnes_notes)  # [16, 18, 15]

# Transformer une liste
noms = ["alice", "bob", "claire"]
noms_maj = [n.capitalize() for n in noms]
print(noms_maj)  # ['Alice', 'Bob', 'Claire']
```

### 3.7 Tableau récapitulatif des méthodes de liste

| Méthode | Description | Exemple |
|---------|-------------|---------|
| `.append(x)` | Ajouter à la fin | `lst.append(5)` |
| `.insert(i, x)` | Insérer à la position i | `lst.insert(0, 5)` |
| `.extend(lst2)` | Fusionner deux listes | `lst.extend([6, 7])` |
| `.remove(x)` | Supprimer première occurrence | `lst.remove(5)` |
| `.pop(i)` | Supprimer et retourner à l'indice i | `lst.pop(0)` |
| `.sort()` | Trier en place | `lst.sort()` |
| `.reverse()` | Inverser en place | `lst.reverse()` |
| `.count(x)` | Compter les occurrences | `lst.count(5)` |
| `.index(x)` | Trouver l'indice | `lst.index(5)` |
| `.clear()` | Vider la liste | `lst.clear()` |
| `len(lst)` | Longueur | `len(lst)` |
| `x in lst` | Vérifier appartenance | `5 in lst` |

---

## 4. Dictionary — Les dictionnaires

### 📖 Définition

Un **dictionnaire** est une collection de paires **clé : valeur**. Chaque valeur est accessible via sa clé (comme un dictionnaire réel : le mot est la clé, la définition est la valeur).

> 💡 **Analogie** : Un dictionnaire Python, c'est comme un **annuaire téléphonique**. Vous cherchez un nom (clé) et vous trouvez le numéro (valeur). Vous n'avez pas besoin de parcourir tout l'annuaire — vous accédez directement à l'information via le nom.

### 4.1 Créer un dictionnaire

```python
# Dictionnaire vide
vide = {}
vide2 = dict()

# Dictionnaire d'un étudiant
etudiant = {
    "nom"     : "Alice Dupont",
    "age"     : 23,
    "note"    : 16.5,
    "actif"   : True,
    "cours"   : ["SQL", "Python", "ML"]
}

print(etudiant)
print(type(etudiant))  # <class 'dict'>
print(len(etudiant))   # 5 (nombre de paires clé:valeur)
```

### 4.2 Accéder aux valeurs

```python
etudiant = {
    "nom"  : "Alice",
    "age"  : 23,
    "ville": "Abidjan"
}

# Accès par clé (avec [])
print(etudiant["nom"])    # Alice
print(etudiant["age"])    # 23

# Accès avec .get() — plus sûr (pas d'erreur si la clé n'existe pas)
print(etudiant.get("ville"))        # Abidjan
print(etudiant.get("email"))        # None  (pas d'erreur !)
print(etudiant.get("email", "N/A")) # N/A   (valeur par défaut)

# ❌ Accès direct à une clé inexistante → erreur
# print(etudiant["email"])  # KeyError: 'email'
```

### 4.3 Parcourir un dictionnaire

```python
etudiant = {"nom": "Alice", "age": 23, "note": 16.5}

# Parcourir les clés (par défaut)
for cle in etudiant:
    print(cle)                        # nom, age, note

# Parcourir les valeurs
for valeur in etudiant.values():
    print(valeur)                     # Alice, 23, 16.5

# Parcourir les paires clé:valeur
for cle, valeur in etudiant.items():
    print(f"{cle} → {valeur}")
# nom → Alice
# age → 23
# note → 16.5
```

---

## 5. Dictionary Manipulation — Manipuler les dictionnaires


