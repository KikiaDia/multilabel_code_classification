# Codeforces – Multilabel Classification of Algorithmic Problems

**Auteur : Kikia Dia**  
**Illuin Technology Challenge : Data Science**

---

## Contexte

Codeforces est une plateforme de programmation compétitive regroupant des milliers de problèmes d’algorithmique, chacun annoté par plusieurs **tags** représentant les notions algorithmiques mobilisées (`math`, `graphs`, `strings`, etc.).

Ce projet s’appuie sur un **sous-ensemble du dataset xCodeEval** composé de **4 982 problèmes distincts**, incluant :
- descriptions textuelles complètes,
- spécifications d’entrée/sortie,
- notes éventuelles,
- solutions validées en Python,
- annotations multi-labels.

---

## Objectif

Construire un **algorithme de classification multi-label** capable de prédire automatiquement les tags associés à un problème d’algorithmique.

L’étude se concentre sur les **8 tags suivants** :

```python
['math', 'graphs', 'strings', 'number theory',
 'trees', 'geometry', 'games', 'probabilities']

## 🔍 Exploratory Data Analysis (EDA)

- Distribution des tags
- Co-occurrence des labels
- Longueur des descriptions
- Wordclouds par tag
- Analyse des patterns algorithmiques dans le code

---

## 🧹 Prétraitement du texte

**Champs utilisés :**
- `prob_desc_description`
- `prob_desc_input_spec`
- `prob_desc_output_spec`
- `prob_desc_notes`

**Étapes :**
- Nettoyage et normalisation
- Tokenisation (NLTK)
- Suppression de stopwords
- Lemmatisation

---

## 🧾 Représentation des labels

- Classification **multi-label**
- `MultiLabelBinarizer`
- Suppression des exemples hors tags cibles

---

## 📐 Vectorisation

- TF-IDF
- `max_features = 5000`
- `ngram_range = (1, 2)`

---

## 🤖 Modélisation

**Stratégies multi-label :**
- One-vs-Rest
- MultiOutputClassifier
- Classifier Chains

**Classificateurs testés :**
- Logistic Regression
- Random Forest
- LinearSVC

---

## 📊 Métriques d’évaluation

- Micro F1-score
- Macro F1-score
- Hamming Loss
- Subset Accuracy
- Precision / Recall par tag

---

## 🏆 Modèle retenu

`OneVsRest + LinearSVC (class_weight="balanced")`

Optimisation via `GridSearchCV` (scoring : Micro F1)

---

## 🔗 Approche hybride : texte + code

**Features extraites du code Python :**
- DFS / BFS
- Récursion
- Opérations modulo
- Structures de graphes et d’arbres
- Indices liés aux jeux et probabilités

---

## ⚖️ Gestion du déséquilibre

- MLSMOTE (Multi-Label SMOTE)
- Amélioration du Macro F1-score
- Meilleure prédiction des tags rares

---

## 🚀 Utilisation

**Entraînement :**
```bash
python src/train.py --data_path data/code_classification_dataset
