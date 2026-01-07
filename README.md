# Codeforces – Multilabel Classification of Algorithmic Problems

**Auteur :** Kikia Dia  
**Illuin Technology Challenge :** Data Science

---

## Contexte

Codeforces est une plateforme de programmation compétitive regroupant des milliers de problèmes d'algorithmique, chacun annoté par plusieurs **tags** représentant les notions algorithmiques mobilisées (`math`, `graphs`, `strings`, etc.).

Ce projet s'appuie sur un **sous-ensemble du dataset xCodeEval** composé de **4 982 problèmes distincts**, incluant :
- `prob_desc_description` : descriptions textuelles complètes ,
- `prob_desc_input_spec`, `prob_desc_output_spec` : spécifications d'entrée/sortie,
- `prob_desc_notes` : notes éventuelles,
- `source_code` : solutions validées en Python,
- `tags` : annotations multi-labels.

## Objectif

Construire un **algorithme de classification multi-label** capable de prédire automatiquement les tags associés à un problème d'algorithmique.

L'étude se concentre sur les **8 tags suivants** :
```python
['math', 'graphs', 'strings', 'number theory',
 'trees', 'geometry', 'games', 'probabilities']
```
Après filtrage pour ne garder que les exemples correspondant à ces tags, le dataset final contient **2 678 problèmes**.

### Dataset preview

| prob_desc_description | prob_desc_input_spec | prob_desc_output_spec | prob_desc_notes | source_code | tags |
|--------------------|------------------|-------------------|---------------|-------------------|------|
| Once upon a time, Oolimry saw a suffix array... | The first line contain 2 integers $$$n$$$ and ... | Print how many strings produce such a suffix array... | NoteIn the first test case, "abb" is the only ... | import sys\ninput = sys.stdin.readline\n... | [math] |
| You are given an array of n elements, you must... | The first line contains integer n (1 ≤ n ≤ 100...) | Print integer k on the first line — the least ... |  | def gcd(a,b):\n if b==0:return a\n return ... | [number theory, math] |

## Structure du projet
```
├── notebooks/
│   ├── EDA.ipynb
│   └── Machine_learning_models.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── training.py
│   ├── evaluate.py
│   ├── main.py
│   └── code_features.py
├── models/
│   ├── model.joblib
│   └── model_tfidf.joblib
├── README.md
└── requirements.txt
```
## Exploratory Data Analysis (EDA)

- Longueur des descriptions
- Distribution des tags
- Co-occurrence des labels
- Wordclouds par tag
- Analyse des patterns algorithmiques dans le code

### Text Preprocessing

**Champs concatenés  pour la description :** `prob_desc_description`,`prob_desc_input_spec`,`prob_desc_output_spec`, `prob_desc_notes`

**Nettoyage du texte :**
- Tokenisation (NLTK)
- Normalisation
- Suppression de stopwords
- Lemmatisation

## Représentation des labels

- Classification **multi-label**
- one-hot-encoding avec `MultiLabelBinarizer`

## Text Vectorisation

- TF-IDF
- `max_features = 5000`
- `ngram_range = (1, 2)`

## Modélisation

**Stratégies multi-label :**
- One-vs-Rest
- MultiOutputClassifier
- Classifier Chains

**Classificateurs testés :**
- Logistic Regression
- Random Forest
- LinearSVC

---

## Métriques d'évaluation

- Micro F1-score
- Macro F1-score
- Hamming Loss
- Subset Accuracy
- Precision / Recall par tag

---

## Modèle retenu

`OneVsRest + LinearSVC (class_weight="balanced")`

Optimisation via `GridSearchCV` (scoring : Micro F1)


---

## Approche hybride : texte + code features

**Features extraites du code Python :**
- DFS / BFS
- Récursion
- Opérations modulo
- Structures de graphes et d'arbres
- Indices liés aux jeux et probabilités

---

## Gestion du déséquilibre

- MLSMOTE (Multi-Label SMOTE)
- Amélioration du Macro F1-score


## Evaluation 




## Utilisation

**Entraînement :**
```bash
python src/train.py --data_path data/code_classification_dataset
```
