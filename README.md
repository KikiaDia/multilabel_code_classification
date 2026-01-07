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

**Nettoyage du texte :** Tokenisation (NLTK), Normalisation, Suppression de stopwords, Lemmatisation
### Wordcloud avant preprocessing

![Wordcloud avant preprocessing](images/before_preprocessing.png)

### Wordcloud après preprocessing

![Wordcloud après preprocessing](images/after_preprocessing.png)

### Représentation des labels :
- one-hot-encoding avec `MultiLabelBinarizer`

---

## Text Vectorisation
- TF-IDF:  `max_features = 5000`, `ngram_range = (1, 2)`

## Modélisation

**Stratégies multi-label :**
- One-vs-Rest
- MultiOutputClassifier
- Classifier Chains

**Classificateurs testés :**
- Logistic Regression
- Random Forest
- LinearSVC

### Métriques d'évaluation

- Micro F1-score
- Macro F1-score
- Hamming Loss
- Subset Accuracy
- Precision / Recall par tag


### Best Modèle 

`OneVsRest + LinearSVC (class_weight="balanced")`

Optimisation via `GridSearchCV` (scoring : Micro F1)

### Approches testées :

- description only : 

- description + features extraites du code Python 

- gestion du déséquilibre avec MLSMOTE (Multi-Label SMOTE)

## Evaluation

### Comparaison des approches (triées par Macro F1)

| Approach | F1 Micro | F1 Macro | Hamming Loss | Subset Accuracy |
|----------|----------|----------|--------------|-----------------|
| **Descriptions + code features** | **0.7251** | **0.6842** | **0.0945** | **0.4813** |
| Descriptions only | 0.7265 | 0.6653 | 0.0910 | 0.4832 |
| Resampled (SMOTE) | 0.6980 | 0.6431 | 0.0989 | 0.4627 |

**🏆 Meilleur modèle :** Descriptions + code features

### Matrices de confusion

![Matrices de confusion](images/matrices_confusion.png)

### Métriques détaillées par label pour le meilleur modèle (Descriptions + code features)

| Label | F1 | Precision | Recall | Accuracy |
|-------|---------|-----------|--------|----------|
| games | 0.8571 | 0.8571 | 0.8571 | 0.9925 |
| geometry | 0.6027 | 0.5116 | 0.7333 | 0.9459 |
| graphs | 0.6063 | 0.5423 | 0.6875 | 0.8134 |
| math | 0.8114 | 0.7729 | 0.8539 | 0.8022 |
| number theory | 0.5763 | 0.4766 | 0.7286 | 0.8601 |
| probabilities | 0.4706 | 0.4444 | 0.5000 | 0.9664 |
| strings | 0.8912 | 0.8958 | 0.8866 | 0.9608 |
| trees | 0.6579 | 0.5556 | 0.8065 | 0.9030 |
| **Mean** | **0.6842** | **0.6320** | **0.7567** | **0.9056** |


## Utilisation du CLI

```bash
# training and evaluation on description only

python src/main.py train --data data/train --output models/model.joblib

python src/main.py evaluate --model models/model.joblib --data data/test  

# training and evaluation on description + code features 

python src/main.py train --data data/train --output models/model_hybrid.joblib --hybrid

python src/main.py evaluate --model models/model_hybrid.joblib --data data/test --hybrid --results-path results/predictions_test_hybrid.csv

```

  