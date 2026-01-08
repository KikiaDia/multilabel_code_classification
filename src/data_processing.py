import re
import json
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from code_features import extract_code_features
from concurrent.futures import ThreadPoolExecutor
import os
from glob import glob
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# -----------------------------
# Fixer la seed globale
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet')

class DataProcessor:
    """Gestion du preprocessing et préparation des données pour multi-label classification."""

    TARGET_TAGS = [
        'math', 'graphs', 'strings', 'number theory',
        'trees', 'geometry', 'games', 'probabilities'
    ]

    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.vectorizer = None  # TF-IDF

        
        self.stopwords = set(stopwords.words("english"))
        # self.lemmatizer = WordNetLemmatizer()
        self.stemmer = nltk.stem.SnowballStemmer("english")

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = text.replace('-', ' ')
        text = contractions.fix(text)              # don't → do not
        text = re.sub(r"\d+", " ", text)            # remove digits
        text = re.sub(r"[^a-z\s]", " ", text)       # keep letters only
        text = re.sub(r"\s+", " ", text).strip()

        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stopwords]
        # tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [self.stemmer.stem(word) for word in tokens]

        return " ".join(tokens)


    def parse_tags(self, tags_str) -> List[str]:
        """Parse les tags depuis format JSON, string ou liste."""
        if tags_str is None:
            return []
        if isinstance(tags_str, list):
            return [tag for tag in tags_str if tag in self.TARGET_TAGS]
        if isinstance(tags_str, str):
            tags_str = tags_str.strip()
            if not tags_str:
                return []
            try:
                tags = json.loads(tags_str.replace("'", '"'))
                if isinstance(tags, list):
                    return [tag for tag in tags if tag in self.TARGET_TAGS]
            except json.JSONDecodeError:
                return []
        return []
      
    @staticmethod
    def load_json_file(path):
        """Lit un fichier JSON et retourne un dict avec champs essentiels."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extraire uniquement les champs nécessaires
            return {
                "prob_desc_description": data.get("prob_desc_description", "") or "",
                "source_code": data.get("source_code", "") or "",
                "prob_desc_input_spec": data.get("prob_desc_input_spec", "") or "",
                "prob_desc_output_spec": data.get("prob_desc_output_spec", "") or "",
                "prob_desc_notes": data.get("prob_desc_notes", "") or "",
                "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else []
            }
        except Exception:
            return None
    def load_and_preprocess(self, dataset_dir: str) -> pd.DataFrame:
        """Charge et prétraite le dataset depuis JSON - VERSION OPTIMISÉE."""

        print(f"Chargement des données depuis {dataset_dir}...")
        json_files = glob(os.path.join(dataset_dir, "*.json"))
        print(f"{len(json_files)} fichiers JSON trouvés")

        # Lecture parallèle optimisée
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            records = list(executor.map(DataProcessor.load_json_file, json_files))
        
        # Filtrer les None
        records = [r for r in records if r is not None]
        
        # print(f"Création du DataFrame...")
        df = pd.DataFrame.from_records(records)

        # Parsing des tags et filtrage précoce (avant concaténation coûteuse)
        # print(f"Parsing des tags...")
        df['tags_list'] = df['tags'].apply(self.parse_tags)
        df = df[df['tags_list'].map(len) > 0].reset_index(drop=True)
        print(f"Après filtrage des tags: {len(df)} exemples")

        # Concaténation vectorisée (plus rapide que apply)
        # print(f"Concaténation des descriptions...")
        df['description'] = (
            df['prob_desc_description'].fillna('') + ' ' +
            df['prob_desc_input_spec'].fillna('') + ' ' +
            df['prob_desc_output_spec'].fillna('') + ' ' +
            df['prob_desc_notes'].fillna('')
        )
        
        # Nettoyage en batch
        df['description_clean'] = df['description'].apply(self.clean_text)

        print(f"✅ Dataset prêt: {len(df)} exemples avec tags cibles")
        return df

    def prepare_labels(self, df: pd.DataFrame, fit=True) -> np.ndarray:
        """Prépare les labels multi-label binaires."""
        if fit:
            return self.mlb.fit_transform(df['tags_list'])
        return self.mlb.transform(df['tags_list'])

    def prepare_features(
        self,
        df: pd.DataFrame,
        text_column="description_clean",
        code_column="source_code",
        tfidf_max_features=5000,
        tfidf_ngram_range=(1, 2),
        test_size=0.2,
        random_state=42
    ) -> Tuple:
        """
        Prépare X_train, X_val, y_train, y_val avec TF-IDF et optional features code.
        """
        # -----------------------------
        # Labels
        # -----------------------------
        y = self.prepare_labels(df, fit=True)

        # -----------------------------
        # Split train / val
        # -----------------------------
        train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=random_state)
        X_train_text = df.iloc[train_idx][text_column].tolist()
        X_val_text   = df.iloc[val_idx][text_column].tolist()
        y_train = y[train_idx]
        y_val   = y[val_idx]

        # -----------------------------
        # TF-IDF
        # -----------------------------
        self.vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_val_tfidf   = self.vectorizer.transform(X_val_text)

        # -----------------------------
        # Features code (optionnel)
        # -----------------------------
        if code_column and code_column in df.columns:
            X_train_code_features = np.array([extract_code_features(df.iloc[i][code_column]) for i in train_idx])
            X_val_code_features   = np.array([extract_code_features(df.iloc[i][code_column]) for i in val_idx])

            X_train_code_sparse = csr_matrix(X_train_code_features)
            X_val_code_sparse   = csr_matrix(X_val_code_features)

            # Combiner TF-IDF + features code
            X_train_combined = hstack([X_train_tfidf, X_train_code_sparse])
            X_val_combined   = hstack([X_val_tfidf, X_val_code_sparse])
        else:
            X_train_combined = X_train_tfidf
            X_val_combined   = X_val_tfidf

        print("✅ Shape X_train_tfidf :", X_train_tfidf.shape)
        print("✅ Shape X_val_tfidf   :", X_val_tfidf.shape)
        print("✅ Shape X_train_combined :", X_train_combined.shape)
        print("✅ Shape X_val_combined   :", X_val_combined.shape)

        return X_train_tfidf, X_val_tfidf, X_train_combined, X_val_combined, y_train, y_val, train_idx, val_idx
