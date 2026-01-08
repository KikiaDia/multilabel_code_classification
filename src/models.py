from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from typing import List
from scipy.special import expit 


class BaselineModel:
    """Modèle baseline: TF-IDF + Logistic Regression avec OneVsRest."""
    
    def __init__(self, max_features: int = 5000 ):
        self.max_features = max_features
        # self.C = C
        self.model = None
        
    def build(self, use_precomputed_features=True):
        """Construit le modèle"""
        linear_svc = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            fit_intercept=True,
            intercept_scaling=1,
            loss="squared_hinge",
            penalty="l2",
            tol=1e-4,
            max_iter=1000,
            random_state=42
        )
        if use_precomputed_features:
            self.model = OneVsRestClassifier(
                linear_svc,
                n_jobs=-1
            )
        else:
            # Pipeline classique si on passe des textes bruts
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )),
                ('clf', OneVsRestClassifier(
                    linear_svc,
                    n_jobs=-1
                ))
            ])

    
    def fit(self, X_text: List[str], y: np.ndarray):
        """Entraînement du modèle."""
        self.model.fit(X_text, y)
        return self
    
    def predict(self, X_text: List[str]) -> np.ndarray:
        """Prédiction."""
        return self.model.predict(X_text)
    
    def predict_proba(self, X_text: List[str]) -> np.ndarray:
        """Simule des probabilités à partir de decision_function."""
        try:
            scores = self.model.decision_function(X_text)
            return expit(scores)  # Sigmoid pour obtenir des valeurs entre 0 et 1
        except AttributeError:
            raise NotImplementedError(
                "Le modèle n'a pas de `decision_function`. Utilisez `predict` directement."
            )
    
    def save(self, filepath: str):
        """Sauvegarde du modèle."""
        joblib.dump(self.model, filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Chargement du modèle."""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance