import click
import time
import sys
import os
import joblib
import numpy as np
import pandas as pd
from data_processing import DataProcessor
from models import BaselineModel
from evaluation import Evaluator
from training import Trainer
from code_features import extract_code_features
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
import warnings

warnings.filterwarnings("ignore")

@click.group()
def cli():
    """CLI pour le classifieur de tags Codeforces."""
    pass


@cli.command()
@click.option('--data', required=True, help='Chemin vers le dossier contenant les fichiers JSON')
@click.option('--output', default='model.joblib', help='Chemin de sauvegarde du modèle')
@click.option('--hybrid', is_flag=True, help='Activer TF-IDF + features code')
def train(data, output, hybrid):
    """Entraîne le modèle sur les données."""
    print("\n🎯 Démarrage du training...\n")

    processor = DataProcessor()
    df = processor.load_and_preprocess(data)
    evaluator = Evaluator(processor.TARGET_TAGS)
    model = BaselineModel(max_features=5000)
    trainer = Trainer(model, processor, evaluator)

    code_column = "source_code" if hybrid else None
    X_train_tfidf, X_val_tfidf, X_train_combined, X_val_combined, y_train, y_val, train_idx, val_idx = \
        processor.prepare_features(df, code_column=code_column)

    # Choisir les matrices selon hybrid ou non
    X_train_matrix = X_train_combined if hybrid else X_train_tfidf
    X_val_matrix   = X_val_combined   if hybrid else X_val_tfidf

    # Entraînement
    trainer.train(
        X_train_matrix, y_train,
        X_val_matrix, y_val
    )

    # Sauvegarde
    model.save(output)

    joblib.dump(processor.mlb, output.replace('.joblib', '_mlb.joblib'))
    joblib.dump(processor.vectorizer, output.replace('.joblib', '_tfidf.joblib'))

    click.echo(f"\n✅ Modèle sauvegardé: {output}")


# -----------------------------
# Commande : ÉVALUATION
# -----------------------------
@cli.command()
@click.option('--model', required=True, help='Chemin vers le modèle')
@click.option('--data', required=True, help='Chemin vers le dossier JSON pour test')
@click.option('--hybrid', is_flag=True, help='Activer TF-IDF + features code')
@click.option('--output', default=None, help='Chemin pour sauvegarder le graphique de performance')
@click.option('--results-path', default=None, help='Chemin pour sauvegarder les résultats détaillés (CSV)')
def evaluate(model, data, hybrid, output,results_path):
    """Évalue le modèle sur un ensemble de test complet."""
    print("\n🎯 Démarrage de l’évaluation...\n")
    
    start_total = time.perf_counter()

    # -----------------------------
    # Chargement du modèle et du ML Binarizer
    # -----------------------------
    processor = DataProcessor()
    clf = BaselineModel.load(model)
    processor.mlb = joblib.load(model.replace('.joblib', '_mlb.joblib'))
    
    # Toujours charger le TF-IDF
    processor.vectorizer = joblib.load(model.replace('.joblib', '_tfidf.joblib'))
        
    evaluator = Evaluator(processor.mlb.classes_)
    
    t0 = time.perf_counter()
    # -----------------------------
    # Chargement et prétraitement du test
    # -----------------------------
    df = processor.load_and_preprocess(data)
    y_true = processor.prepare_labels(df, fit=False)
    
    print(f"\nTemps de Chargement + preprocessing test: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    # -----------------------------
    # Préparer la matrice d'entrée
    # -----------------------------
    if hybrid:
        # TF-IDF
        X_text = processor.vectorizer.transform(df['description_clean'])
        # features code
        X_code = csr_matrix([extract_code_features(c) for c in df['source_code']])
        X_combined = hstack([X_text, X_code])
    else:
        X_combined = processor.vectorizer.transform(df['description_clean'])

    print(f"Temps de Feature engineering test: {time.perf_counter() - t0:.2f}s")
    
    t0 = time.perf_counter()
    # -----------------------------
    # Prédiction
    # -----------------------------
    y_pred = clf.predict(X_combined)
    print(f"Temps de Prédiction: {time.perf_counter() - t0:.2f}s")

    metrics = evaluator.compute_metrics(y_true, y_pred)
    evaluator.print_report(metrics)

    total_time = time.perf_counter() - start_total
    print(f"\n Temps d'exécution total : {total_time:.2f}s\n")
    
    true_tags = processor.mlb.inverse_transform(y_true)
    pred_tags = processor.mlb.inverse_transform(y_pred)

    results_df = pd.DataFrame({
    "description": df["description"],
    "source_code": df["source_code"],
    "ground_truth_tags": [list(tags) for tags in true_tags],
    "predicted_tags": [list(tags) for tags in pred_tags],
    })
    
    if results_path:
        results_df.to_csv(results_path, index=False)
        print(f"\n Résultats détaillés sauvegardés dans : {results_path}")


    # -----------------------------
    # Optionnel : graphique
    # -----------------------------
    # if output:
    #     evaluator.plot_performance(metrics, save_path=output)


# -----------------------------
# Commande : PREDICTION SUR UN EXEMPLE
# -----------------------------
@cli.command()
@click.option('--model', required=True, help='Chemin vers le modèle')
@click.option('--text', required=True, help='Description du problème')
@click.option('--code', default=None, help='Code source associé (optionnel)')
def predict(model, text, code):
    """Prédiction pour un seul exemple."""
    processor = DataProcessor()
    clf = BaselineModel.load(model)
    processor.mlb = joblib.load(model.replace('.joblib', '_mlb.joblib'))

    # Charger le TF-IDF associé au modèle
    processor.vectorizer = joblib.load(model.replace('.joblib', '_tfidf.joblib'))

    # -----------------------------
    # Préparation de l'exemple
    # -----------------------------
    text_clean = processor.clean_text(text)
    X_input = processor.vectorizer.transform([text_clean])

    if code:
        X_code = csr_matrix([extract_code_features(code)])
        X_input = hstack([X_input, X_code])

    # -----------------------------
    # Prédiction
    # -----------------------------
    start = time.time()
    y_pred = clf.predict(X_input)  # <- array 2D
    predicted_tags = processor.mlb.inverse_transform(y_pred)[0]

    proba = expit(clf.model.decision_function(X_input))[0]

    duration = time.time() - start

    click.echo(f"\n📝 Texte: {text[:100]}...")
    click.echo(f"\n🏷️  Tags prédits: {', '.join(predicted_tags) if predicted_tags else 'Aucun'}")
    click.echo(f"\n📊 Probabilités:")
    for i, tag in enumerate(processor.mlb.classes_):
        click.echo(f"  {tag:<20} {proba[i]:.4f}")
    click.echo(f"\n⏱️  Temps de prédiction: {duration:.4f}s")


if __name__ == '__main__':
    cli()
