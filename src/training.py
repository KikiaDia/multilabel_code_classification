class Trainer:
    """Gestion de l'entraînement et de la validation."""
    
    def __init__(self, model, processor, evaluator):
        self.model = model
        self.processor = processor
        self.evaluator = evaluator
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Entraîne le modèle avec validation optionnelle."""
        print("\n🚀 Début de l'entraînement...")

        # X_train et X_val sont déjà prêts (TF-IDF seul ou hybride)
        self.model.build(use_precomputed_features=True)
        self.model.fit(X_train, y_train)

        print("✅ Entraînement terminé!")

        if X_val is not None and y_val is not None:
            print("\n📈 Évaluation sur l'ensemble de validation...")
            y_pred_val = self.model.predict(X_val)
            metrics = self.evaluator.compute_metrics(y_val, y_pred_val)
            self.evaluator.print_report(metrics)
            return metrics

        return None
