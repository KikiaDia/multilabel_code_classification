from sklearn.metrics import (
    hamming_loss, accuracy_score, precision_recall_fscore_support,
    classification_report, multilabel_confusion_matrix
)
from typing import List, Dict

class Evaluator:
    """Évaluation complète des performances multi-label."""
    
    def __init__(self, target_tags: List[str]):
        self.target_tags = target_tags
        
    def compute_metrics(self, y_true, y_pred) -> Dict:
        """Calcule les métriques globales et prépare le rapport par tag."""
        metrics = {}
        
        # # Identifier les colonnes (tags) avec au moins un vrai positif
        # non_zero_classes = y_true.sum(axis=0) > 0
        # y_true = y_true[:, non_zero_classes]
        # y_pred = y_pred[:, non_zero_classes]


        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)

        # Micro et Macro
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred , average='micro', zero_division=0
        )

        # Calculer la macro uniquement sur ces classes
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        metrics['micro_precision'] = prec_micro
        metrics['micro_recall'] = rec_micro
        metrics['micro_f1'] = f1_micro
        metrics['macro_f1'] = f1_macro

        # target_names_filtered = [tag for i, tag in enumerate(self.target_tags) if non_zero_classes[i]]

        # Rapport détaillé par tag
        metrics['report'] = classification_report(
            y_true, y_pred, target_names= self.target_tags, 
            zero_division=0
        )
        return metrics

    def print_report(self, metrics: Dict):
        """Affiche les métriques globales et le rapport par tag."""
        print("\n" + "="*70)
        print("RAPPORT D'ÉVALUATION - CLASSIFICATION MULTI-LABEL")
        print("="*70)

        # ✅ Affichage des métriques globales simplifiées
        print(f"\n✅ Micro F1: {metrics['micro_f1']:.3f}, Macro F1: {metrics['macro_f1']:.3f}")
        print(f"Hamming Loss: {metrics['hamming_loss']:.3f}, Subset Accuracy: {metrics['subset_accuracy']:.3f}")
        print(f"Micro Precision: {metrics['micro_precision']:.3f}, Micro Recall: {metrics['micro_recall']:.3f}")

        # Rapport détaillé par tag
        print("\n - Classification report par tag :")
        print(metrics['report'])
        print("="*70 + "\n")
    
    # def plot_performance(self, metrics: Dict, save_path: str = None):
    #     """Visualisation des performances par tag."""
    #     tags = list(metrics['per_tag'].keys())
    #     f1_scores = [metrics['per_tag'][tag]['f1'] for tag in tags]
    #     precisions = [metrics['per_tag'][tag]['precision'] for tag in tags]
    #     recalls = [metrics['per_tag'][tag]['recall'] for tag in tags]
        
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     x = np.arange(len(tags))
    #     width = 0.25
        
    #     ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    #     ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    #     ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
    #     ax.set_xlabel('Tags', fontsize=12)
    #     ax.set_ylabel('Score', fontsize=12)
    #     ax.set_title('Performance par Tag', fontsize=14, fontweight='bold')
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(tags, rotation=45, ha='right')
    #     ax.legend()
    #     ax.grid(axis='y', alpha=0.3)
        
    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"Graphique sauvegardé: {save_path}")
    #     else:
    #         plt.show()