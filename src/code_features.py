# src/code_features.py
import re

def extract_code_features(code: str):
    """
    Extraire des features pertinentes pour prédire les tags d'algorithmique.
    Retourne une liste de valeurs binaires.
    """
    features = {}
    code_lower = code.lower() if isinstance(code, str) else ""
    
    # -----------------------------
    # 1️⃣ Structures mathématiques
    # -----------------------------
    features['has_mod'] = int('%' in code)
    features['has_pow'] = int('**' in code or 'pow(' in code)
    features['has_factorial'] = int('fact' in code)
    features['has_comb'] = int('comb' in code)
    features['has_math_import'] = int('import math' in code)
    
    # -----------------------------
    # Graphes / parcours
    # -----------------------------
    features['has_dfs'] = int('dfs' in code_lower)
    features['has_bfs'] = int('bfs' in code_lower)
    features['has_edges'] = int('edges' in code_lower)
    features['has_adj'] = int('adj' in code_lower)
    features['has_graph_list'] = int('graph' in code_lower)
    
    # -----------------------------
    # Récursion / structures arborescentes
    # -----------------------------
    features['has_recursion'] = int('def' in code and code.count('def') > 1)
    features['has_tree'] = int('tree' in code_lower)
    
    # -----------------------------
    # Chaînes et manipulation de strings
    # -----------------------------
    features['has_string'] = int('str(' in code or '"' in code or "'" in code)
    features['has_join'] = int('.join(' in code)
    
    # -----------------------------
    # Jeux / probabilités
    # -----------------------------
    features['has_random'] = int('random' in code)
    features['has_probability'] = int('prob' in code_lower or 'chance' in code_lower)
    
    # -----------------------------
    # Boucles / itérations
    # -----------------------------
    features['has_while'] = int('while ' in code)
    
    # -----------------------------
    # Listes / tableaux
    # -----------------------------
    features['has_append'] = int('.append(' in code)
    
    return list(features.values())
