# src/code_features.py
import re

import re

def extract_code_features(code: str):
    code_l = code.lower()
    features = {}

    # =====================================================
    # MATH / NUMBER THEORY
    # =====================================================
    features["has_mod"] = int("%" in code)
    features["has_pow"] = int("**" in code or "pow(" in code)
    features["has_gcd"] = int("gcd" in code_l)
    features["has_lcm"] = int("lcm" in code_l)
    features["has_prime"] = int("prime" in code_l)
    features["has_factorial"] = int("fact" in code_l)
    features["has_math_import"] = int("import math" in code_l)
    features["has_bit_ops"] = int(any(op in code for op in ["<<", ">>", "&", "|", "^"]))

    # =====================================================
    # GRAPHS
    # =====================================================
    features["has_dfs"] = int("dfs" in code_l)
    features["has_bfs"] = int("bfs" in code_l)
    features["has_adj_list"] = int("adj" in code_l or "neighbors" in code_l)
    features["has_edges"] = int("edges" in code_l)
    features["has_queue"] = int("deque" in code_l or "queue" in code_l)
    features["has_stack"] = int("stack" in code_l)
    features["has_visited"] = int("visited" in code_l)

    # =====================================================
    # TREES
    # =====================================================
    features["has_tree"] = int("tree" in code_l)
    features["has_node"] = int("node" in code_l)
    features["has_left_right"] = int("left" in code_l and "right" in code_l)
    features["has_recursion"] = int(code.count("def") > 1)
    features["has_depth"] = int("depth" in code_l or "height" in code_l)

    # =====================================================
    # STRINGS
    # =====================================================
    features["has_string_literal"] = int(bool(re.search(r"['\"]", code)))
    features["has_split"] = int(".split(" in code)
    features["has_join"] = int(".join(" in code)
    features["has_replace"] = int(".replace(" in code)
    features["has_substring"] = int("substr" in code_l or "substring" in code_l)
    features["has_ord_chr"] = int("ord(" in code or "chr(" in code)

    # =====================================================
    # PROBABILITIES
    # =====================================================
    features["has_probability"] = int("prob" in code_l or "chance" in code_l)
    features["has_fraction"] = int("fraction" in code_l)
    features["has_float_div"] = int("/" in code and "//" not in code)
    features["has_expectation"] = int("expect" in code_l)

    # =====================================================
    # GAMES
    # =====================================================
    features["has_random"] = int("random" in code_l)
    features["has_turn"] = int("turn" in code_l)
    features["has_player"] = int("player" in code_l)
    features["has_score"] = int("score" in code_l)
    features["has_game_dp"] = int("dp" in code_l and ("win" in code_l or "lose" in code_l))

    # =====================================================
    # GEOMETRY
    # =====================================================
    features["has_point"] = int("point" in code_l)
    features["has_distance"] = int("dist" in code_l)
    features["has_angle"] = int("angle" in code_l)
    features["has_cross_product"] = int("cross" in code_l)
    features["has_dot_product"] = int("dot" in code_l)
    features["has_hypot"] = int("hypot" in code_l)

    return list(features.values())
