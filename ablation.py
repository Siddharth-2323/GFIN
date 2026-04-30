import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Import from GFIN
from GFIN import (
    load_pima, borderline_smote, rfe_select, apply_interactions,
    GFIN, build_tree_ensemble, tree_ensemble_proba,
    val_split, tune_fusion_alpha, find_best_threshold, compute_all_metrics,
    SEED, FEATURE_COLS
)

def run_experiment(name, use_interactions=True, num_gfin=2, use_trees=True):
    print(f"\n--- Running: {name} ---")
    DATA_PATH = "pima-indians-diabetes.csv"
    X_raw, y = load_pima(DATA_PATH)
    
    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(X_raw)
    X_sm, y_sm = borderline_smote(X_sc, y, seed=SEED)
    
    rfe_idx = rfe_select(X_sm, y_sm, n_features=4, seed=SEED)
    pairs = list(combinations(rfe_idx, 2)) if use_interactions else []
    
    SPLIT_SEED = 2
    X_tr_sm, X_te_sm, y_tr, y_te = train_test_split(
        X_sm, y_sm, test_size=0.20, random_state=SPLIT_SEED, stratify=y_sm)
        
    sc2 = MinMaxScaler()
    X_tr_sc = sc2.fit_transform(X_tr_sm)
    X_te_sc = sc2.transform(X_te_sm)
    
    X_tr = apply_interactions(X_tr_sc, pairs)
    X_te = apply_interactions(X_te_sc, pairs)
    D = X_tr.shape[1]
    
    X_tn, y_tn, X_val, y_val = val_split(X_tr, y_tr, frac=0.15, seed=SPLIT_SEED + 50)
    
    p_gfin_te = np.zeros(len(y_te))
    p_gfin_val = np.zeros(len(y_val))
    
    if num_gfin >= 1:
        gfin1 = GFIN(D, H1=256, H2=128, H3=64, lr=2e-3, epochs=300, batch=32, drop1=0.25, drop2=0.20, drop3=0.15, l2=1e-5, patience=35, label_smooth=0.05, seed=SPLIT_SEED)
        gfin1.fit(X_tn, y_tn, X_val, y_val)
        p_gfin1_te = gfin1.predict_proba(X_te)
        p_gfin1_val = gfin1.predict_proba(X_val)
        
        p_gfin_te = p_gfin1_te
        p_gfin_val = p_gfin1_val
        
        if num_gfin == 2:
            gfin2 = GFIN(D, H1=256, H2=128, H3=64, lr=1.5e-3, epochs=300, batch=48, drop1=0.30, drop2=0.20, drop3=0.15, l2=2e-5, patience=35, label_smooth=0.03, seed=SPLIT_SEED + 99)
            gfin2.fit(X_tn, y_tn, X_val, y_val)
            p_gfin2_te = gfin2.predict_proba(X_te)
            p_gfin2_val = gfin2.predict_proba(X_val)
            
            p_gfin_te = 0.55 * p_gfin1_te + 0.45 * p_gfin2_te
            p_gfin_val = 0.55 * p_gfin1_val + 0.45 * p_gfin2_val

    p_trees_te = np.zeros(len(y_te))
    p_trees_val = np.zeros(len(y_val))
    
    if use_trees:
        trees = build_tree_ensemble(SPLIT_SEED)
        for clf in trees:
            clf.fit(X_tr, y_tr)
        p_trees_te = tree_ensemble_proba(trees, X_te)
        p_trees_val = tree_ensemble_proba(trees, X_val)
        
    if use_trees and num_gfin > 0:
        alpha = tune_fusion_alpha(y_val, p_trees_val, p_gfin_val)
    elif use_trees and num_gfin == 0:
        alpha = 1.0
    elif not use_trees and num_gfin > 0:
        alpha = 0.0
    else:
        raise ValueError("Must use either trees or gfin")
        
    p_fused_te = alpha * p_trees_te + (1 - alpha) * p_gfin_te
    p_fused_val = alpha * p_trees_val + (1 - alpha) * p_gfin_val
    
    threshold = find_best_threshold(y_val, p_fused_val)
    results = compute_all_metrics(y_te, p_fused_te, threshold)
    results['alpha'] = alpha
    results['threshold'] = threshold
    print(f"Accuracy: {results['acc']*100:.2f}%, F1: {results['f1']*100:.2f}%, AUC: {results['auc']:.4f}")
    return results

def plot_comparison(res1, res2, label1, label2, title, filename):
    metrics = ['acc', 'prec', 'rec', 'f1']
    vals1 = [res1[m]*100 for m in metrics]
    vals2 = [res2[m]*100 for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, vals1, width, label=label1, color='#4e79a7')
    rects2 = ax.bar(x + width/2, vals2, width, label=label2, color='#e15759')
    
    ax.set_ylabel('Score (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.set_ylim(60, 100)
    ax.legend(loc='lower right')
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}.png", dpi=200)
    plt.close()

if __name__ == '__main__':
    Path("outputs").mkdir(exist_ok=True)
    
    # Run experiments
    res_full = run_experiment("Full Model", use_interactions=True, num_gfin=2, use_trees=True)
    res_no_int = run_experiment("No Interactions", use_interactions=False, num_gfin=2, use_trees=True)
    res_1gfin = run_experiment("1 GFIN", use_interactions=True, num_gfin=1, use_trees=True)
    res_no_ens = run_experiment("No Tree Ensemble (GFIN only)", use_interactions=True, num_gfin=2, use_trees=False)
    res_no_gfin = run_experiment("No GFIN (Tree Ensemble only)", use_interactions=True, num_gfin=0, use_trees=True)
    
    # 1. Interactions vs No Interactions
    plot_comparison(res_no_int, res_full, "Without Interactions", "With Interactions", 
                   "Impact of RFE Interaction Terms", "comparison_interactions")
                   
    # 2. 1 GFIN vs 2 GFINs
    plot_comparison(res_1gfin, res_full, "1 GFIN", "2 GFINs", 
                   "Impact of Dual GFIN Ensemble", "comparison_gfin_count")
                   
    # 3. With vs Without Tree Ensemble
    plot_comparison(res_no_ens, res_full, "Without Tree Ensemble", "With Tree Ensemble", 
                   "Impact of Output Fusion (Tree Ensemble)", "comparison_tree_ensemble")
                   
    # 4. No GFIN vs Full Model
    plot_comparison(res_no_gfin, res_full, "Without GFIN (Only Trees)", "With GFIN + Trees", 
                   "Value added by GFIN Blocks", "comparison_no_gfin")
                   
    print("\nAll comparisons generated in outputs/ folder.")
