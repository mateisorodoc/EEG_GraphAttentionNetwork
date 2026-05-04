"""
═══════════════════════════════════════════════════════════════════════════════
DeepGAT: Optuna Bayesian Hyperparameter Optimization
═══════════════════════════════════════════════════════════════════════════════
Optimizes GAT hyperparameters for both:
  - Tier 2 (trial-aware within-subject) on DEAP
  - Tier 3 (LOSO cross-subject) on DEAP

Objective: Mean of Tier 1 + Tier 2 + Tier 3 F1 scores (equal weight)

Search space:
  - Backbone: 2-4 layers, 32-128 dim
  - Heads: 2-8
  - Attention dropout: 0.0-0.3
  - Head dense: 64-256
  - Head dropout: 0.1-0.5
  - LR: 1e-4 to 5e-3 (log)
  - Batch size: 128, 256, 512, 1024
  - Weight decay: 1e-5 to 1e-2 (log)
  - Label smoothing: 0.0-0.15
  - Input noise: 0.0-0.2
  - Epochs: 100-300
  - Patience: 10-30
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import optuna
from optuna.trial import TrialState

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

import torch

warnings.filterwarnings('ignore')

# Import from our evaluation module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation_gat import (
    DeepGAT_DEAP, EEGDataset, load_deap_data, train_gat_deap,
    device, SEED, DEAP_N_CH, SUBJECTS, CHANNEL_NAMES_DEAP
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
N_OPTUNA_TRIALS = 150  # Number of Optuna trials
TIER2_FOLDS = 3  # Fewer folds during tuning for speed
TIER3_SUBJECTS = 3  # Subset of subjects for LOSO during tuning
TUNING_SUBJECTS = 5  # Subset of subjects for Tier 2 during tuning
TUNING_MAX_EPOCHS = 80  # Cap epochs during tuning for speed
TUNING_PATIENCE = 10  # Fixed patience during tuning


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def objective(trial, deap_data, n_feats):
    """Optuna objective: mean of Tier 1 + Tier 2 + Tier 3 F1."""
    
    # ─── Sample hyperparameters ───────────────────────────────────────────────
    n_layers = trial.suggest_int('n_layers', 2, 4)
    backbone_dim = trial.suggest_categorical('backbone_dim', [32, 48, 64, 96, 128])
    backbone_dims = [backbone_dim] * n_layers
    
    hparams = {
        'backbone_dims': backbone_dims,
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'attn_dropout': trial.suggest_float('attn_dropout', 0.0, 0.3),
        'head_dense': trial.suggest_categorical('head_dense', [64, 128, 256]),
        'head_dropout': trial.suggest_float('head_dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.15),
        'input_noise_std': trial.suggest_float('input_noise_std', 0.0, 0.2),
        'epochs': TUNING_MAX_EPOCHS,  # Fixed during tuning for speed
        'patience': TUNING_PATIENCE,  # Fixed during tuning for speed
    }
    
    # Derive n_ch from actual data
    n_ch = list(deap_data.values())[0]['X'].shape[1]
    
    # ─── Tier 1: Leaky (subset of subjects, 2 folds) ─────────────────────────
    from sklearn.model_selection import StratifiedKFold
    tier1_scores = []
    subject_keys = list(deap_data.keys())[:TUNING_SUBJECTS]
    
    for sub_id in subject_keys:
        sub_data = deap_data[sub_id]
        X = sub_data['X']
        Y = np.stack([sub_data['Y_aro'], sub_data['Y_val']], axis=1).astype(np.float32)
        strat = sub_data['Y_aro'] * 2 + sub_data['Y_val']
        
        if len(np.unique(strat)) < 2:
            continue
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, strat)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_ch*n_feats)).reshape(-1, n_ch, n_feats)
            X_te_s = scaler.transform(X_te.reshape(-1, n_ch*n_feats)).reshape(-1, n_ch, n_feats)
            
            preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED+fold)
            f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
                  f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
            tier1_scores.append(f1)
    
    tier1_mean = np.mean(tier1_scores) if tier1_scores else 0.0
    
    # Early pruning after Tier 1
    trial.report(tier1_mean, 0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # ─── Tier 2: Trial-aware (subset of subjects) ────────────────────────────
    tier2_scores = []
    
    for sub_id in subject_keys:
        sub_data = deap_data[sub_id]
        X = sub_data['X']
        Y = np.stack([sub_data['Y_aro'], sub_data['Y_val']], axis=1).astype(np.float32)
        trials = sub_data['trials']
        strat = sub_data['Y_aro'] * 2 + sub_data['Y_val']
        
        n_unique_trials = len(np.unique(trials))
        actual_folds = min(TIER2_FOLDS, n_unique_trials)
        if len(np.unique(strat)) < 2 or actual_folds < 2:
            continue
        
        sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
        
        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, strat, groups=trials)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_ch*n_feats)).reshape(-1, n_ch, n_feats)
            X_te_s = scaler.transform(X_te.reshape(-1, n_ch*n_feats)).reshape(-1, n_ch, n_feats)
            
            preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED+fold)
            f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
                  f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
            tier2_scores.append(f1)
        
        # Early pruning: report intermediate result after each subject
        if tier2_scores:
            trial.report(np.mean(tier2_scores), len(tier2_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    tier2_mean = np.mean(tier2_scores) if tier2_scores else 0.0
    
    # ─── Tier 3: LOSO (subset of subjects) ───────────────────────────────────
    all_X = np.concatenate([d['X'] for d in deap_data.values()])
    all_Y = np.stack([
        np.concatenate([d['Y_aro'] for d in deap_data.values()]),
        np.concatenate([d['Y_val'] for d in deap_data.values()])
    ], axis=1).astype(np.float32)
    all_subs = np.concatenate([
        np.full(len(d['X']), int(sid)) for sid, d in deap_data.items()
    ])
    
    tier3_scores = []
    test_subjects = list(deap_data.keys())[:TIER3_SUBJECTS]
    
    for test_sub in test_subjects:
        test_mask = all_subs == int(test_sub)
        train_mask = ~test_mask
        X_tr, X_te = all_X[train_mask], all_X[test_mask]
        Y_tr, Y_te = all_Y[train_mask], all_Y[test_mask]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_ch*n_feats)).reshape(-1, n_ch, n_feats)
        X_te_s = scaler.transform(X_te.reshape(-1, n_ch*n_feats)).reshape(-1, n_ch, n_feats)
        
        preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        tier3_scores.append(f1)
    
    tier3_mean = np.mean(tier3_scores) if tier3_scores else 0.0
    
    # Combined objective: equal weight across all 3 tiers
    combined = (tier1_mean + tier2_mean + tier3_mean) / 3
    
    # Log intermediate info
    trial.set_user_attr('tier1_f1', tier1_mean)
    trial.set_user_attr('tier2_f1', tier2_mean)
    trial.set_user_attr('tier3_f1', tier3_mean)
    
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument('--tier2-subjects', type=int, default=TUNING_SUBJECTS)
    parser.add_argument('--tier3-subjects', type=int, default=TIER3_SUBJECTS)
    parser.add_argument('--tier2-folds', type=int, default=TIER2_FOLDS)
    parser.add_argument('--resume', action='store_true', help='Resume from existing study')
    args = parser.parse_args()
    
    TUNING_SUBJECTS = args.tier2_subjects
    TIER3_SUBJECTS = args.tier3_subjects
    TIER2_FOLDS = args.tier2_folds
    
    print(f'{"="*70}')
    print('DeepGAT Optuna Hyperparameter Optimization')
    print(f'{"="*70}')
    print(f'Device: {device}')
    print(f'Trials: {args.n_trials}')
    print(f'Tier 2: {TUNING_SUBJECTS} subjects x {TIER2_FOLDS} folds')
    print(f'Tier 3: {TIER3_SUBJECTS} LOSO iterations')
    print()
    
    # Load data
    print('Loading DEAP features...')
    deap_data, feat_source = load_deap_data()
    n_feats = list(deap_data.values())[0]['X'].shape[2]
    print(f'  Source: {feat_source}')
    print(f'  Subjects: {len(deap_data)} | Features: {n_feats}/channel')
    print()
    
    # Create/load study
    study_path = os.path.join(OUT_DIR, 'optuna_study.db')
    storage = f'sqlite:///{study_path}'
    
    study = optuna.create_study(
        study_name='deepgat_eeg_32ch',
        direction='maximize',
        storage=storage,
        load_if_exists=args.resume,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
    )
    
    print(f'Study: {study.study_name}')
    if args.resume:
        print(f'  Existing trials: {len(study.trials)}')
    print(f'  Storage: {study_path}')
    print()
    
    t0 = time.time()
    
    study.optimize(
        lambda trial: objective(trial, deap_data, n_feats),
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    elapsed = time.time() - t0
    
    # ─── Results ──────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('OPTIMIZATION COMPLETE')
    print(f'{"="*70}')
    print(f'Time: {elapsed/3600:.1f}h ({elapsed:.0f}s)')
    print(f'Trials: {len(study.trials)} total, '
          f'{len([t for t in study.trials if t.state == TrialState.COMPLETE])} complete, '
          f'{len([t for t in study.trials if t.state == TrialState.PRUNED])} pruned')
    
    best = study.best_trial
    print(f'\nBest trial #{best.number}:')
    print(f'  Combined F1: {best.value:.4f}')
    print(f'  Tier 1 F1: {best.user_attrs.get("tier1_f1", "N/A")}')
    print(f'  Tier 2 F1: {best.user_attrs.get("tier2_f1", "N/A")}')
    print(f'  Tier 3 F1: {best.user_attrs.get("tier3_f1", "N/A")}')
    print(f'  Params:')
    for k, v in best.params.items():
        print(f'    {k}: {v}')
    
    # Convert to hparams dict
    best_hparams = {
        'backbone_dims': [best.params['backbone_dim']] * best.params['n_layers'],
        'num_heads': best.params['num_heads'],
        'attn_dropout': best.params['attn_dropout'],
        'head_dense': best.params['head_dense'],
        'head_dropout': best.params['head_dropout'],
        'lr': best.params['lr'],
        'weight_decay': best.params['weight_decay'],
        'batch_size': best.params['batch_size'],
        'label_smoothing': best.params['label_smoothing'],
        'input_noise_std': best.params['input_noise_std'],
        'epochs': 200,  # Use full training epochs for final run
        'patience': 20,  # Use full patience for final run
    }
    
    # Save best hyperparameters
    hparams_path = os.path.join(OUT_DIR, 'best_hparams.json')
    with open(hparams_path, 'w') as f:
        json.dump(best_hparams, f, indent=2)
    print(f'\nSaved best hyperparameters to: {hparams_path}')
    
    # Save top-10 trials summary
    top_trials = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE],
        key=lambda t: t.value, reverse=True
    )[:10]
    
    summary = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'elapsed_seconds': elapsed,
        'best_value': best.value,
        'best_hparams': best_hparams,
        'top_10': [{
            'number': t.number,
            'value': t.value,
            'tier1_f1': t.user_attrs.get('tier1_f1'),
            'tier2_f1': t.user_attrs.get('tier2_f1'),
            'tier3_f1': t.user_attrs.get('tier3_f1'),
            'params': t.params,
        } for t in top_trials]
    }
    
    summary_path = os.path.join(OUT_DIR, 'optuna_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved optimization summary to: {summary_path}')
    
    print(f'\n{"="*70}')
    print(f'Next: Run full evaluation with best hparams:')
    print(f'  python run_evaluation_gat.py')
    print(f'{"="*70}')
