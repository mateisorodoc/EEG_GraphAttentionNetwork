"""
═══════════════════════════════════════════════════════════════════════════════
TRANSPARENT THREE-TIER EVALUATION: EEG Emotion Classification
═══════════════════════════════════════════════════════════════════════════════
TIER 1: With trial leakage (random split)
TIER 2: Trial-aware CV (GroupKFold by trial)
TIER 3: Cross-subject LOSO (train 19, test 1)

Models: SVM (RBF), MLP (128→64), DeepGAT (3×GATLayer[64])
Features: 10/channel (5 BP + 5 DE), 16 channels
Folds: 10
GAT: 200 epochs, early stopping (patience=15)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, pickle, warnings, glob, math, time, json, sys
# Fix Windows cp1252 encoding issues with unicode chars
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from scipy import stats as scipy_stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut
)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEAP_FEAT_V2 = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'output', 'features_v2')
OPENBCI_BASE = os.path.join(PROJECT_ROOT, '..', 'data', 'recordings_clean')
OPENBCI_FEAT = os.path.join(OPENBCI_BASE, 'data_extracted_v2')
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
N_CH = 16
N_FEATS = 10
N_FOLDS = 10
GAT_EPOCHS = 200
GAT_PATIENCE = 15
GAT_LR = 5e-4
GAT_BATCH = 512
MAX_GAT_TRAIN = 5000  # Cap GAT training samples per fold (same as SVM/MLP)

BAND_LABELS = ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma']
FEAT_LABELS = [f'BP_{b}' for b in BAND_LABELS] + [f'DE_{b}' for b in BAND_LABELS]
CHANNEL_NAMES_DEAP = ['Fp1','F3','F7','C3','T7','P3','P7','O1',
                      'Fp2','F4','F8','C4','T8','P4','P8','O2']
LEFT_CH  = list(range(0, 8))
RIGHT_CH = list(range(8, 16))
FRONTAL_CH = [0, 1, 2, 8, 9, 10]
F3_IDX, F4_IDX = 1, 9
ALPHA_BAND_IDX = 1

SUBJECTS = [f'{i:02d}' for i in range(1, 21)]
SVM_KWARGS = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
MLP_KWARGS = {'hidden_layer_sizes': (128, 64), 'max_iter': 300,
              'early_stopping': True, 'random_state': SEED}
MAX_SKLEARN_TRAIN = 5000  # Cap SVM/MLP training samples (RBF SVM is O(n^2))

print(f'{"═"*70}')
print(f'TRANSPARENT THREE-TIER EVALUATION')
print(f'{"═"*70}')
print(f'Device: {device}')
print(f'Folds: {N_FOLDS} | GAT epochs: {GAT_EPOCHS} | Patience: {GAT_PATIENCE}')
print(f'Features: {N_FEATS}/channel × {N_CH} channels = {N_CH * N_FEATS} total')
print(f'Output: {OUT_DIR}')
print()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4,
                 attn_dropout=0.05, residual=True):
        super().__init__()
        self.H, self.d = num_heads, out_features
        self.residual = residual
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        # Efficient additive attention: split into source and target
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.bn = nn.BatchNorm1d(out_features)
        if residual:
            self.res_proj = (nn.Linear(in_features, out_features, bias=False)
                            if in_features != out_features else nn.Identity())

    def forward(self, x, return_attn=False):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.H, self.d)  # B, N, H, d
        # Efficient: compute e_src and e_dst separately, then broadcast-add
        e_src = (h * self.a_src).sum(-1)  # B, N, H
        e_dst = (h * self.a_dst).sum(-1)  # B, N, H
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # B, N, N, H
        alpha = self.attn_drop(F.softmax(e, dim=2))  # B, N, N, H
        # Weighted sum: alpha[b,i,j,h] * h[b,j,h,d] -> out[b,i,h,d]
        out = torch.einsum('bqkh, bkhd -> bqhd', alpha, h).mean(dim=2)  # B, N, d
        out = self.bn(out.reshape(B * N, self.d)).reshape(B, N, self.d)
        out = F.elu(out)
        if self.residual:
            out = out + self.res_proj(x)
        if return_attn:
            return out, alpha.permute(0, 3, 1, 2)  # B, H, N, N
        return out


class TaskHead(nn.Module):
    def __init__(self, in_dim, dense=128, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, dense), nn.LayerNorm(dense),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dense, 1))
    def forward(self, x):
        return self.head(x.mean(dim=1))


class DeepGAT_DEAP(nn.Module):
    def __init__(self, n_ch=16, in_feats=10, backbone_dims=[64,64,64],
                 dense=128, num_heads=4, attn_dropout=0.05, head_dropout=0.3):
        super().__init__()
        d0 = backbone_dims[0]
        self.input_proj = nn.Linear(in_feats, d0)
        self.ch_embed = nn.Parameter(torch.randn(1, n_ch, d0) * 0.02)
        dims = [d0] + backbone_dims
        self.backbone = nn.ModuleList([
            GATLayer(dims[i], dims[i+1], num_heads, attn_dropout)
            for i in range(len(backbone_dims))
        ])
        self.head_aro = TaskHead(backbone_dims[-1], dense, head_dropout)
        self.head_val = TaskHead(backbone_dims[-1], dense, head_dropout)

    def forward(self, x, return_attn=False):
        x = F.gelu(self.input_proj(x)) + self.ch_embed
        attns = []
        for layer in self.backbone:
            if return_attn:
                x, aw = layer(x, return_attn=True)
                attns.append(aw)
            else:
                x = layer(x)
        out = torch.cat([self.head_aro(x), self.head_val(x)], dim=1)
        if return_attn:
            return out, attns
        return out


class DeepGAT_MultiClass(nn.Module):
    def __init__(self, n_ch=16, in_feats=10, n_classes=4,
                 backbone_dims=[64,64,64], dense=128, num_heads=4,
                 attn_dropout=0.05, head_dropout=0.3):
        super().__init__()
        d0 = backbone_dims[0]
        self.input_proj = nn.Linear(in_feats, d0)
        self.ch_embed = nn.Parameter(torch.randn(1, n_ch, d0) * 0.02)
        dims = [d0] + backbone_dims
        self.backbone = nn.ModuleList([
            GATLayer(dims[i], dims[i+1], num_heads, attn_dropout)
            for i in range(len(backbone_dims))
        ])
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dims[-1], dense), nn.LayerNorm(dense),
            nn.GELU(), nn.Dropout(head_dropout), nn.Linear(dense, n_classes))

    def forward(self, x, return_attn=False):
        x = F.gelu(self.input_proj(x)) + self.ch_embed
        attns = []
        for layer in self.backbone:
            if return_attn:
                x, aw = layer(x, return_attn=True)
                attns.append(aw)
            else:
                x = layer(x)
        out = self.classifier(x.mean(dim=1))
        if return_attn:
            return out, attns
        return out


class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float() if Y.ndim > 1 else torch.from_numpy(Y).long()
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_gat_deap(X_tr, Y_tr, X_te, Y_te, epochs=GAT_EPOCHS, lr=GAT_LR,
                   batch_size=GAT_BATCH, patience=GAT_PATIENCE, seed=42):
    torch.manual_seed(seed)
    # Subsample training data if too large
    if len(X_tr) > MAX_GAT_TRAIN:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_tr), MAX_GAT_TRAIN, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
    model = DeepGAT_DEAP().to(device)
    pw_aro = torch.tensor([(Y_tr[:,0]==0).sum() / max((Y_tr[:,0]==1).sum(), 1)],
                          dtype=torch.float32).to(device)
    pw_val = torch.tensor([(Y_tr[:,1]==0).sum() / max((Y_tr[:,1]==1).sum(), 1)],
                          dtype=torch.float32).to(device)
    loss_aro = nn.BCEWithLogitsLoss(pos_weight=pw_aro)
    loss_val = nn.BCEWithLogitsLoss(pos_weight=pw_val)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_dl = DataLoader(EEGDataset(X_tr, Y_tr), batch_size=batch_size,
                          shuffle=True, drop_last=len(X_tr) > batch_size)
    val_dl = DataLoader(EEGDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    best_loss, wait, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = (loss_aro(logits[:,0], Yb[:,0]) + loss_val(logits[:,1], Yb[:,1])) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_loss_sum, v_n = 0.0, 0
        with torch.no_grad():
            for Xb, Yb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                logits = model(Xb)
                v_loss_sum += (loss_aro(logits[:,0], Yb[:,0]) + loss_val(logits[:,1], Yb[:,1])).item() * len(Xb)
                v_n += len(Xb)
        v_loss = v_loss_sum / (2 * v_n)

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in val_dl:
            logits = model(Xb.to(device))
            all_preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
    return np.concatenate(all_preds)


def train_gat_multiclass(X_tr, Y_tr, X_te, Y_te, n_classes=4, epochs=GAT_EPOCHS,
                         lr=GAT_LR, batch_size=GAT_BATCH, patience=GAT_PATIENCE, seed=42):
    torch.manual_seed(seed)
    # Subsample training data if too large
    if len(X_tr) > MAX_GAT_TRAIN:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_tr), MAX_GAT_TRAIN, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
    model = DeepGAT_MultiClass(n_classes=n_classes).to(device)
    counts = np.bincount(Y_tr, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * counts + 1e-6),
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_dl = DataLoader(EEGDataset(X_tr, Y_tr), batch_size=batch_size,
                          shuffle=True, drop_last=len(X_tr) > batch_size)
    val_dl = DataLoader(EEGDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    best_loss, wait, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), Yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_loss_sum, v_n = 0.0, 0
        with torch.no_grad():
            for Xb, Yb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                v_loss_sum += criterion(model(Xb), Yb).item() * len(Xb)
                v_n += len(Xb)
        v_loss = v_loss_sum / v_n

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in val_dl:
            all_preds.append(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


def eval_sklearn_deap(clf_class, clf_kwargs, X_tr, Y_tr, X_te, Y_te):
    Xtr_flat = X_tr.reshape(len(X_tr), -1)
    Xte_flat = X_te.reshape(len(X_te), -1)
    # Subsample training if too large
    if len(Xtr_flat) > MAX_SKLEARN_TRAIN:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(Xtr_flat), MAX_SKLEARN_TRAIN, replace=False)
        Xtr_flat, Y_tr = Xtr_flat[idx], Y_tr[idx]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_flat)
    Xte_s = scaler.transform(Xte_flat)
    clf_aro = clf_class(**clf_kwargs); clf_aro.fit(Xtr_s, Y_tr[:, 0])
    clf_val = clf_class(**clf_kwargs); clf_val.fit(Xtr_s, Y_tr[:, 1])
    return np.stack([clf_aro.predict(Xte_s), clf_val.predict(Xte_s)], axis=1)


def eval_sklearn_multiclass(clf_class, clf_kwargs, X_tr, Y_tr, X_te, Y_te):
    Xtr_flat = X_tr.reshape(len(X_tr), -1)
    Xte_flat = X_te.reshape(len(X_te), -1)
    if len(Xtr_flat) > MAX_SKLEARN_TRAIN:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(Xtr_flat), MAX_SKLEARN_TRAIN, replace=False)
        Xtr_flat, Y_tr = Xtr_flat[idx], Y_tr[idx]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_flat)
    Xte_s = scaler.transform(Xte_flat)
    clf = clf_class(**clf_kwargs); clf.fit(Xtr_s, Y_tr)
    return clf.predict(Xte_s)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print('Loading DEAP features (v2 cache)...')
deap_data = {}
for sub in SUBJECTS:
    fp = os.path.join(DEAP_FEAT_V2, f's{sub}.npz')
    if not os.path.exists(fp):
        print(f'  ⚠ Missing: {fp}')
        continue
    d = np.load(fp)
    X = d['features'].reshape(-1, N_CH, N_FEATS)
    labels = d['labels']
    trials = d['trials']
    Y_aro = (labels[:, 0] > 5.0).astype(np.int32)
    Y_val = (labels[:, 1] > 5.0).astype(np.int32)
    deap_data[sub] = {'X': X, 'Y_aro': Y_aro, 'Y_val': Y_val, 'trials': trials}
print(f'  Loaded: {len(deap_data)} subjects, {sum(len(d["X"]) for d in deap_data.values()):,} total windows')

print('Loading OpenBCI features (v2 cache)...')
LABEL_MAP_OPENBCI = {'calm': 0, 'happy': 1, 'sad': 2, 'stressed': 3}
CLASS_NAMES = ['calm', 'happy', 'sad', 'stressed']
openbci_X, openbci_Y, openbci_groups = [], [], []
feat_files = sorted(glob.glob(os.path.join(OPENBCI_FEAT, '*_clean_trial_*.npy')))
for fp in feat_files:
    fname = os.path.basename(fp)
    cat = fname.split('_clean_trial_')[0]
    if cat not in LABEL_MAP_OPENBCI:
        continue
    arr = np.load(fp, allow_pickle=True)
    for row in arr:
        openbci_X.append(row[0])
        openbci_Y.append(LABEL_MAP_OPENBCI[cat])
        openbci_groups.append(fname)
openbci_X = np.array(openbci_X, dtype=np.float32)
openbci_Y = np.array(openbci_Y, dtype=np.int64)
openbci_groups = np.array(openbci_groups)
print(f'  Windows: {len(openbci_X):,} | Trials: {len(np.unique(openbci_groups))} | Classes: {Counter(openbci_Y.tolist())}')
print(flush=True)
sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: WITH TRIAL LEAKAGE (Per-subject, random split ignoring trials)
# ═══════════════════════════════════════════════════════════════════════════════

print('═' * 70)
print('TIER 1: WITH TRIAL LEAKAGE (per-subject, random split ignoring trials)')
print('═' * 70)
t0 = time.time()

tier1_results = {'SVM': [], 'MLP': [], 'DeepGAT': []}

for sub_id, sub_data in tqdm(list(deap_data.items()), desc='Tier1 subjects'):
    X = sub_data['X']
    Y = np.stack([sub_data['Y_aro'], sub_data['Y_val']], axis=1).astype(np.float32)
    strat = sub_data['Y_aro'] * 2 + sub_data['Y_val']

    n_unique_strat = len(np.unique(strat))
    if n_unique_strat < 2:
        continue

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    sub_scores = {'SVM': [], 'MLP': [], 'DeepGAT': []}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, strat)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]

        t_step = time.time()
        preds = eval_sklearn_deap(SVC, SVM_KWARGS, X_tr, Y_tr, X_te, Y_te)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        sub_scores['SVM'].append(f1)
        if fold == 0 and sub_id == list(deap_data.keys())[0]:
            print(f'    [timing] SVM fold 1: {time.time()-t_step:.1f}s', flush=True)

        t_step = time.time()
        preds = eval_sklearn_deap(MLPClassifier, MLP_KWARGS, X_tr, Y_tr, X_te, Y_te)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        sub_scores['MLP'].append(f1)
        if fold == 0 and sub_id == list(deap_data.keys())[0]:
            print(f'    [timing] MLP fold 1: {time.time()-t_step:.1f}s', flush=True)

        t_step = time.time()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
        X_te_s = scaler.transform(X_te.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
        preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, seed=SEED+fold)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        sub_scores['DeepGAT'].append(f1)
        if fold == 0 and sub_id == list(deap_data.keys())[0]:
            print(f'    [timing] GAT fold 1: {time.time()-t_step:.1f}s', flush=True)

    for m in sub_scores:
        tier1_results[m].append(np.mean(sub_scores[m]))

print(f'\n── TIER 1 SUMMARY (WITH LEAKAGE) ── [{time.time()-t0:.0f}s]')
for m, s in tier1_results.items():
    print(f'  {m:8s}: F1 = {np.mean(s):.4f} ± {np.std(s):.4f}  '
          f'(range: {np.min(s):.3f}–{np.max(s):.3f})')
print()


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: TRIAL-AWARE WITHIN-SUBJECT (GroupKFold)
# ═══════════════════════════════════════════════════════════════════════════════

print('═' * 70)
print('TIER 2: TRIAL-AWARE WITHIN-SUBJECT (StratifiedGroupKFold by trial)')
print('═' * 70)
t0 = time.time()

tier2_results = {'SVM': [], 'MLP': [], 'DeepGAT': []}

for sub_id, sub_data in tqdm(list(deap_data.items()), desc='Tier2 subjects'):
    X = sub_data['X']
    Y = np.stack([sub_data['Y_aro'], sub_data['Y_val']], axis=1).astype(np.float32)
    trials = sub_data['trials']
    strat = sub_data['Y_aro'] * 2 + sub_data['Y_val']

    n_unique_strat = len(np.unique(strat))
    n_unique_trials = len(np.unique(trials))
    actual_folds = min(N_FOLDS, n_unique_trials)
    if n_unique_strat < 2 or actual_folds < 2:
        continue

    sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
    sub_scores = {'SVM': [], 'MLP': [], 'DeepGAT': []}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, strat, groups=trials)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]

        preds = eval_sklearn_deap(SVC, SVM_KWARGS, X_tr, Y_tr, X_te, Y_te)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        sub_scores['SVM'].append(f1)

        preds = eval_sklearn_deap(MLPClassifier, MLP_KWARGS, X_tr, Y_tr, X_te, Y_te)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        sub_scores['MLP'].append(f1)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
        X_te_s = scaler.transform(X_te.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
        preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, seed=SEED+fold)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        sub_scores['DeepGAT'].append(f1)

    for m in sub_scores:
        tier2_results[m].append(np.mean(sub_scores[m]))

print(f'\n── TIER 2 SUMMARY (TRIAL-AWARE) ── [{time.time()-t0:.0f}s]')
for m, s in tier2_results.items():
    print(f'  {m:8s}: F1 = {np.mean(s):.4f} ± {np.std(s):.4f}  '
          f'(range: {np.min(s):.3f}–{np.max(s):.3f})')
print()


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: CROSS-SUBJECT LOSO
# ═══════════════════════════════════════════════════════════════════════════════

print('═' * 70)
print('TIER 3: CROSS-SUBJECT LOSO (train 19, test 1)')
print('═' * 70)
t0 = time.time()

all_X_loso = np.concatenate([d['X'] for d in deap_data.values()])
all_Y_loso = np.stack([
    np.concatenate([d['Y_aro'] for d in deap_data.values()]),
    np.concatenate([d['Y_val'] for d in deap_data.values()])
], axis=1).astype(np.float32)
all_subs = np.concatenate([
    np.full(len(d['X']), int(sid)) for sid, d in deap_data.items()
])

# For SVM/MLP: subsampling is handled inside eval_sklearn_deap()

tier3_results = {'SVM': [], 'MLP': [], 'DeepGAT': []}

for test_sub in tqdm(sorted(deap_data.keys()), desc='LOSO'):
    test_mask = all_subs == int(test_sub)
    train_mask = ~test_mask
    X_tr_full, X_te = all_X_loso[train_mask], all_X_loso[test_mask]
    Y_tr_full, Y_te = all_Y_loso[train_mask], all_Y_loso[test_mask]

    preds = eval_sklearn_deap(SVC, SVM_KWARGS, X_tr_full, Y_tr_full, X_te, Y_te)
    f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
          f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
    tier3_results['SVM'].append(f1)

    preds = eval_sklearn_deap(MLPClassifier, MLP_KWARGS, X_tr_full, Y_tr_full, X_te, Y_te)
    f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
          f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
    tier3_results['MLP'].append(f1)

    # GAT uses full training data (GPU handles it fine)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_full.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
    X_te_s = scaler.transform(X_te.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
    preds = train_gat_deap(X_tr_s, Y_tr_full, X_te_s, Y_te, seed=SEED)
    f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
          f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
    tier3_results['DeepGAT'].append(f1)

    print(f'  s{test_sub}: SVM={tier3_results["SVM"][-1]:.4f} | '
          f'MLP={tier3_results["MLP"][-1]:.4f} | GAT={tier3_results["DeepGAT"][-1]:.4f}')

print(f'\n── TIER 3 SUMMARY (LOSO) ── [{time.time()-t0:.0f}s]')
for m, s in tier3_results.items():
    print(f'  {m:8s}: F1 = {np.mean(s):.4f} ± {np.std(s):.4f}  '
          f'(range: {np.min(s):.3f}–{np.max(s):.3f})')
print()


# ═══════════════════════════════════════════════════════════════════════════════
# OpenBCI: LEAKY vs TRIAL-AWARE (4-class)
# ═══════════════════════════════════════════════════════════════════════════════

print('═' * 70)
print('OpenBCI 4-CLASS: LEAKY vs TRIAL-AWARE')
print('═' * 70)
t0 = time.time()

# Leaky
print('\n── Tier 1: WITH LEAKAGE ──')
skf_ob = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ob_tier1 = {'SVM': [], 'MLP': [], 'DeepGAT': []}

for fold, (tr_idx, te_idx) in enumerate(skf_ob.split(openbci_X, openbci_Y)):
    X_tr, X_te = openbci_X[tr_idx], openbci_X[te_idx]
    Y_tr, Y_te = openbci_Y[tr_idx], openbci_Y[te_idx]

    preds = eval_sklearn_multiclass(SVC, SVM_KWARGS, X_tr, Y_tr, X_te, Y_te)
    ob_tier1['SVM'].append(f1_score(Y_te, preds, average='macro'))

    preds = eval_sklearn_multiclass(MLPClassifier, MLP_KWARGS, X_tr, Y_tr, X_te, Y_te)
    ob_tier1['MLP'].append(f1_score(Y_te, preds, average='macro'))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
    X_te_s = scaler.transform(X_te.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
    preds = train_gat_multiclass(X_tr_s, Y_tr, X_te_s, Y_te, seed=SEED+fold)
    ob_tier1['DeepGAT'].append(f1_score(Y_te, preds, average='macro'))

    print(f'  Fold {fold+1:2d}: SVM={ob_tier1["SVM"][-1]:.4f} | '
          f'MLP={ob_tier1["MLP"][-1]:.4f} | GAT={ob_tier1["DeepGAT"][-1]:.4f}')

# Trial-aware
print('\n── Tier 2: TRIAL-AWARE ──')
sgkf_ob = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ob_tier2 = {'SVM': [], 'MLP': [], 'DeepGAT': []}

for fold, (tr_idx, te_idx) in enumerate(
    sgkf_ob.split(openbci_X, openbci_Y, groups=openbci_groups)):
    X_tr, X_te = openbci_X[tr_idx], openbci_X[te_idx]
    Y_tr, Y_te = openbci_Y[tr_idx], openbci_Y[te_idx]

    preds = eval_sklearn_multiclass(SVC, SVM_KWARGS, X_tr, Y_tr, X_te, Y_te)
    ob_tier2['SVM'].append(f1_score(Y_te, preds, average='macro'))

    preds = eval_sklearn_multiclass(MLPClassifier, MLP_KWARGS, X_tr, Y_tr, X_te, Y_te)
    ob_tier2['MLP'].append(f1_score(Y_te, preds, average='macro'))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
    X_te_s = scaler.transform(X_te.reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
    preds = train_gat_multiclass(X_tr_s, Y_tr, X_te_s, Y_te, seed=SEED+fold)
    ob_tier2['DeepGAT'].append(f1_score(Y_te, preds, average='macro'))

    print(f'  Fold {fold+1:2d}: SVM={ob_tier2["SVM"][-1]:.4f} | '
          f'MLP={ob_tier2["MLP"][-1]:.4f} | GAT={ob_tier2["DeepGAT"][-1]:.4f}')

print(f'\n── OpenBCI SUMMARY ── [{time.time()-t0:.0f}s]')
print('  WITH LEAKAGE:')
for m, s in ob_tier1.items():
    print(f'    {m:8s}: F1 = {np.mean(s):.4f} ± {np.std(s):.4f}')
print('  TRIAL-AWARE:')
for m, s in ob_tier2.items():
    print(f'    {m:8s}: F1 = {np.mean(s):.4f} ± {np.std(s):.4f}')
print()


# ═══════════════════════════════════════════════════════════════════════════════
# XAI: ATTENTION ANALYSIS + BIOLOGICAL INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

print('═' * 70)
print('XAI: BIOLOGICAL INTERPRETATION')
print('═' * 70)

# Pool data for XAI
all_X = np.concatenate([d['X'] for d in deap_data.values()])
all_Y_aro = np.concatenate([d['Y_aro'] for d in deap_data.values()])
all_Y_val = np.concatenate([d['Y_val'] for d in deap_data.values()])
all_Y = np.stack([all_Y_aro, all_Y_val], axis=1).astype(np.float32)

# Train XAI model on pooled DEAP
print('\nTraining GAT for attention extraction...')
n = len(all_X)
idx = np.random.permutation(n)
split = int(0.8 * n)
tr_idx_xai, te_idx_xai = idx[:split], idx[split:]

scaler_xai = StandardScaler()
X_tr_xai = scaler_xai.fit_transform(all_X[tr_idx_xai].reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
X_te_xai = scaler_xai.transform(all_X[te_idx_xai].reshape(-1, N_CH*N_FEATS)).reshape(-1, N_CH, N_FEATS)
Y_tr_xai = all_Y[tr_idx_xai]
Y_te_xai = all_Y[te_idx_xai]

torch.manual_seed(SEED)
xai_model = DeepGAT_DEAP().to(device)
pw_aro = torch.tensor([(Y_tr_xai[:,0]==0).sum()/(Y_tr_xai[:,0]==1).sum()], dtype=torch.float32).to(device)
pw_val = torch.tensor([(Y_tr_xai[:,1]==0).sum()/(Y_tr_xai[:,1]==1).sum()], dtype=torch.float32).to(device)
loss_fn_aro = nn.BCEWithLogitsLoss(pos_weight=pw_aro)
loss_fn_val = nn.BCEWithLogitsLoss(pos_weight=pw_val)
opt = optim.AdamW(xai_model.parameters(), lr=5e-4, weight_decay=1e-4)
train_dl = DataLoader(EEGDataset(X_tr_xai, Y_tr_xai), batch_size=256, shuffle=True, drop_last=True)
best_loss, best_state = float('inf'), None

for epoch in range(GAT_EPOCHS):
    xai_model.train()
    for Xb, Yb in train_dl:
        Xb, Yb = Xb.to(device), Yb.to(device)
        opt.zero_grad()
        logits = xai_model(Xb)
        loss = (loss_fn_aro(logits[:,0], Yb[:,0]) + loss_fn_val(logits[:,1], Yb[:,1])) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xai_model.parameters(), 1.0)
        opt.step()
    xai_model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_te_xai).float().to(device)
        Yt = torch.from_numpy(Y_te_xai).float().to(device)
        logits = xai_model(Xt)
        v_loss = (loss_fn_aro(logits[:,0], Yt[:,0]) + loss_fn_val(logits[:,1], Yt[:,1])).item() / 2
    if v_loss < best_loss:
        best_loss = v_loss
        best_state = {k: v.cpu().clone() for k, v in xai_model.state_dict().items()}

xai_model.load_state_dict(best_state)
xai_model.eval()
print(f'  XAI model trained (val loss: {best_loss:.4f})')

# Extract attention
print('Extracting attention weights (1000 test samples)...')
n_xai_samples = min(1000, len(X_te_xai))
x_xai_input = torch.from_numpy(X_te_xai[:n_xai_samples]).float().to(device)
with torch.no_grad():
    _, attns = xai_model(x_xai_input, return_attn=True)

attn_maps = [aw.mean(dim=(0, 1)).cpu().numpy() for aw in attns]
final_attn = attn_maps[-1]
ch_importance = final_attn.sum(axis=0)

# H1: Hemisphere Asymmetry
print('\n── H1: Hemisphere Asymmetry ──')
left_imp = ch_importance[LEFT_CH]
right_imp = ch_importance[RIGHT_CH]
t_stat, p_val = scipy_stats.ttest_rel(right_imp, left_imp)
print(f'  Left hemisphere attention:  {left_imp.mean():.4f} ± {left_imp.std():.4f}')
print(f'  Right hemisphere attention: {right_imp.mean():.4f} ± {right_imp.std():.4f}')
print(f'  Paired t-test (R > L): t={t_stat:.3f}, p={p_val:.4f}')

# H2: FAA
print('\n── H2: Frontal Alpha Asymmetry (FAA) ──')
alpha_de_idx = N_FEATS // 2 + ALPHA_BAND_IDX  # = 6
faa = all_X[:, F4_IDX, alpha_de_idx] - all_X[:, F3_IDX, alpha_de_idx]
faa_high_val = faa[all_Y_val == 1]
faa_low_val = faa[all_Y_val == 0]
t_faa, p_faa = scipy_stats.ttest_ind(faa_high_val, faa_low_val)
print(f'  FAA (high valence): {faa_high_val.mean():.4f} ± {faa_high_val.std():.4f}')
print(f'  FAA (low valence):  {faa_low_val.mean():.4f} ± {faa_low_val.std():.4f}')
print(f'  Independent t-test: t={t_faa:.3f}, p={p_faa:.2e}')

# H3: Channel Importance
print('\n── H3: Channel Importance (GAT attention) ──')
order = np.argsort(ch_importance)[::-1]
for rank, ch_idx in enumerate(order):
    region = 'FRONTAL' if ch_idx in FRONTAL_CH else 'OTHER'
    hemi = 'L' if ch_idx in LEFT_CH else 'R'
    print(f'  {rank+1:2d}. {CHANNEL_NAMES_DEAP[ch_idx]:4s} '
          f'(attn={ch_importance[ch_idx]:.4f}, {hemi}, {region})')

frontal_imp = ch_importance[FRONTAL_CH].mean()
other_imp = ch_importance[[i for i in range(N_CH) if i not in FRONTAL_CH]].mean()
print(f'\n  Frontal avg: {frontal_imp:.4f} | Non-frontal avg: {other_imp:.4f} | Ratio: {frontal_imp/other_imp:.2f}x')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print('\nGenerating plots...')

# Plot 1: Three-tier comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models = ['SVM', 'MLP', 'DeepGAT']
x = np.arange(len(models))
w = 0.25

t1_means = [np.mean(tier1_results[m]) for m in models]
t2_means = [np.mean(tier2_results[m]) for m in models]
t3_means = [np.mean(tier3_results[m]) for m in models]
t1_stds = [np.std(tier1_results[m]) for m in models]
t2_stds = [np.std(tier2_results[m]) for m in models]
t3_stds = [np.std(tier3_results[m]) for m in models]

axes[0].bar(x - w, t1_means, w, yerr=t1_stds, label='Tier 1 (Leaky)', color='#F44336', alpha=0.8, capsize=3)
axes[0].bar(x, t2_means, w, yerr=t2_stds, label='Tier 2 (Trial-Aware)', color='#4CAF50', alpha=0.8, capsize=3)
axes[0].bar(x + w, t3_means, w, yerr=t3_stds, label='Tier 3 (LOSO)', color='#2196F3', alpha=0.8, capsize=3)
axes[0].set_xticks(x); axes[0].set_xticklabels(models)
axes[0].set_ylabel('Macro F1'); axes[0].set_title('DEAP: Three-Tier Evaluation')
axes[0].legend(); axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

ob1_means = [np.mean(ob_tier1[m]) for m in models]
ob2_means = [np.mean(ob_tier2[m]) for m in models]
ob1_stds = [np.std(ob_tier1[m]) for m in models]
ob2_stds = [np.std(ob_tier2[m]) for m in models]
axes[1].bar(x - w/2, ob1_means, w, yerr=ob1_stds, label='Tier 1 (Leaky)', color='#F44336', alpha=0.8, capsize=3)
axes[1].bar(x + w/2, ob2_means, w, yerr=ob2_stds, label='Tier 2 (Trial-Aware)', color='#4CAF50', alpha=0.8, capsize=3)
axes[1].set_xticks(x); axes[1].set_xticklabels(models)
axes[1].set_ylabel('Macro F1'); axes[1].set_title('OpenBCI: Leaky vs Trial-Aware (4-class)')
axes[1].legend(); axes[1].set_ylim(0, 1)
axes[1].axhline(0.25, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'three_tier_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: XAI
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(final_attn, xticklabels=CHANNEL_NAMES_DEAP, yticklabels=CHANNEL_NAMES_DEAP,
            cmap='YlOrRd', ax=axes[0], linewidths=0.2)
axes[0].set_title('GAT Attention (Final Layer)\nQuery→Key connectivity')

colors = ['#2196F3' if i in LEFT_CH else '#F44336' for i in order]
axes[1].bar([CHANNEL_NAMES_DEAP[i] for i in order], ch_importance[order], color=colors)
axes[1].set_title('Channel Importance\nBlue=Left, Red=Right')
axes[1].tick_params(axis='x', rotation=45)

axes[2].hist(faa_high_val, bins=50, alpha=0.6, label=f'High Valence (n={len(faa_high_val)})', color='green')
axes[2].hist(faa_low_val, bins=50, alpha=0.6, label=f'Low Valence (n={len(faa_low_val)})', color='red')
axes[2].axvline(0, color='black', linestyle='--', alpha=0.5)
axes[2].set_title(f'Frontal Alpha Asymmetry (F4-F3)\nt={t_faa:.2f}, p={p_faa:.2e}')
axes[2].set_xlabel('FAA (DE_Alpha_F4 - DE_Alpha_F3)')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'xai_biological_interpretation.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f'  Saved: three_tier_comparison.png')
print(f'  Saved: xai_biological_interpretation.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

results = {
    'config': {
        'n_folds': N_FOLDS, 'gat_epochs': GAT_EPOCHS, 'gat_patience': GAT_PATIENCE,
        'features': '5BP+5DE', 'n_channels': N_CH, 'seed': SEED
    },
    'deap_tier1': {m: [float(x) for x in s] for m, s in tier1_results.items()},
    'deap_tier2': {m: [float(x) for x in s] for m, s in tier2_results.items()},
    'deap_tier3': {m: [float(x) for x in s] for m, s in tier3_results.items()},
    'openbci_tier1': {m: [float(x) for x in s] for m, s in ob_tier1.items()},
    'openbci_tier2': {m: [float(x) for x in s] for m, s in ob_tier2.items()},
    'xai': {
        'channel_importance': ch_importance.tolist(),
        'channel_names': CHANNEL_NAMES_DEAP,
        'hemisphere_test': {'t': float(t_stat), 'p': float(p_val),
                           'left_mean': float(left_imp.mean()), 'right_mean': float(right_imp.mean())},
        'faa_test': {'t': float(t_faa), 'p': float(p_faa),
                     'high_val_mean': float(faa_high_val.mean()),
                     'low_val_mean': float(faa_low_val.mean())},
        'frontal_ratio': float(frontal_imp / other_imp),
    }
}

with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n  Saved: all_results.json')


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '═' * 70)
print('FINAL RESULTS')
print('═' * 70)

print('\n┌─────────────────────────────────────────────────────────────────────────┐')
print('│ DEAP (20 subjects, binary Aro/Val, Macro F1, {}-fold)                    │'.format(N_FOLDS))
print('├──────────┬───────────────────┬───────────────────┬──────────────────────┤')
print('│ Model    │ Tier 1 (LEAKY)    │ Tier 2 (PROPER)   │ Tier 3 (LOSO)        │')
print('├──────────┼───────────────────┼───────────────────┼──────────────────────┤')
for m in models:
    t1 = f'{np.mean(tier1_results[m]):.3f}±{np.std(tier1_results[m]):.3f}'
    t2 = f'{np.mean(tier2_results[m]):.3f}±{np.std(tier2_results[m]):.3f}'
    t3 = f'{np.mean(tier3_results[m]):.3f}±{np.std(tier3_results[m]):.3f}'
    print(f'│ {m:8s} │ {t1:17s} │ {t2:17s} │ {t3:20s} │')
print('└──────────┴───────────────────┴───────────────────┴──────────────────────┘')

print('\n┌───────────────────────────────────────────────────────────┐')
print('│ OpenBCI (1 subject, 4-class, Macro F1, {}-fold)            │'.format(N_FOLDS))
print('├──────────┬───────────────────┬────────────────────────────┤')
print('│ Model    │ Tier 1 (LEAKY)    │ Tier 2 (TRIAL-AWARE)       │')
print('├──────────┼───────────────────┼────────────────────────────┤')
for m in models:
    t1 = f'{np.mean(ob_tier1[m]):.3f}±{np.std(ob_tier1[m]):.3f}'
    t2 = f'{np.mean(ob_tier2[m]):.3f}±{np.std(ob_tier2[m]):.3f}'
    print(f'│ {m:8s} │ {t1:17s} │ {t2:26s} │')
print('└──────────┴───────────────────┴────────────────────────────┘')

print('\n── XAI SUMMARY ──')
print(f'  Hemisphere: R={right_imp.mean():.4f} vs L={left_imp.mean():.4f}, t={t_stat:.3f}, p={p_val:.4f}')
print(f'  FAA: high_val={faa_high_val.mean():.4f} vs low_val={faa_low_val.mean():.4f}, t={t_faa:.1f}, p={p_faa:.2e}')
print(f'  Frontal importance ratio: {frontal_imp/other_imp:.2f}x')
print(f'  Top-3 channels: {", ".join(CHANNEL_NAMES_DEAP[i] for i in order[:3])}')

print(f'\n{"═"*70}')
print(f'DONE — all results in: {OUT_DIR}')
print(f'{"═"*70}')
