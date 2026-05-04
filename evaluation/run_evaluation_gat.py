"""
═══════════════════════════════════════════════════════════════════════════════
DeepGAT: THREE-TIER EVALUATION (GAT-ONLY)
═══════════════════════════════════════════════════════════════════════════════
TIER 1: With trial leakage (random split) — baseline/sanity check
TIER 2: Trial-aware CV (StratifiedGroupKFold by trial) — proper evaluation
TIER 3: Cross-subject LOSO (train N-1, test 1) — generalization test

Model: DeepGAT (3×GATLayer + dual/single task head)
Features: Configurable (default: 26/channel from features_v6)
Channels: 32 (DEAP), 16 (OpenBCI)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, pickle, warnings, glob, math, time, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
DEAP_FEAT_DIR = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'output', 'features_v6')
DEAP_FEAT_FALLBACK = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'output', 'features_v5')
OPENBCI_BASE = os.path.join(PROJECT_ROOT, '..', 'data', 'recordings_clean')
OPENBCI_FEAT_V5 = os.path.join(OPENBCI_BASE, 'features_v5', 'openbci_all.npz')
OPENBCI_FEAT_V2 = os.path.join(OPENBCI_BASE, 'data_extracted_v2')
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DEAP_N_CH = 32
OPENBCI_N_CH = 16
N_CH = 16  # Legacy default (OpenBCI); DEAP uses DEAP_N_CH
N_FOLDS = 10
GAT_EPOCHS = 200
GAT_PATIENCE = 20
GAT_LR = 5e-4
GAT_BATCH = 512

BAND_LABELS = ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma']
CHANNEL_NAMES_DEAP = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','F4','F8','FC6','FC2','C4','T8',
    'CP6','CP2','P4','P8','PO4','O2','O9','O10'
]
# Hemisphere indices for 32-channel layout
LEFT_CH  = list(range(0, 16))   # Channels 0-15 (left hemisphere + midline)
RIGHT_CH = list(range(16, 32))  # Channels 16-31 (right hemisphere + midline)
# Frontal channels: Fp1(0),AF3(1),F3(2),F7(3),FC5(4),FC1(5), Fp2(16),AF4(17),F4(18),F8(19),FC6(20),FC2(21)
FRONTAL_CH = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21]
F3_IDX, F4_IDX = 2, 18  # F3 and F4 indices for FAA
ALPHA_BAND_IDX = 1

SUBJECTS = [f'{i:02d}' for i in range(1, 21)]

# ─── Hyperparameters (can be overridden by Optuna) ────────────────────────────
DEFAULT_HPARAMS = {
    'backbone_dims': [64, 64, 64],
    'num_heads': 4,
    'attn_dropout': 0.05,
    'head_dense': 128,
    'head_dropout': 0.3,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'batch_size': 512,
    'epochs': 200,
    'patience': 20,
    'label_smoothing': 0.0,
    'input_noise_std': 0.0,
}


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
        h = self.W(x).view(B, N, self.H, self.d)
        e_src = (h * self.a_src).sum(-1)
        e_dst = (h * self.a_dst).sum(-1)
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
        alpha = self.attn_drop(F.softmax(e, dim=2))
        out = torch.einsum('bqkh, bkhd -> bqhd', alpha, h).mean(dim=2)
        out = self.bn(out.reshape(B * N, self.d)).reshape(B, N, self.d)
        out = F.elu(out)
        if self.residual:
            out = out + self.res_proj(x)
        if return_attn:
            return out, alpha.permute(0, 3, 1, 2)
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
    def __init__(self, n_ch=32, in_feats=26, backbone_dims=None,
                 dense=128, num_heads=4, attn_dropout=0.05, head_dropout=0.3):
        super().__init__()
        if backbone_dims is None:
            backbone_dims = [64, 64, 64]
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
    def __init__(self, n_ch=16, in_feats=26, n_classes=4,
                 backbone_dims=None, dense=128, num_heads=4,
                 attn_dropout=0.05, head_dropout=0.3):
        super().__init__()
        if backbone_dims is None:
            backbone_dims = [64, 64, 64]
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

def train_gat_deap(X_tr, Y_tr, X_te, Y_te, hparams=None, seed=42):
    """Train DeepGAT for binary arousal/valence classification.
    
    No training sample cap — uses all available data.
    """
    if hparams is None:
        hparams = DEFAULT_HPARAMS
    
    torch.manual_seed(seed)
    n_ch = X_tr.shape[1]
    n_feats = X_tr.shape[2]
    
    model = DeepGAT_DEAP(
        n_ch=n_ch,
        in_feats=n_feats,
        backbone_dims=hparams.get('backbone_dims', [64, 64, 64]),
        num_heads=hparams.get('num_heads', 4),
        attn_dropout=hparams.get('attn_dropout', 0.05),
        dense=hparams.get('head_dense', 128),
        head_dropout=hparams.get('head_dropout', 0.3),
    ).to(device)
    
    label_smooth = hparams.get('label_smoothing', 0.0)
    noise_std = hparams.get('input_noise_std', 0.0)
    
    pw_aro = torch.tensor([(Y_tr[:,0]==0).sum() / max((Y_tr[:,0]==1).sum(), 1)],
                          dtype=torch.float32).to(device)
    pw_val = torch.tensor([(Y_tr[:,1]==0).sum() / max((Y_tr[:,1]==1).sum(), 1)],
                          dtype=torch.float32).to(device)
    loss_aro = nn.BCEWithLogitsLoss(pos_weight=pw_aro)
    loss_val = nn.BCEWithLogitsLoss(pos_weight=pw_val)
    
    optimizer = optim.AdamW(model.parameters(),
                           lr=hparams.get('lr', 5e-4),
                           weight_decay=hparams.get('weight_decay', 1e-4))
    epochs = hparams.get('epochs', 200)
    patience = hparams.get('patience', 20)
    batch_size = hparams.get('batch_size', 512)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_dl = DataLoader(EEGDataset(X_tr, Y_tr), batch_size=batch_size,
                          shuffle=True, drop_last=len(X_tr) > batch_size)
    val_dl = DataLoader(EEGDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    best_loss, wait, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            
            # Input noise augmentation
            if noise_std > 0:
                Xb = Xb + torch.randn_like(Xb) * noise_std
            
            # Label smoothing
            if label_smooth > 0:
                Yb_smooth = Yb * (1 - label_smooth) + 0.5 * label_smooth
            else:
                Yb_smooth = Yb
            
            optimizer.zero_grad()
            logits = model(Xb)
            loss = (loss_aro(logits[:,0], Yb_smooth[:,0]) + loss_val(logits[:,1], Yb_smooth[:,1])) / 2
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


def train_gat_multiclass(X_tr, Y_tr, X_te, Y_te, n_classes=4, hparams=None, seed=42):
    """Train DeepGAT for multi-class classification.
    
    No training sample cap — uses all available data.
    """
    if hparams is None:
        hparams = DEFAULT_HPARAMS
    
    torch.manual_seed(seed)
    n_feats = X_tr.shape[2]
    
    model = DeepGAT_MultiClass(
        in_feats=n_feats,
        n_classes=n_classes,
        backbone_dims=hparams.get('backbone_dims', [64, 64, 64]),
        num_heads=hparams.get('num_heads', 4),
        attn_dropout=hparams.get('attn_dropout', 0.05),
        dense=hparams.get('head_dense', 128),
        head_dropout=hparams.get('head_dropout', 0.3),
    ).to(device)
    
    noise_std = hparams.get('input_noise_std', 0.0)
    label_smooth = hparams.get('label_smoothing', 0.0)
    
    counts = np.bincount(Y_tr, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * counts + 1e-6),
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smooth)
    
    optimizer = optim.AdamW(model.parameters(),
                           lr=hparams.get('lr', 5e-4),
                           weight_decay=hparams.get('weight_decay', 1e-4))
    epochs = hparams.get('epochs', 200)
    patience = hparams.get('patience', 20)
    batch_size = hparams.get('batch_size', 512)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_dl = DataLoader(EEGDataset(X_tr, Y_tr), batch_size=batch_size,
                          shuffle=True, drop_last=len(X_tr) > batch_size)
    val_dl = DataLoader(EEGDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    best_loss, wait, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            if noise_std > 0:
                Xb = Xb + torch.randn_like(Xb) * noise_std
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_deap_data():
    """Load DEAP features — prefer v6 (32ch), fall back to v5 (16ch)."""
    feat_dir = DEAP_FEAT_DIR
    if not os.path.exists(feat_dir) or len(glob.glob(os.path.join(feat_dir, '*.npz'))) == 0:
        print('  features_v6 not found, falling back to features_v5')
        feat_dir = DEAP_FEAT_FALLBACK
    
    deap_data = {}
    for sub in SUBJECTS:
        fp = os.path.join(feat_dir, f's{sub}.npz')
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        X = d['features']
        if X.ndim == 2:
            # v2 format: (N, 160) → reshape to 16ch
            n_feats = X.shape[1] // 16
            X = X.reshape(-1, 16, n_feats)
        n_ch = X.shape[1]  # Will be 32 for v6, 16 for v5
        labels = d['labels']
        trials = d['trials']
        Y_aro = (labels[:, 0] > 5.0).astype(np.int32)
        Y_val = (labels[:, 1] > 5.0).astype(np.int32)
        deap_data[sub] = {'X': X, 'Y_aro': Y_aro, 'Y_val': Y_val, 'trials': trials}
    
    return deap_data, feat_dir


def load_openbci_data():
    """Load OpenBCI features — prefer v5, fall back to v2."""
    label_map = {'calm': 0, 'happy': 1, 'sad': 2, 'stressed': 3}
    
    if os.path.exists(OPENBCI_FEAT_V5):
        d = np.load(OPENBCI_FEAT_V5, allow_pickle=True)
        return d['X'], d['Y'], d['groups'], 'v5'
    
    # Fallback: v2 format
    openbci_X, openbci_Y, openbci_groups = [], [], []
    feat_files = sorted(glob.glob(os.path.join(OPENBCI_FEAT_V2, '*_clean_trial_*.npy')))
    for fp in feat_files:
        fname = os.path.basename(fp)
        cat = fname.split('_clean_trial_')[0]
        if cat not in label_map:
            continue
        arr = np.load(fp, allow_pickle=True)
        for row in arr:
            openbci_X.append(row[0])
            openbci_Y.append(label_map[cat])
            openbci_groups.append(fname)
    
    X = np.array(openbci_X, dtype=np.float32)
    Y = np.array(openbci_Y, dtype=np.int64)
    groups = np.array(openbci_groups)
    
    # Reshape if needed
    if X.ndim == 2:
        n_feats = X.shape[1] // N_CH
        X = X.reshape(-1, N_CH, n_feats)
    
    return X, Y, groups, 'v2'


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Load hyperparameters (can be overridden by best_hparams.json from Optuna)
    hparams_file = os.path.join(OUT_DIR, 'best_hparams.json')
    if os.path.exists(hparams_file):
        with open(hparams_file) as f:
            hparams = json.load(f)
        print(f'Loaded tuned hyperparameters from {hparams_file}')
    else:
        hparams = DEFAULT_HPARAMS
        print('Using default hyperparameters')
    
    print(f'\n{"="*70}')
    print(f'DeepGAT THREE-TIER EVALUATION')
    print(f'{"="*70}')
    print(f'Device: {device}')
    print(f'Folds: {N_FOLDS} | Epochs: {hparams["epochs"]} | Patience: {hparams["patience"]}')
    print(f'Backbone: {hparams["backbone_dims"]} | Heads: {hparams["num_heads"]}')
    print(f'LR: {hparams["lr"]} | Batch: {hparams["batch_size"]}')
    print(f'Label smoothing: {hparams["label_smoothing"]} | Input noise: {hparams["input_noise_std"]}')
    print(f'Output: {OUT_DIR}')
    print()

    # Load data
    print('Loading DEAP features...')
    deap_data, feat_source = load_deap_data()
    deap_n_ch = list(deap_data.values())[0]['X'].shape[1]  # 32 for v6, 16 for v5
    N_FEATS = list(deap_data.values())[0]['X'].shape[2]
    print(f'  Source: {feat_source}')
    print(f'  Loaded: {len(deap_data)} subjects, {sum(len(d["X"]) for d in deap_data.values()):,} windows')
    print(f'  Shape: ({deap_n_ch}, {N_FEATS}) per window')

    print('\nLoading OpenBCI features...')
    openbci_X, openbci_Y, openbci_groups, ob_source = load_openbci_data()
    print(f'  Source: {ob_source}')
    print(f'  Windows: {len(openbci_X):,} | Trials: {len(np.unique(openbci_groups))} | Classes: {Counter(openbci_Y.tolist())}')
    
    # Reshape OpenBCI if different n_feats (pad or truncate)
    ob_n_ch = OPENBCI_N_CH
    ob_n_feats = openbci_X.shape[2] if openbci_X.ndim == 3 else openbci_X.shape[1] // ob_n_ch
    if openbci_X.ndim == 2:
        openbci_X = openbci_X.reshape(-1, ob_n_ch, ob_n_feats)
    
    print(flush=True)
    sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 1: WITH TRIAL LEAKAGE (Per-subject, random split)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('TIER 1: WITH TRIAL LEAKAGE (per-subject, random split)')
    print('=' * 70)
    t0 = time.time()

    tier1_results = []

    for sub_id, sub_data in tqdm(list(deap_data.items()), desc='Tier1 subjects'):
        X = sub_data['X']
        Y = np.stack([sub_data['Y_aro'], sub_data['Y_val']], axis=1).astype(np.float32)
        strat = sub_data['Y_aro'] * 2 + sub_data['Y_val']

        if len(np.unique(strat)) < 2:
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        sub_scores = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, strat)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr.reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
            X_te_s = scaler.transform(X_te.reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
            preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED+fold)
            f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
                  f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
            sub_scores.append(f1)

        tier1_results.append(np.mean(sub_scores))

    print(f'\n-- TIER 1 SUMMARY (WITH LEAKAGE) -- [{time.time()-t0:.0f}s]')
    print(f'  DeepGAT: F1 = {np.mean(tier1_results):.4f} +/- {np.std(tier1_results):.4f}  '
          f'(range: {np.min(tier1_results):.3f}-{np.max(tier1_results):.3f})')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 2: TRIAL-AWARE WITHIN-SUBJECT (StratifiedGroupKFold)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('TIER 2: TRIAL-AWARE WITHIN-SUBJECT (StratifiedGroupKFold by trial)')
    print('=' * 70)
    t0 = time.time()

    tier2_results = []

    for sub_id, sub_data in tqdm(list(deap_data.items()), desc='Tier2 subjects'):
        X = sub_data['X']
        Y = np.stack([sub_data['Y_aro'], sub_data['Y_val']], axis=1).astype(np.float32)
        trials = sub_data['trials']
        strat = sub_data['Y_aro'] * 2 + sub_data['Y_val']

        n_unique_trials = len(np.unique(trials))
        actual_folds = min(N_FOLDS, n_unique_trials)
        if len(np.unique(strat)) < 2 or actual_folds < 2:
            continue

        sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
        sub_scores = []

        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, strat, groups=trials)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr.reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
            X_te_s = scaler.transform(X_te.reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
            preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED+fold)
            f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
                  f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
            sub_scores.append(f1)

        tier2_results.append(np.mean(sub_scores))

    print(f'\n-- TIER 2 SUMMARY (TRIAL-AWARE) -- [{time.time()-t0:.0f}s]')
    print(f'  DeepGAT: F1 = {np.mean(tier2_results):.4f} +/- {np.std(tier2_results):.4f}  '
          f'(range: {np.min(tier2_results):.3f}-{np.max(tier2_results):.3f})')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 3: CROSS-SUBJECT LOSO
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('TIER 3: CROSS-SUBJECT LOSO (train N-1, test 1)')
    print('=' * 70)
    t0 = time.time()

    all_X_loso = np.concatenate([d['X'] for d in deap_data.values()])
    all_Y_loso = np.stack([
        np.concatenate([d['Y_aro'] for d in deap_data.values()]),
        np.concatenate([d['Y_val'] for d in deap_data.values()])
    ], axis=1).astype(np.float32)
    all_subs = np.concatenate([
        np.full(len(d['X']), int(sid)) for sid, d in deap_data.items()
    ])

    tier3_results = []

    for test_sub in tqdm(sorted(deap_data.keys()), desc='LOSO'):
        test_mask = all_subs == int(test_sub)
        train_mask = ~test_mask
        X_tr, X_te = all_X_loso[train_mask], all_X_loso[test_mask]
        Y_tr, Y_te = all_Y_loso[train_mask], all_Y_loso[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
        X_te_s = scaler.transform(X_te.reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
        preds = train_gat_deap(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED)
        f1 = (f1_score(Y_te[:,0], preds[:,0], zero_division=0) +
              f1_score(Y_te[:,1], preds[:,1], zero_division=0)) / 2
        tier3_results.append(f1)
        print(f'  s{test_sub}: F1={tier3_results[-1]:.4f}')

    print(f'\n-- TIER 3 SUMMARY (LOSO) -- [{time.time()-t0:.0f}s]')
    print(f'  DeepGAT: F1 = {np.mean(tier3_results):.4f} +/- {np.std(tier3_results):.4f}  '
          f'(range: {np.min(tier3_results):.3f}-{np.max(tier3_results):.3f})')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # OpenBCI: LEAKY vs TRIAL-AWARE (4-class)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('OpenBCI 4-CLASS: LEAKY vs TRIAL-AWARE')
    print('=' * 70)
    t0 = time.time()

    # Leaky
    print('\n-- Tier 1: WITH LEAKAGE --')
    skf_ob = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ob_tier1 = []

    for fold, (tr_idx, te_idx) in enumerate(skf_ob.split(openbci_X, openbci_Y)):
        X_tr, X_te = openbci_X[tr_idx], openbci_X[te_idx]
        Y_tr, Y_te = openbci_Y[tr_idx], openbci_Y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, ob_n_ch*ob_n_feats)).reshape(-1, ob_n_ch, ob_n_feats)
        X_te_s = scaler.transform(X_te.reshape(-1, ob_n_ch*ob_n_feats)).reshape(-1, ob_n_ch, ob_n_feats)
        preds = train_gat_multiclass(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED+fold)
        ob_tier1.append(f1_score(Y_te, preds, average='macro'))
        print(f'  Fold {fold+1:2d}: F1={ob_tier1[-1]:.4f}')

    # Trial-aware
    print('\n-- Tier 2: TRIAL-AWARE --')
    sgkf_ob = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ob_tier2 = []

    for fold, (tr_idx, te_idx) in enumerate(
        sgkf_ob.split(openbci_X, openbci_Y, groups=openbci_groups)):
        X_tr, X_te = openbci_X[tr_idx], openbci_X[te_idx]
        Y_tr, Y_te = openbci_Y[tr_idx], openbci_Y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, ob_n_ch*ob_n_feats)).reshape(-1, ob_n_ch, ob_n_feats)
        X_te_s = scaler.transform(X_te.reshape(-1, ob_n_ch*ob_n_feats)).reshape(-1, ob_n_ch, ob_n_feats)
        preds = train_gat_multiclass(X_tr_s, Y_tr, X_te_s, Y_te, hparams=hparams, seed=SEED+fold)
        ob_tier2.append(f1_score(Y_te, preds, average='macro'))
        print(f'  Fold {fold+1:2d}: F1={ob_tier2[-1]:.4f}')

    print(f'\n-- OpenBCI SUMMARY -- [{time.time()-t0:.0f}s]')
    print(f'  Tier 1 (Leaky):      F1 = {np.mean(ob_tier1):.4f} +/- {np.std(ob_tier1):.4f}')
    print(f'  Tier 2 (Trial-Aware): F1 = {np.mean(ob_tier2):.4f} +/- {np.std(ob_tier2):.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # XAI: ATTENTION ANALYSIS + BIOLOGICAL INTERPRETATION
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('XAI: BIOLOGICAL INTERPRETATION')
    print('=' * 70)

    all_X = np.concatenate([d['X'] for d in deap_data.values()])
    all_Y_aro = np.concatenate([d['Y_aro'] for d in deap_data.values()])
    all_Y_val = np.concatenate([d['Y_val'] for d in deap_data.values()])
    all_Y = np.stack([all_Y_aro, all_Y_val], axis=1).astype(np.float32)

    print('\nTraining GAT for attention extraction...')
    n = len(all_X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx_xai, te_idx_xai = idx[:split], idx[split:]

    scaler_xai = StandardScaler()
    X_tr_xai = scaler_xai.fit_transform(all_X[tr_idx_xai].reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
    X_te_xai = scaler_xai.transform(all_X[te_idx_xai].reshape(-1, deap_n_ch*N_FEATS)).reshape(-1, deap_n_ch, N_FEATS)
    Y_tr_xai = all_Y[tr_idx_xai]
    Y_te_xai = all_Y[te_idx_xai]

    torch.manual_seed(SEED)
    xai_model = DeepGAT_DEAP(n_ch=deap_n_ch, in_feats=N_FEATS,
                              backbone_dims=hparams.get('backbone_dims', [64,64,64]),
                              num_heads=hparams.get('num_heads', 4),
                              attn_dropout=hparams.get('attn_dropout', 0.05),
                              dense=hparams.get('head_dense', 128),
                              head_dropout=hparams.get('head_dropout', 0.3)).to(device)
    pw_aro = torch.tensor([(Y_tr_xai[:,0]==0).sum()/(Y_tr_xai[:,0]==1).sum()], dtype=torch.float32).to(device)
    pw_val = torch.tensor([(Y_tr_xai[:,1]==0).sum()/(Y_tr_xai[:,1]==1).sum()], dtype=torch.float32).to(device)
    loss_fn_aro = nn.BCEWithLogitsLoss(pos_weight=pw_aro)
    loss_fn_val = nn.BCEWithLogitsLoss(pos_weight=pw_val)
    opt = optim.AdamW(xai_model.parameters(), lr=hparams.get('lr', 5e-4),
                      weight_decay=hparams.get('weight_decay', 1e-4))
    train_dl = DataLoader(EEGDataset(X_tr_xai, Y_tr_xai), batch_size=256, shuffle=True, drop_last=True)
    best_loss, best_state = float('inf'), None

    for epoch in range(hparams.get('epochs', 200)):
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
    print('\n-- H1: Hemisphere Asymmetry --')
    left_imp = ch_importance[LEFT_CH]
    right_imp = ch_importance[RIGHT_CH]
    t_stat, p_val = scipy_stats.ttest_rel(right_imp, left_imp)
    print(f'  Left hemisphere attention:  {left_imp.mean():.4f} +/- {left_imp.std():.4f}')
    print(f'  Right hemisphere attention: {right_imp.mean():.4f} +/- {right_imp.std():.4f}')
    print(f'  Paired t-test (R > L): t={t_stat:.3f}, p={p_val:.4f}')

    # H2: FAA
    print('\n-- H2: Frontal Alpha Asymmetry (FAA) --')
    # In v5 features: BP(5) + DE(5) + PLV(5) + COH(5) + PAC(1) + TV(5) = 26
    # Alpha DE index: 5 (BP) + 1 (alpha index in DE) = 6
    alpha_de_idx = 5 + ALPHA_BAND_IDX  # Works for both v2 (5+1=6) and v5 (5+1=6)
    if N_FEATS > alpha_de_idx:
        faa = all_X[:, F4_IDX, alpha_de_idx] - all_X[:, F3_IDX, alpha_de_idx]
        faa_high_val = faa[all_Y_val == 1]
        faa_low_val = faa[all_Y_val == 0]
        t_faa, p_faa = scipy_stats.ttest_ind(faa_high_val, faa_low_val)
        print(f'  FAA (high valence): {faa_high_val.mean():.4f} +/- {faa_high_val.std():.4f}')
        print(f'  FAA (low valence):  {faa_low_val.mean():.4f} +/- {faa_low_val.std():.4f}')
        print(f'  Independent t-test: t={t_faa:.3f}, p={p_faa:.2e}')
    else:
        faa_high_val = faa_low_val = np.array([0])
        t_faa, p_faa = 0, 1
        print('  (insufficient features for FAA analysis)')

    # H3: Channel Importance
    print('\n-- H3: Channel Importance (GAT attention) --')
    order = np.argsort(ch_importance)[::-1]
    for rank, ch_idx in enumerate(order):
        region = 'FRONTAL' if ch_idx in FRONTAL_CH else 'OTHER'
        hemi = 'L' if ch_idx in LEFT_CH else 'R'
        print(f'  {rank+1:2d}. {CHANNEL_NAMES_DEAP[ch_idx]:4s} '
              f'(attn={ch_importance[ch_idx]:.4f}, {hemi}, {region})')

    frontal_imp = ch_importance[FRONTAL_CH].mean()
    other_imp = ch_importance[[i for i in range(deap_n_ch) if i not in FRONTAL_CH]].mean()
    print(f'\n  Frontal avg: {frontal_imp:.4f} | Non-frontal avg: {other_imp:.4f} | Ratio: {frontal_imp/other_imp:.2f}x')

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════════

    print('\nGenerating plots...')

    # Plot 1: DeepGAT performance across tiers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # DEAP tiers
    tiers = ['Tier 1\n(Leaky)', 'Tier 2\n(Trial-Aware)', 'Tier 3\n(LOSO)']
    means = [np.mean(tier1_results), np.mean(tier2_results), np.mean(tier3_results)]
    stds = [np.std(tier1_results), np.std(tier2_results), np.std(tier3_results)]
    colors = ['#F44336', '#4CAF50', '#2196F3']
    
    bars = axes[0].bar(tiers, means, yerr=stds, color=colors, alpha=0.8, capsize=5, width=0.5)
    axes[0].set_ylabel('Macro F1')
    axes[0].set_title('DEAP: DeepGAT Three-Tier Evaluation')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    for bar, m, s in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                    f'{m:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0].legend()

    # OpenBCI
    ob_tiers = ['Tier 1\n(Leaky)', 'Tier 2\n(Trial-Aware)']
    ob_means = [np.mean(ob_tier1), np.mean(ob_tier2)]
    ob_stds = [np.std(ob_tier1), np.std(ob_tier2)]
    
    bars = axes[1].bar(ob_tiers, ob_means, yerr=ob_stds, color=['#F44336', '#4CAF50'],
                      alpha=0.8, capsize=5, width=0.4)
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('OpenBCI: DeepGAT (4-class)')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (4-class)')
    for bar, m, s in zip(bars, ob_means, ob_stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                    f'{m:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'deepgat_tier_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: XAI
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(final_attn, xticklabels=CHANNEL_NAMES_DEAP, yticklabels=CHANNEL_NAMES_DEAP,
                cmap='YlOrRd', ax=axes[0], linewidths=0.2)
    axes[0].set_title('GAT Attention (Final Layer)\nQuery->Key connectivity')

    colors_bar = ['#2196F3' if i in LEFT_CH else '#F44336' for i in order]
    axes[1].bar([CHANNEL_NAMES_DEAP[i] for i in order], ch_importance[order], color=colors_bar)
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

    print(f'  Saved: deepgat_tier_comparison.png')
    print(f'  Saved: xai_biological_interpretation.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    results = {
        'config': {
            'n_folds': N_FOLDS, 'deap_n_channels': deap_n_ch,
            'openbci_n_channels': ob_n_ch, 'n_feats': N_FEATS,
            'seed': SEED, 'hparams': hparams,
            'feat_source': feat_source, 'ob_source': ob_source,
        },
        'deap_tier1': [float(x) for x in tier1_results],
        'deap_tier2': [float(x) for x in tier2_results],
        'deap_tier3': [float(x) for x in tier3_results],
        'openbci_tier1': [float(x) for x in ob_tier1],
        'openbci_tier2': [float(x) for x in ob_tier2],
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

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════

    print(f'\n{"="*70}')
    print('FINAL RESULTS')
    print('=' * 70)

    print(f'\n  DEAP (20 subjects, binary Aro/Val, Macro F1, {N_FOLDS}-fold)')
    print(f'  {"─"*50}')
    print(f'  Tier 1 (Leaky):      {np.mean(tier1_results):.4f} +/- {np.std(tier1_results):.4f}')
    print(f'  Tier 2 (Trial-Aware): {np.mean(tier2_results):.4f} +/- {np.std(tier2_results):.4f}')
    print(f'  Tier 3 (LOSO):       {np.mean(tier3_results):.4f} +/- {np.std(tier3_results):.4f}')

    print(f'\n  OpenBCI (1 subject, 4-class, Macro F1, {N_FOLDS}-fold)')
    print(f'  {"─"*50}')
    print(f'  Tier 1 (Leaky):      {np.mean(ob_tier1):.4f} +/- {np.std(ob_tier1):.4f}')
    print(f'  Tier 2 (Trial-Aware): {np.mean(ob_tier2):.4f} +/- {np.std(ob_tier2):.4f}')

    print(f'\n  XAI Summary')
    print(f'  {"─"*50}')
    print(f'  Hemisphere: R={right_imp.mean():.4f} vs L={left_imp.mean():.4f}, t={t_stat:.3f}, p={p_val:.4f}')
    print(f'  FAA: high_val={faa_high_val.mean():.4f} vs low_val={faa_low_val.mean():.4f}, t={t_faa:.1f}, p={p_faa:.2e}')
    print(f'  Frontal importance: {frontal_imp/other_imp:.2f}x')
    print(f'  Top-3 channels: {", ".join(CHANNEL_NAMES_DEAP[i] for i in order[:3])}')

    print(f'\n{"="*70}')
    print(f'DONE - all results in: {OUT_DIR}')
    print(f'{"="*70}')
