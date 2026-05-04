"""
═══════════════════════════════════════════════════════════════════════════════
DeepGAT: Enhanced Feature Extraction for EEG Emotion Classification
═══════════════════════════════════════════════════════════════════════════════
Extracts enhanced features from raw DEAP/OpenBCI data:
  - Band Power (BP): 5 bands × 16 channels
  - Differential Entropy (DE): 5 bands × 16 channels
  - Phase-Locking Value (PLV): mean connectivity per channel per band
  - Coherence: mean Welch coherence per channel per band
  - Phase-Amplitude Coupling (PAC): theta-gamma MI per channel
  - Temporal Variability: sub-window BP variance per channel per band

Output: (N_windows, 32, N_FEATS) arrays cached as features_v6/ [DEAP]
        (N_windows, 16, N_FEATS) arrays cached as features_v5/ [OpenBCI]
═══════════════════════════════════════════════════════════════════════════════
"""

import os, pickle, sys, glob
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from tqdm import tqdm

from eval_config import (
    BANDS,
    BAND_NAMES,
    N_BANDS,
    DEAP_CHANNEL_NAMES,
    OPENBCI_CHANNEL_NAMES,
    DEAP_N_CH,
    OPENBCI_N_CH,
    FEATURE_SETS,
    DEFAULT_FEATURE_SET,
    get_feature_set,
    get_cache_dirs,
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Configuration ────────────────────────────────────────────────────────────
FS = 128  # Sampling frequency
WINDOW = 256  # 2 seconds
STEP = 128  # 1 second (reduced overlap from 93.75% to 50% for better independence)
PRE_TRIAL = 3 * FS  # 3s baseline to skip in DEAP

DEAP_32_CHANNELS = list(range(DEAP_N_CH))  # Use all 32 EEG channels

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEAP_RAW = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'data_preprocessed_python')
OPENBCI_RAW = os.path.join(PROJECT_ROOT, '..', 'data', 'recordings_clean')


def resolve_output_dirs(feature_set):
    deap_out, openbci_out = get_cache_dirs(PROJECT_ROOT, feature_set)
    os.makedirs(deap_out, exist_ok=True)
    os.makedirs(openbci_out, exist_ok=True)
    return deap_out, openbci_out


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(data, lo, hi, fs=FS, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2
    lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
    b, a = signal.butter(order, [lo_n, hi_n], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def compute_band_power(window, fs=FS):
    """Compute band power for all channels using Welch's method.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        bp: (N_CH, N_BANDS) array
    """
    freqs, psd = signal.welch(window, fs=fs, nperseg=min(128, window.shape[-1]),
                              noverlap=64, axis=-1)
    bp = np.zeros((window.shape[0], N_BANDS))
    for b, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        bp[:, b] = np.log1p(psd[:, mask].mean(axis=-1))
    return bp


def compute_differential_entropy(window, fs=FS):
    """Compute DE for all channels per band.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        de: (N_CH, N_BANDS) array
    """
    de = np.zeros((window.shape[0], N_BANDS))
    for b, (lo, hi) in enumerate(BANDS):
        filtered = bandpass_filter(window, lo, hi, fs)
        var = np.var(filtered, axis=-1) + 1e-10
        de[:, b] = 0.5 * np.log(2 * np.pi * np.e * var)
    return de


def compute_plv_summary(window, fs=FS):
    """Compute mean Phase-Locking Value for each channel per band.
    
    PLV between channels i and j in band b:
      PLV_ij = |mean(exp(j*(phase_i - phase_j)))|
    
    Per-channel summary: mean PLV to all other channels.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        plv_summary: (N_CH, N_BANDS) array
    """
    n_ch = window.shape[0]
    plv_summary = np.zeros((n_ch, N_BANDS))
    
    for b, (lo, hi) in enumerate(BANDS):
        filtered = bandpass_filter(window, lo, hi, fs)
        analytic = hilbert(filtered, axis=-1)
        phases = np.angle(analytic)  # (N_CH, WINDOW)
        
        # Compute pairwise PLV efficiently
        # Phase differences: (N_CH, N_CH, WINDOW)
        phase_diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
        plv_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))  # (N_CH, N_CH)
        
        # Per-channel mean (exclude self-connection)
        np.fill_diagonal(plv_matrix, 0)
        plv_summary[:, b] = plv_matrix.sum(axis=1) / (n_ch - 1)
    
    return plv_summary


def compute_coherence_summary(window, fs=FS):
    """Compute mean magnitude-squared coherence per channel per band.
    
    Uses Welch cross-spectral density.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        coh_summary: (N_CH, N_BANDS) array
    """
    n_ch = window.shape[0]
    nperseg = min(64, window.shape[-1] // 2)
    coh_summary = np.zeros((n_ch, N_BANDS))
    
    # Compute PSD for each channel
    freqs, psd_all = signal.welch(window, fs=fs, nperseg=nperseg, axis=-1)
    
    # For each channel pair, compute coherence
    # This is O(n_ch^2) but n_ch is small, so it's fine
    coh_matrix = np.zeros((n_ch, n_ch, N_BANDS))
    
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs_c, cxy = signal.coherence(window[i], window[j], fs=fs, nperseg=nperseg)
            for b, (lo, hi) in enumerate(BANDS):
                mask = (freqs_c >= lo) & (freqs_c < hi)
                if mask.any():
                    coh_val = cxy[mask].mean()
                    coh_matrix[i, j, b] = coh_val
                    coh_matrix[j, i, b] = coh_val
    
    # Per-channel mean coherence
    for b in range(N_BANDS):
        mat = coh_matrix[:, :, b]
        coh_summary[:, b] = mat.sum(axis=1) / (n_ch - 1)
    
    return coh_summary


def compute_pac(window, fs=FS):
    """Compute Phase-Amplitude Coupling (theta phase → gamma amplitude).
    
    Modulation Index (MI) using mean vector length method.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        pac: (N_CH, 1) array
    """
    n_ch = window.shape[0]
    
    # Theta phase (4-8 Hz)
    theta_filtered = bandpass_filter(window, 4, 8, fs)
    theta_phase = np.angle(hilbert(theta_filtered, axis=-1))
    
    # Gamma amplitude (25-45 Hz)
    gamma_filtered = bandpass_filter(window, 25, 45, fs)
    gamma_amp = np.abs(hilbert(gamma_filtered, axis=-1))
    
    # Modulation Index: |mean(amp * exp(j*phase))| / mean(amp)
    mi = np.abs(np.mean(gamma_amp * np.exp(1j * theta_phase), axis=-1)) / (np.mean(gamma_amp, axis=-1) + 1e-10)
    
    return mi.reshape(n_ch, 1)


def compute_temporal_variability(window, fs=FS, n_subwindows=4):
    """Compute temporal variability of band power within the window.
    
    Split window into n_subwindows chunks, compute BP for each,
    then take the variance across chunks.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        tv: (N_CH, N_BANDS) array
    """
    n_ch = window.shape[0]
    chunk_size = window.shape[-1] // n_subwindows
    
    # BP for each sub-window
    sub_bps = []
    for i in range(n_subwindows):
        chunk = window[:, i*chunk_size:(i+1)*chunk_size]
        freqs, psd = signal.welch(chunk, fs=fs, nperseg=min(32, chunk_size),
                                  noverlap=16, axis=-1)
        bp = np.zeros((n_ch, N_BANDS))
        for b, (lo, hi) in enumerate(BANDS):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                bp[:, b] = psd[:, mask].mean(axis=-1)
        sub_bps.append(bp)
    
    sub_bps = np.stack(sub_bps, axis=0)  # (n_subwindows, N_CH, N_BANDS)
    tv = np.var(sub_bps, axis=0)  # (N_CH, N_BANDS)
    tv = np.log1p(tv)  # Log-scale to prevent explosion
    
    return tv


def extract_features_window(window, fs=FS, feature_groups=None):
    """Extract all features for a single window.
    
    Args:
        window: (N_CH, WINDOW_SAMPLES) array
        fs: sampling frequency
        feature_groups: list of groups to compute, or None for all
    Returns:
        features: (N_CH, N_FEATS) array where N_FEATS depends on groups selected
    """
    if feature_groups is None:
        feature_groups = ['bp', 'de', 'plv', 'coherence', 'pac', 'temporal']
    
    parts = []
    
    if 'bp' in feature_groups:
        parts.append(compute_band_power(window, fs))  # (16, 5)
    
    if 'de' in feature_groups:
        parts.append(compute_differential_entropy(window, fs))  # (16, 5)
    
    if 'plv' in feature_groups:
        parts.append(compute_plv_summary(window, fs))  # (16, 5)
    
    if 'coherence' in feature_groups:
        parts.append(compute_coherence_summary(window, fs))  # (16, 5)
    
    if 'pac' in feature_groups:
        parts.append(compute_pac(window, fs))  # (16, 1)
    
    if 'temporal' in feature_groups:
        parts.append(compute_temporal_variability(window, fs))  # (16, 5)
    
    return np.concatenate(parts, axis=1)  # (16, total_feats)


# ═══════════════════════════════════════════════════════════════════════════════
# DEAP EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_deap_subject(sub_id, feature_groups=None):
    """Extract enhanced features for one DEAP subject.
    
    Args:
        sub_id: subject string like '01'
    Returns:
        dict with 'features' (N, 32, F), 'labels' (N, 2), 'trials' (N,)
    """
    fp = os.path.join(DEAP_RAW, f's{sub_id}.dat')
    with open(fp, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    
    data = d['data']  # (40, 40, 8064)
    labels = d['labels']  # (40, 4): valence, arousal, dominance, liking
    
    all_features = []
    all_labels = []
    all_trials = []
    
    for trial_idx in range(40):
        # Extract all 32 EEG channels
        eeg = data[trial_idx, :DEAP_N_CH, :]  # (32, 8064)
        
        # Skip pre-trial baseline (first 3 seconds)
        eeg = eeg[:, PRE_TRIAL:]  # (32, 7680)
        
        # Sliding window
        n_samples = eeg.shape[1]
        n_windows = (n_samples - WINDOW) // STEP + 1
        
        for w_idx in range(n_windows):
            start = w_idx * STEP
            window = eeg[:, start:start + WINDOW]
            
            feats = extract_features_window(window, FS, feature_groups)
            all_features.append(feats)
            all_labels.append([labels[trial_idx, 1], labels[trial_idx, 0]])  # arousal, valence
            all_trials.append(trial_idx)
    
    return {
        'features': np.array(all_features, dtype=np.float32),
        'labels': np.array(all_labels, dtype=np.float32),
        'trials': np.array(all_trials, dtype=np.int32)
    }


def extract_all_deap(feature_groups=None, subjects=None, out_dir=None):
    """Extract features for all DEAP subjects and cache."""
    if out_dir is None:
        raise ValueError('DEAP output directory is required.')
    if subjects is None:
        subjects = [f'{i:02d}' for i in range(1, 21)]
    
    print(f'Extracting DEAP features (v6) for {len(subjects)} subjects...')
    print(f'  Window: {WINDOW} samples ({WINDOW/FS:.1f}s), Step: {STEP} ({STEP/FS:.2f}s)')
    print(f'  Feature groups: {feature_groups or "all"}')
    
    for sub_id in tqdm(subjects, desc='DEAP subjects'):
        out_fp = os.path.join(out_dir, f's{sub_id}.npz')
        if os.path.exists(out_fp):
            # Verify shape
            d = np.load(out_fp)
            if d['features'].ndim == 3 and d['features'].shape[1] == DEAP_N_CH:
                continue
        
        result = extract_deap_subject(sub_id, feature_groups)
        np.savez_compressed(out_fp,
                           features=result['features'],
                           labels=result['labels'],
                           trials=result['trials'])
        print(f'  s{sub_id}: {result["features"].shape[0]} windows, '
              f'{result["features"].shape[2]} features/channel')
    
    print('DEAP extraction complete.')


# ═══════════════════════════════════════════════════════════════════════════════
# OpenBCI EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_openbci_trial(filepath, feature_groups=None, fs=FS):
    """Extract features from one OpenBCI .npy trial file.
    
    The .npy files in data_extracted_v2 contain pre-extracted features.
    For raw data we need to look in recordings_*_cleaned/ folders.
    """
    # data_extracted_v3 has raw-like format: load and window
    arr = np.load(filepath, allow_pickle=True)
    
    # arr is array of [features_array, ...] from old extraction
    # We need to re-extract from the cleaned recordings instead
    # For now, use the existing data_extracted_v2 format
    # Each element is (features_flat,) with shape (160,) = 16ch * 10feats
    windows = []
    for row in arr:
        if isinstance(row, np.ndarray) and row.ndim == 1:
            # Old format: flat 160-dim vector
            feat = row[0] if isinstance(row[0], np.ndarray) else row
            windows.append(feat.reshape(OPENBCI_N_CH, -1))
        elif isinstance(row, (list, np.ndarray)) and len(row) >= 1:
            feat = row[0] if isinstance(row[0], np.ndarray) else row
            if hasattr(feat, 'shape'):
                if feat.ndim == 1:
                    windows.append(feat.reshape(OPENBCI_N_CH, -1))
                elif feat.ndim == 2:
                    windows.append(feat)
    
    return np.array(windows, dtype=np.float32) if windows else None


def extract_openbci_from_npy(feature_groups=None, feature_set=None):
    """Extract OpenBCI features from existing .npy caches.
    
    Since we don't have raw waveforms easily accessible for re-windowing,
    we'll load the v3 .npy files which contain raw EEG segments, and
    re-extract with the new feature pipeline.
    """
    if feature_set is None:
        raise ValueError('feature_set is required for OpenBCI extraction.')

    v3_dir = os.path.join(OPENBCI_RAW, 'data_extracted_v3')
    
    if not os.path.exists(v3_dir):
        print(f'  ⚠ data_extracted_v3 not found, falling back to v2 format')
        return _extract_openbci_v2_fallback(feature_set)
    
    label_map = {'calm': 0, 'happy': 1, 'sad': 2, 'stressed': 3}
    all_X, all_Y, all_groups = [], [], []
    
    files = sorted(glob.glob(os.path.join(v3_dir, '*_clean_trial_*.npy')))
    print(f'  Found {len(files)} trial files in data_extracted_v3')
    
    for fp in tqdm(files, desc='OpenBCI trials'):
        fname = os.path.basename(fp)
        cat = fname.split('_clean_trial_')[0]
        if cat not in label_map:
            continue
        
        try:
            raw_data = np.load(fp, allow_pickle=True)
            # v3 format: (n_windows, 16, raw_samples_per_window)
            if raw_data.ndim == 3 and raw_data.shape[1] == OPENBCI_N_CH:
                for w_idx in range(raw_data.shape[0]):
                    window = raw_data[w_idx]  # (16, samples)
                    if window.shape[1] >= WINDOW:
                        window = window[:, :WINDOW]
                        feats = extract_features_window(window, FS, feature_groups)
                        all_X.append(feats)
                        all_Y.append(label_map[cat])
                        all_groups.append(fname)
            else:
                # Fallback: treat as pre-extracted
                for row in raw_data:
                    feat = row[0] if isinstance(row, (list, np.ndarray)) and len(row) > 0 else row
                    if hasattr(feat, 'shape') and feat.size == OPENBCI_N_CH * 10:
                        all_X.append(feat.reshape(OPENBCI_N_CH, 10))
                        all_Y.append(label_map[cat])
                        all_groups.append(fname)
        except Exception as e:
            print(f'  ⚠ Error processing {fname}: {e}')
            continue
    
    if all_X:
        return (np.array(all_X, dtype=np.float32),
                np.array(all_Y, dtype=np.int64),
                np.array(all_groups))
    return _extract_openbci_v2_fallback(feature_set)


def _extract_openbci_v2_fallback(feature_set):
    """Fallback: load OpenBCI from v2/v3 format (pre-extracted BP+DE only).
    
    v3 non-v3-suffix files: (N_windows, 2) object array where col 0 is (16, 10).
    v2 files: same format.
    """
    if feature_set.groups != ('bp', 'de'):
        raise ValueError(
            f'OpenBCI v2/v3 fallback only supports bp+de (10 features). '
            f'Feature set "{feature_set.key}" requires {feature_set.n_feats} features.'
        )
    # Try v3 first (more files), then v2
    v3_dir = os.path.join(OPENBCI_RAW, 'data_extracted_v3')
    v2_dir = os.path.join(OPENBCI_RAW, 'data_extracted_v2')
    
    src_dir = v3_dir if os.path.exists(v3_dir) else v2_dir
    label_map = {'calm': 0, 'happy': 1, 'sad': 2, 'stressed': 3}
    all_X, all_Y, all_groups = [], [], []
    
    # Get non-v3-suffix files from v3, or all from v2
    files = sorted(glob.glob(os.path.join(src_dir, '*_clean_trial_*.npy')))
    files = [f for f in files if not f.endswith('_v3.npy')]
    print(f'  Loading {len(files)} trial files from {os.path.basename(src_dir)} (BP+DE only)')
    
    for fp in files:
        fname = os.path.basename(fp)
        cat = fname.split('_clean_trial_')[0]
        if cat not in label_map:
            continue
        arr = np.load(fp, allow_pickle=True)
        for row in arr:
            feat = row[0] if isinstance(row, (list, np.ndarray)) and len(row) > 0 else row
            if hasattr(feat, 'shape') and feat.size == OPENBCI_N_CH * 10:
                all_X.append(feat.reshape(OPENBCI_N_CH, 10))
                all_Y.append(label_map[cat])
                all_groups.append(fname)
    
    return (np.array(all_X, dtype=np.float32),
            np.array(all_Y, dtype=np.int64),
            np.array(all_groups))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['deap', 'openbci', 'all'], default='all')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='DEAP subject IDs (e.g., 01 02 03)')
    parser.add_argument('--force', action='store_true', help='Re-extract even if cache exists')
    parser.add_argument('--feature-set', choices=FEATURE_SETS.keys(),
                        default=DEFAULT_FEATURE_SET,
                        help='Feature set to extract (bpde=10, full=26)')
    args = parser.parse_args()
    
    feature_set = get_feature_set(args.feature_set)
    feature_groups = list(feature_set.groups)
    deap_out, openbci_out = resolve_output_dirs(feature_set)
    
    if args.dataset in ('deap', 'all'):
        if args.force:
            # Remove existing cache
            for f in glob.glob(os.path.join(deap_out, '*.npz')):
                os.remove(f)
        extract_all_deap(feature_groups, args.subjects, out_dir=deap_out)
    
    if args.dataset in ('openbci', 'all'):
        print('\nExtracting OpenBCI features...')
        X, Y, groups = extract_openbci_from_npy(feature_groups, feature_set=feature_set)
        if X is not None:
            np.savez_compressed(os.path.join(openbci_out, 'openbci_all.npz'),
                              X=X, Y=Y, groups=groups)
            print(f'  Saved: {X.shape[0]} windows, shape per window: {X.shape[1:]}')
