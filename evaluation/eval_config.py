from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import os

BANDS = [(4, 8), (8, 12), (12, 16), (16, 25), (25, 45)]
BAND_NAMES = ['theta', 'alpha', 'beta_l', 'beta_h', 'gamma']
N_BANDS = len(BANDS)
ALPHA_BAND_IDX = BAND_NAMES.index('alpha')

DEAP_CHANNEL_NAMES = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','F4','F8','FC6','FC2','C4','T8',
    'CP6','CP2','P4','P8','PO4','O2','O9','O10'
]
OPENBCI_CHANNEL_NAMES = ['Fp1','F3','F7','C3','T7','P3','P7','O1',
                         'Fp2','F4','F8','C4','T8','P4','P8','O2']
DEAP_N_CH = len(DEAP_CHANNEL_NAMES)
OPENBCI_N_CH = len(OPENBCI_CHANNEL_NAMES)

FEATURE_GROUP_SIZES = {
    'bp': N_BANDS,
    'de': N_BANDS,
    'plv': N_BANDS,
    'coherence': N_BANDS,
    'pac': 1,
    'temporal': N_BANDS,
}


@dataclass(frozen=True)
class FeatureSet:
    key: str
    name: str
    groups: Tuple[str, ...]
    description: str

    @property
    def n_feats(self) -> int:
        return sum(FEATURE_GROUP_SIZES[group] for group in self.groups)


FEATURE_SETS: Dict[str, FeatureSet] = {
    'bpde': FeatureSet(
        key='bpde',
        name='bpde_10',
        groups=('bp', 'de'),
        description='Band power + differential entropy (10/channel)',
    ),
    'full': FeatureSet(
        key='full',
        name='full_26',
        groups=('bp', 'de', 'plv', 'coherence', 'pac', 'temporal'),
        description='BP+DE+PLV+Coherence+PAC+Temporal (26/channel)',
    ),
}

DEFAULT_FEATURE_SET = 'full'


def get_feature_set(feature_set_key: str) -> FeatureSet:
    try:
        return FEATURE_SETS[feature_set_key]
    except KeyError as exc:
        raise ValueError(
            f'Unknown feature set "{feature_set_key}". '
            f'Valid options: {", ".join(sorted(FEATURE_SETS))}.'
        ) from exc


def feature_group_offsets(feature_set: FeatureSet) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    idx = 0
    for group in feature_set.groups:
        offsets[group] = idx
        idx += FEATURE_GROUP_SIZES[group]
    return offsets


def feature_index(feature_set: FeatureSet, group: str, band_idx: int | None = None) -> int:
    if group not in FEATURE_GROUP_SIZES:
        raise ValueError(f'Unknown feature group "{group}".')
    offsets = feature_group_offsets(feature_set)
    if group not in offsets:
        raise ValueError(f'Feature group "{group}" not present in set "{feature_set.key}".')
    if group == 'pac':
        if band_idx is not None:
            raise ValueError('PAC feature does not take a band index.')
        return offsets[group]
    if band_idx is None:
        raise ValueError(f'Band index required for group "{group}".')
    if not 0 <= band_idx < N_BANDS:
        raise ValueError(f'Band index {band_idx} out of range for {N_BANDS} bands.')
    return offsets[group] + band_idx


def get_cache_dirs(project_root: str, feature_set: FeatureSet) -> Tuple[str, str]:
    deap_dir = os.path.join(project_root, '..', 'data', 'DEAP', 'output',
                            f'features_{feature_set.name}')
    openbci_dir = os.path.join(project_root, '..', 'data', 'recordings_clean',
                               f'features_{feature_set.name}')
    return deap_dir, openbci_dir


def validate_feature_array(dataset_label: str, X, n_channels: int, feature_set: FeatureSet) -> None:
    if X.ndim != 3:
        raise ValueError(
            f'{dataset_label} features must be 3D (N, channels, feats); got shape {X.shape}.'
        )
    if X.shape[1] != n_channels:
        raise ValueError(
            f'{dataset_label} channels mismatch: expected {n_channels}, got {X.shape[1]}.'
        )
    if X.shape[2] != feature_set.n_feats:
        raise ValueError(
            f'{dataset_label} feature count mismatch: expected {feature_set.n_feats}, '
            f'got {X.shape[2]}.'
        )
