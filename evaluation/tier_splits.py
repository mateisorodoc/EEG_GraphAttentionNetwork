from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_tier1_splits(X, strat, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(strat)) < 2:
        return []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(X, strat))


def make_tier2_splits(
    X,
    strat,
    groups,
    n_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(strat)) < 2:
        return []
    n_unique_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique_groups)
    if actual_folds < 2:
        return []
    sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(X, strat, groups=groups))


def iter_loso_splits(all_subjects, subject_ids: Iterable[int] | None = None):
    subjects = sorted(np.unique(all_subjects)) if subject_ids is None else list(subject_ids)
    for test_sub in subjects:
        test_mask = all_subjects == int(test_sub)
        train_mask = ~test_mask
        yield int(test_sub), train_mask, test_mask
