"""wikisql_data.py: the WikiSQL rows every model uses (run 5), as the main repository's v12_wikisql_baselines.py draws them, so that the
reader, the transformer of the paper and BERT train on the SAME questions: train_all permuted by rng(0); the first n_val (2,000) are the
validation held out from train (never dev/test); the rest is the training pool, and a curve size n takes its first n questions. The dev
set is cut into the calibration half (even rows: thresholds and tuning choices, rule 25) and the report half (odd rows); the test set is
read once, at the largest size. Data path: the main repository's experiment/corp/wikisql/data/ (untracked; present on the DGX), or
$WIKISQL_DATA."""
import os
import numpy as np
from . import mainrepo

def set_path():
    mainrepo.import_experiment(); import v11_wikisql as v11
    v11.P = os.environ.get("WIKISQL_DATA", os.path.join(mainrepo.EXP, "corp", "wikisql", "data")).rstrip(os.sep) + os.sep
    return v11

def load(n_val=2000, n_dev=None, with_test=True):
    v11 = set_path()
    train_all, ttabs = v11.load("train"); dev, dtabs = v11.load("dev")
    if n_dev: dev = dev[:n_dev]
    test, tetabs = v11.load("test") if with_test else ([], {})
    perm = np.random.default_rng(0).permutation(len(train_all)); val = [train_all[i] for i in perm[:n_val]]; pool = [train_all[i] for i in perm[n_val:]]
    return {"train_all": train_all, "pool": pool, "val": val, "ttabs": ttabs, "dev": dev, "dtabs": dtabs, "cal": dev[0::2], "rep": dev[1::2],
            "test": test, "tetabs": tetabs}
