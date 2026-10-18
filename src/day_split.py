"""One rule for splitting operating days, shared by the tabular pipeline and fcll_net.

Both used to carry their own copy of the shuffle-and-cut code, so a change to one
silently desynchronised the test days of the two model families.
"""
import numpy as np

from config.const_training import RANDOM_STATE, TRAIN_FRACTION


def split_days(days, seed=RANDOM_STATE, train_fraction=TRAIN_FRACTION):
    """Shuffle the unique `days` and cut them into (train_days, test_days) sets.

    Not stratified: stratifying by label would require balancing the positive rate
    across whole days, which is not generally satisfiable and would silently drop
    days to get there. The base rate is stable enough across days (about 23.5%)
    that random day assignment holds it.
    """
    unique = np.array(sorted(set(days)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(round(len(unique) * train_fraction))
    return set(unique[:n_train]), set(unique[n_train:])
