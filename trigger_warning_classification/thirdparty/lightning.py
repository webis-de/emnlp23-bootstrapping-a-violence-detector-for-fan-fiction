# Taken from: https://github.com/Lightning-AI/lightning/blob/cfab52af1c873b23564de8f150ce602e532d0d4f/src/lightning/fabric/utilities/seed.py
# License: Apache 2.0 (see LICENSE-lightning)

import logging
import os
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Optional

import numpy as np
import torch


log = logging.getLogger(__name__)

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: int) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.
    """
    if not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        log.warning(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    log.info(f"Global seed set to {seed}", _get_rank())
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def _select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)
