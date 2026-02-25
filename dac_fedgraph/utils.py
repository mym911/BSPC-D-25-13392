from __future__ import annotations
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch

def set_global_seed(seed: int, deterministic: bool = False):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_logging(log_file: str = "model_training.log", level: int = logging.WARNING) -> logging.Logger:
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    logging.basicConfig(filename=log_file, level=level, format="%(asctime)s:%(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
    logging.getLogger("").addHandler(console)

    logging.getLogger("matplotlib").setLevel(level)
    logging.getLogger("nilearn").setLevel(level)
    return logging.getLogger(__name__)
