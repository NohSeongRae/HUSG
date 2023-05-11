import numpy as np
import torch
import random

def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED=10
EVALSEED=20

torch.backends.cudnn.benchmark = False

fixseed(SEED)