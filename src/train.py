import os
import time
import math

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# default config values for training


#data
dataset = 'coqa'


# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or 'mps' for the macbook

# DDP
backend = 'nccl'