import os
import time
import math

import numpy as np
import torch

# default config values for training
# TODO: decide the default dataset for training


dataset = ''

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks