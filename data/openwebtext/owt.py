import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

num_processes = 8 
#try two processes 
dataset = load_dataset("openwebtext", num_proc=8)

### Created as a benchmark for a comparison with NanoGPT