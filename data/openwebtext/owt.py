import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import multiprocessing

num_proc = multiprocessing.cpu_count() // 2
#try two processes 
dataset = load_dataset("openwebtext", num_proc = num_proc)
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)

### Created as a benchmark for a comparison with NanoGPT