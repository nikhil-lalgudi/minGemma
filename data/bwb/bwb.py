# reference: https://huggingface.co/datasets/billion-word-benchmark/lm1b
import numpy as np
import os
import tqdr

num_proc = 8 
num_proc_load_dataset = num_proc

ds = load_dataset("billion-word-benchmark/lm1b")