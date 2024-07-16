# reference: https://huggingface.co/datasets/billion-word-benchmark/lm1b
import numpy as np
import os
import tqdr
from datasets import load_dataset
import multiprocessing


num_proc = multiprocessing.cpu_count() // 2
num_proc_load_dataset = num_proc

ds = load_dataset("billion-word-benchmark/lm1b")