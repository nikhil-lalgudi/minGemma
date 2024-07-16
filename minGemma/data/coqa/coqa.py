import os
import numpy as np
import tqdm # progress bar -- helps with debugging
from datasets import load_dataset
import multiprocessing

num_proc = multiprocessing.cpu_count() // 2
data_dir = 'data/coqa' # directory ?

# tokenize the dataset
"""
Start off by creating some form of encoding function
"""

def tokenize_function(example):
    # TODO: implement this function
    pass


ds = load_dataset("stanfordnlp/coqa")


##reference: https://huggingface.co/datasets/stanfordnlp/coqa