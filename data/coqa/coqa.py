import os
import numpy as np
import tqdm # progress bar -- helps with debugging
from datasets import load_dataset

num_proc = 8 # number of processes to use
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