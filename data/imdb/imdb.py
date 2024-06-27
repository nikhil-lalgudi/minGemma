import numpy as np
from datasets import load_dataset
import os
import tqdm

ds = load_dataset("stanfordnlp/imdb")

num_proc = 8 # number of processes to use
num_proc_load_dataset = num_proc