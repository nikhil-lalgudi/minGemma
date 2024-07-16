import numpy as np
from datasets import load_dataset
import os
import tqdm
import multiprocessing

ds = load_dataset("stanfordnlp/imdb")

num_proc = multiprocessing.cpu_count() // 2
load_dataset_proc = num_proc

"""
learning_rate: 5e-05
train_batch_size: 8
eval_batch_size: 8
seed: 42
optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
lr_scheduler_type: linear
num_epochs: 1.0
"""