## reference: https://huggingface.co/datasets/Salesforce/wikitext
import os
import numpy as np
import tqdm 

from datasets import load_dataset


num_proc = 8 # number of processes to use
num_proc_load_dataset = num_proc

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
split_dataset = dataset["train"].train_test_split(test_size= ... , seed= ...., shuffle=True)


