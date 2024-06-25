## reference: https://huggingface.co/datasets/Salesforce/wikitext-103

import os
import numpy as np
import tqdm # progress bar -- helps with debugging

# TODO: decide what other things we can use for the dataset

num_proc = 8 # number of processes to use