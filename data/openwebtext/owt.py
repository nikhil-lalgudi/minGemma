import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import multiprocessing

# config
NUM_PROC = max(1, multiprocessing.cpu_count() // 2)
ENCODING = "gpt2"
VAL_SIZE = 0.0005
SEED = 2357
BATCH_SIZE = 1000

def encode_text(text):
    ids = enc.encode_ordinary(text) + [enc.eot_token]
    return {'ids': ids, 'len': len(ids)}

def process_and_save(split, dataset, output_file):
    total_length = sum(dataset['len'])
    
    with open(output_file, 'wb') as f:
        for batch in tqdm(dataset.iter(batch_size=BATCH_SIZE), total=len(dataset)//BATCH_SIZE, desc=f"Processing {split}"):
            all_ids = np.concatenate(batch['ids'])
            f.write(all_ids.astype(np.uint16).tobytes())
    
    print(f"{split}.bin: {total_length} tokens, {os.path.getsize(output_file) / 1e9:.2f} GB")

if __name__ == '__main__':
    enc = tiktoken.get_encoding(ENCODING)
    
    print("Loading dataset...")
    dataset = load_dataset("openwebtext", num_proc=NUM_PROC)
    
    print("Splitting dataset...")
    split_dataset = dataset["train"].train_test_split(test_size=VAL_SIZE, seed=SEED, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    
    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        encode_text,
        remove_columns=['text'],
        num_proc=NUM_PROC,
        desc="Tokenizing",
    )
    
    print("Processing and saving splits...")
    for split, dset in tokenized.items():
        output_file = f'{split}.bin'
        process_and_save(split, dset, output_file)
    
    print("Done!")