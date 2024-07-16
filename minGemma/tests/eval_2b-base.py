# evaluate the ENTIRE base gemma-7b model
# 2.51 billion parameters
batch_size = 8 # 8 GPUS
eval_iters = 1000
wandb_log = True
initialization = '2b-base-full'