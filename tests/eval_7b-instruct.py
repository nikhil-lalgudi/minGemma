# evaluate the ENTIRE gemma-7b instruct model
# 8.54 billion parameters
batch_size = 8 # 8 GPUS
eval_iters = 1000
wandb_log = True
initialization = '7b-instruct-full'