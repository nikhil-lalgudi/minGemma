# evaluate the gemma-2b instruct model
# 2.51 billion parameters
batch_size = 8 # 8 GPUS
eval_iters = 1000
wandb_log = True
initialization = '2b-instruct-full'