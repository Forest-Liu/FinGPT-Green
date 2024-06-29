export CUDA_VISIBLE_DEVICES=0
deepspeed train_lora.py > train.log 2>&1