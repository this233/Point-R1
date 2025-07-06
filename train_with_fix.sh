#!/bin/bash

# 设置环境变量来修复CUDA libcufile链接问题
export LD_PRELOAD="/lib/x86_64-linux-gnu/libdl.so.2:$LD_PRELOAD"

# 可选：设置CUDA相关环境变量
export CUDA_HOME="/data/liweihong/cuda-11.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 运行训练命令
python -m pointllm.train.train "$@" 