# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -O flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
# pip install flash-attn --no-build-isolation
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]

pip install json_repair
pip install peft