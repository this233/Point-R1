# PartNeXt QA 训练指南

本指南介绍如何使用 PartNeXt QA 数据集训练 PointLLM 模型，包括 SFT（监督微调）和 RL（强化学习）两个阶段。

## 概述

### 数据格式

PartNeXt QA 数据集包含四种类型的问答：

| QA 类型 | 输入 | 说明 |
|---------|------|------|
| `object_identity` | 点云 + prompt | 识别整个物体是什么 |
| `part_identity` | 点云（带高亮）+ prompt | 识别高亮部件是什么 |
| `part_count` | 点云 + prompt | 统计某类部件的数量 |
| `other` | 点云 + prompt | 其他关于物体的问题 |

**关于 `part_identity` 的高亮蒙版：**
- 目标部件的点会被染成随机颜色（红/绿/蓝/黄等）
- 其他点会被调暗（dim_factor=0.4）
- prompt 中会说明高亮颜色

## 快速开始

### 1. 生成 QA 数据

```bash
cd Point-R1-data

# 批量生成 QA（使用 MLLM）
python batch_partnext_qa.py \
    --partnext_glb_dir /path/to/PartNeXt_mesh/glbs \
    --partnext_ann_dir /path/to/PartNeXt_data \
    --output_dir outputs/partnext_qa_batch \
    --num_workers 4 \
    --generate_qa
```

### 2. 转换为 PointLLM 格式

```bash
# 转换数据格式（添加高亮蒙版）
python convert_partnext_qa_to_pointllm.py \
    --qa_jsonl outputs/partnext_qa_batch/merged_qa.jsonl \
    --partnext_glb_dir /path/to/PartNeXt_mesh/glbs \
    --partnext_ann_dir /path/to/PartNeXt_data \
    --output_dir outputs/pointllm_format \
    --pointnum 8192 \
    --use_color
```

输出结构：
```
outputs/pointllm_format/
├── pointclouds/                    # 点云 .npy 文件
│   ├── {object_id}_8192.npy       # 普通点云
│   └── {object_id}_part_{node_id}_8192.npy  # 带高亮的点云
└── anno_data/
    ├── partnext_qa_train.json      # 训练集标注
    ├── partnext_qa_val.json        # 验证集标注
    └── partnext_qa_all.json        # 全部标注
```

### 3. 设置数据软链接

```bash
cd Point-R1

# 创建数据目录软链接
mkdir -p data
ln -s /path/to/outputs/pointllm_format/pointclouds data/partnext_data

# 复制标注文件
cp /path/to/outputs/pointllm_format/anno_data/* data/anno_data/
```

### 4. SFT 训练

```bash
cd Point-R1

# Stage 1: 对齐训练（只训练 point_proj）
STAGE=1 bash scripts/PointLLM_train_partnext_sft.sh

# Stage 2: 指令微调（训练 point_proj + LLM LoRA）
STAGE=2 bash scripts/PointLLM_train_partnext_sft.sh
```

**SFT 训练参数说明：**

| 参数 | Stage 1 | Stage 2 | 说明 |
|------|---------|---------|------|
| `--stage` | 1 | 2 | 训练阶段 |
| `--llm_train_type` | fix | lora | LLM 训练方式 |
| `--train_point_proj` | True | True | 是否训练点云投影层 |
| `--train_norm` | False | True | 是否训练归一化层 |
| `--learning_rate` | 3e-4 | 1e-4 | 学习率 |
| `--lora_r` | - | 128 | LoRA 秩 |
| `--lora_alpha` | - | 256 | LoRA alpha |

### 5. RL 训练 (GRPO)

```bash
cd Point-R1

# 首先将 JSON 转为 JSONL 格式
python -c "
import json
with open('data/anno_data/partnext_qa_train.json') as f:
    data = json.load(f)
with open('data/anno_data/partnext_qa_train.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
"

# 运行 GRPO 训练
bash run_scripts/run_grpo_partnext.sh
```

**GRPO 训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--beta` | 0.04 | KL 散度权重 |
| `--num_generations` | 4 | 每个 prompt 生成的样本数 |
| `--reward_funcs` | accuracy format | 使用的奖励函数 |

## 奖励函数

GRPO 训练使用以下奖励函数：

### 1. 准确性奖励 (`accuracy`)
- 物体/部件识别：模糊字符串匹配
- 部件计数：数值精确匹配
- 其他问题：模糊匹配

### 2. 格式奖励 (`format`)
- 检查是否符合 `<think>...</think><answer>...</answer>` 格式

### 3. 思考奖励 (`partnext_think`)
- 评估思考过程的质量和长度

### 4. 综合奖励 (`partnext_combined`)
- 准确性 60% + 格式 20% + 思考 20%

## 高亮蒙版说明

对于 `part_identity` 类型的 QA：

1. **数据生成阶段**：使用 `convert_partnext_qa_to_pointllm.py` 时，会为每个部件生成带高亮的点云文件

2. **点云高亮方法**：
   - 目标部件点：颜色混合（原色 30% + 高亮色 70%）
   - 其他点：亮度调暗（乘以 0.4）

3. **Prompt 说明**：
   ```
   <point>
   Note: The target part is highlighted in red color in the point cloud.
   What is this part?
   ```

4. **颜色选择**：
   - 根据 `object_id` 和 `node_id` 的 hash 值选择颜色
   - 保证同一部件每次使用相同颜色

## 数据格式详解

### 标注文件格式 (JSON)

```json
{
    "object_id": "xxx_part_yyy",
    "conversations": [
        {"from": "human", "value": "<point>\nNote: The target part is highlighted in red color.\nWhat is this part?"},
        {"from": "gpt", "value": "<think>思考过程...</think><answer>部件名称和描述</answer>"}
    ],
    "conversation_type": "partnext_part_identity",
    "meta": {
        "original_object_id": "xxx",
        "qa_type": "part_identity",
        "node_id": "yyy",
        "node_name": "部件名称"
    }
}
```

### 点云文件格式 (.npy)

- 形状：`(N, 6)` 其中 N 通常为 8192
- 前 3 列：XYZ 坐标（已归一化到单位球）
- 后 3 列：RGB 颜色（范围 0-1）

## 常见问题

### Q: 如何只训练特定类型的 QA？

修改训练脚本中的 `--conversation_types` 参数：

```bash
--conversation_types "partnext_object_identity"  # 只训练物体识别
--conversation_types "partnext_part_identity"     # 只训练部件识别
```

### Q: 显存不足怎么办？

1. 减小 `--per_device_train_batch_size`
2. 增大 `--gradient_accumulation_steps`
3. 使用更小的 LoRA 秩（`--lora_r 64`）
4. 使用 DeepSpeed ZeRO-3

### Q: 如何评估模型？

```bash
cd Point-R1

# 使用 PointLLM 评估脚本
python pointllm/eval/eval_partnext.py \
    --model_path outputs/PointLLM_partnext_sft/xxx \
    --data_path data/partnext_data \
    --anno_path data/anno_data/partnext_qa_val.json
```

### Q: 如何添加自定义奖励函数？

1. 在 `src/open-r1-multimodal/src/open_r1/partnext_rewards.py` 中定义新函数
2. 添加到 `PARTNEXT_REWARD_FUNCS` 字典
3. 在 `grpo_jsonl.py` 中注册

## 文件结构

```
Point-R1/
├── Point-R1-data/
│   ├── batch_partnext_qa.py           # QA 生成脚本
│   ├── convert_partnext_qa_to_pointllm.py  # 格式转换脚本
│   └── README_PARTNEXT_QA.md          # 本文档
├── Point-R1/
│   └── scripts/
│       └── PointLLM_train_partnext_sft.sh  # SFT 训练脚本
├── run_scripts/
│   └── run_grpo_partnext.sh           # GRPO 训练脚本
└── src/open-r1-multimodal/src/open_r1/
    ├── data/
    │   └── partnext_dataset.py        # 数据集类
    └── partnext_rewards.py            # 奖励函数
```

## 引用

如果您使用了本工具，请引用：

```bibtex
@misc{partnext_qa,
    title={PartNeXt QA: Part-aware 3D Object Understanding with Point Cloud},
    year={2025}
}
```


