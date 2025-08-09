import argparse
import torch
from torch.utils.data import DataLoader
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from pointllm.eval.evaluator import start_evaluation

from peft import PeftModel
import os
import json

# 预定义的提示词列表，用于不同的评估任务
# Predefined prompt lists for different evaluation tasks
PROMPT_LISTS = [
    "What is this?",                        # 用于分类任务的简单问题
    "This is an object of ",                # 用于分类任务的填空形式
    "Caption this 3D model in detail."     # 用于描述生成任务的详细描述
]

def init_model(args):
    """
    初始化模型、分词器和对话模板
    Initialize model, tokenizer, and conversation template
    
    Args:
        args: 包含模型名称等参数的命令行参数
        
    Returns:
        tuple: (model, tokenizer, conv) - 初始化好的模型、分词器和对话模板
    """
    # 禁用torch的初始化以节省内存
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # 打印模型名称（获取基本名称）
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载模型，使用bfloat16精度以节省显存
    # model = Point_R1ForCausalLM.from_pretrained(
    #     model_name, 
    #     low_cpu_mem_usage=False, 
    #     use_cache=True, 
    #     torch_dtype=torch.bfloat16
    # ).cuda()
    
    # # 初始化分词器和点云骨干网络配置
    # model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model = Point_R1ForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True).cuda()
    # if "PointLLM_train_stage1" not in model_name:
    #     model = PeftModel.from_pretrained(model, model_name)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    # 设置对话模式为vicuna_v1_1
    # conv_mode = "vicuna_v1_1"
    # conv = conv_templates[conv_mode].copy()
    conv = []

    return model, tokenizer, conv

def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    """
    加载验证数据集
    Load validation dataset
    
    Args:
        data_path: 数据路径
        anno_path: 注释文件路径
        pointnum: 点云中点的数量
        conversation_types: 对话类型
        use_color: 是否使用颜色信息
        
    Returns:
        ObjectPointCloudDataset: 加载的数据集对象
    """
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None  # 只加载点云数据，不进行分词
    )
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    """
    创建数据加载器
    Create data loader
    
    Args:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否随机打乱数据
        num_workers: 数据加载的进程数
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    """
    生成模型输出
    Generate model outputs
    
    Args:
        model: 预训练模型
        tokenizer: 分词器
        input_ids: 输入的token ID张量
        point_clouds: 点云数据张量
        stopping_criteria: 停止条件
        do_sample: 是否进行采样
        temperature: 温度参数，控制生成的随机性
        top_k: Top-k采样参数
        max_length: 最大生成长度
        top_p: Top-p采样参数
        
    Returns:
        list: 生成的输出文本列表
    """
    model.eval()  # 设置模型为评估模式
    with torch.inference_mode():  # 使用推理模式以节省显存
        # print(input_ids)
        output_ids = model.generate(
            input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p
        )  # 生成输出，形状为 B, L'

    # 计算输入token的长度
    input_token_len = input_ids.shape[1]
    
    # 检查输入和输出是否一致（调试信息）
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    
    # 解码输出，只保留新生成的部分
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]  # 去除首尾空白

    return outputs

def start_generation(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):
    """
    开始生成过程
    Start the generation process
    
    Args:
        model: 预训练模型
        tokenizer: 分词器
        conv: 对话模板
        dataloader: 数据加载器
        annos: 注释数据字典
        prompt_index: 提示词索引
        output_dir: 输出目录
        output_file: 输出文件名
        
    Returns:
        dict: 包含所有结果的字典
    """
    # 获取停止字符串
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    # 根据提示词索引选择问题
    qs = PROMPT_LISTS[prompt_index]

    # 初始化结果字典
    results = {"prompt": qs}

    # 获取点云骨干网络的配置
    point_backbone_config = model.get_model().language_model.point_backbone_config
    point_token_len = point_backbone_config['point_token_len']-2
    # DEBUG
    # point_token_len = 1+64
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    print(default_point_patch_token,default_point_start_token,default_point_end_token,mm_use_point_start_end)
    # 根据配置构建包含点云token的问题
    if mm_use_point_start_end:
        # 使用开始和结束token包围点云patch token
        qs = "<|vision_start|>"+default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token +"<|vision_end|>" '\n' + qs
    else:
        # 只使用点云patch token
        qs = default_point_patch_token * point_token_len + '\n' + qs
    print(qs)
    # 构建对话
    # conv.append_message(conv.roles[0], qs)  # 用户消息
    # conv.append_message(conv.roles[1], None)  # 助手消息（待生成）
    conv.append({"role": "user", "content": qs})
    # conv.append({"role": "assistant", "content": ""})

    # 获取完整的提示词
    # prompt = conv.get_prompt()
    # print("!!!",conv)
    prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt])

    # 转换为张量并移到GPU
    input_ids_ = torch.as_tensor(inputs.input_ids).cuda()  # 形状为 1, L

    # 创建停止条件
    # stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    # 遍历数据加载器中的每个批次
    for batch in tqdm(dataloader):
        # 获取点云数据并转换为模型数据类型
        point_clouds = batch["point_clouds"].cuda().to(model.dtype)  # 形状为 B, N, C(3)
        object_ids = batch["object_ids"]  # 对象ID列表

        batchsize = len(object_ids)
        # print("!!!$$",batchsize)

        # 复制input_ids以匹配批次大小
        input_ids = input_ids_.repeat(batchsize, 1)  # 形状为 B, L

        # 生成输出
        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds)

        # 保存结果
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output
            })
    
    results["results"] = responses

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 将结果保存到JSON文件
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # 打印保存信息
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    """
    主函数，orchestrates整个评估过程
    Main function that orchestrates the entire evaluation process
    
    Args:
        args: 命令行参数
    """
    # 设置输出目录
    args.output_dir = os.path.join(args.model_name, "evaluation")
    
    # 设置输出文件名
    anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
    args.output_file = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # 首先进行推理，然后评估
    if not os.path.exists(args.output_file_path):
        # 需要进行推理
        print(f'[INFO] Output file does not exist, starting inference...')
        
        # 加载注释文件
        with open(args.anno_path, 'r') as fp:
            annos = json.load(fp)

        # 加载数据集
        dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
        
        # 初始化模型
        model, tokenizer, conv = init_model(args)

        # 将注释文件从列表格式转换为字典格式
        # 从 [{"object_id": ...}] 转换为 {"object_id": ...}
        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)

        # 释放模型和分词器，清空CUDA内存
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # 直接加载已存在的结果
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    # 如果需要开始评估
    if args.start_eval:
        # 生成评估后的输出文件名
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        
        # 任务类型映射
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        
        # 开始评估
        start_evaluation(
            results, 
            output_dir=args.output_dir, 
            output_file=evaluated_output_file, 
            eval_type=eval_type_mapping[args.task_type], 
            model_type=args.gpt_type, 
            parallel=True, 
            num_workers=20
        )

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, 
        default="./outputs/PointLLM_train_stage3/PointLLM_train_stage3",
        help="预训练模型的名称或路径")

    # 数据集相关参数
    parser.add_argument("--data_path", type=str, 
        default="../data/objaverse_data", required=False,
        help="点云数据的路径")
    parser.add_argument("--anno_path", type=str, 
        default="../data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False,
        help="注释数据的路径")
    parser.add_argument("--pointnum", type=int, default=8192,
        help="每个点云中的点数量")
    parser.add_argument("--use_color", action="store_true", default=True,
        help="是否使用点云的颜色信息")

    # 数据加载器相关参数
    parser.add_argument("--batch_size", type=int, default=6,
        help="批次大小")
    parser.add_argument("--shuffle", type=bool, default=False,
        help="是否随机打乱数据")
    parser.add_argument("--num_workers", type=int, default=10,
        help="数据加载的进程数")

    # 评估相关参数
    parser.add_argument("--prompt_index", type=int, default=0,
        help="提示词索引（0-2）")
    parser.add_argument("--start_eval", action="store_true", default=False,
        help="是否在生成后立即开始评估")
    parser.add_argument("--gpt_type", type=str, default="gpt-4.1", 
        choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4.1"],
        help="用于评估的GPT模型类型")
    parser.add_argument("--task_type", type=str, default="classification", 
        choices=["captioning", "classification"],
        help="评估任务类型")

    args = parser.parse_args()

    # 检查提示词索引是否与任务类型匹配
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)