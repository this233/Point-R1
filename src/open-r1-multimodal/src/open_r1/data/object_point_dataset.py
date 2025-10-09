import os  # 导入操作系统接口模块，用于文件路径操作
import json  # 导入JSON处理模块，用于读写JSON格式文件
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy数值计算库，简称为np

import copy  # 导入copy模块，用于对象的深拷贝和浅拷贝
import transformers  # 导入transformers库，用于预训练模型和分词器
from torch.utils.data import Dataset  # 从PyTorch工具包导入Dataset基类

from .utils import *  # 从当前包的utils模块导入所有函数和类


def make_object_point_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:  # 定义函数创建点云数据模块，接收分词器和数据参数
    """Make dataset and collator for Joint3Ddataset with text and point cloud data."""  # 函数文档字符串：为联合3D数据集创建数据集和数据整理器
    """Initialize datasets."""  # 初始化数据集

    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)  # 创建点云文本数据集的数据整理器
    if data_args.split_train_val:  # 如果需要分割训练集和验证集
        print("Loading training datasets.")  # 打印加载训练数据集信息
        train_dataset = ObjectPointCloudDataset(  # 创建训练数据集对象
            split='train',  # 设置数据集分割类型为训练
            data_path=data_args.data_path,  # 设置数据路径
            anno_path=data_args.anno_path,  # 设置标注文件路径
            pointnum=data_args.pointnum,  # 设置点云中点的数量
            conversation_types=data_args.conversation_types,  # 设置对话类型
            tokenizer=tokenizer,  # 传入分词器
            use_color=data_args.use_color,  # 设置是否使用颜色信息
            data_args=data_args  # 传入数据参数
        )
        print("Done!")  # 打印完成信息
        if data_args.data_debug_num > 0:  # 如果设置了调试数据数量
            print('Debug mode, using training set as val set.')  # 打印调试模式信息，使用训练集作为验证集
            val_dataset = train_dataset  # 将训练集赋值给验证集
        else:  # 否则创建独立的验证集
            # * make a val dataset  # 创建验证数据集
            print("Loading validation datasets.")  # 打印加载验证数据集信息
            val_dataset = ObjectPointCloudDataset(  # 创建验证数据集对象
                split='val', # * load train split  # 设置分割类型为验证，但实际加载训练分割
                data_path=data_args.data_path,  # 设置数据路径
                anno_path=data_args.anno_path,  # 设置标注文件路径
                pointnum=data_args.pointnum,  # 设置点云中点的数量
                conversation_types=data_args.conversation_types,  # 设置对话类型
                tokenizer=tokenizer,  # 传入分词器
                use_color=data_args.use_color,  # 设置是否使用颜色信息
                data_args=data_args  # 传入数据参数
            )
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)  # 返回包含训练集、验证集和数据整理器的字典
    else:  # 如果不分割训练集和验证集
        # * use all data as training data  # 使用所有数据作为训练数据（走这条路）
        train_dataset = ObjectPointCloudDataset(  # 创建训练数据集对象
            split='train',  # 设置数据集分割类型为训练
            data_path=data_args.data_path,  # 设置数据路径
            anno_path=data_args.anno_path,  # 设置标注文件路径
            pointnum=data_args.pointnum,  # 设置点云中点的数量
            conversation_types=data_args.conversation_types,  # 设置对话类型
            use_color=data_args.use_color,  # 设置是否使用颜色信息
            tokenizer=tokenizer,  # 传入分词器
            data_args=data_args  # 传入数据参数
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)  # 返回包含训练集（验证集为None）和数据整理器的字典

class ObjectPointCloudDataset(Dataset):  # 定义点云数据集类，继承自PyTorch的Dataset基类
    """Dataset utilities for objaverse."""  # 类文档字符串：用于objaverse的数据集工具
    def __init__(self,  # 定义初始化方法
                 data_path=None,  # 数据路径参数，默认为None
                 anno_path=None,  # 标注文件路径参数，默认为None
                 tokenizer=None,  # 分词器参数，默认为None
                 pointnum=8192,  # 点云中点的数量，默认为8192
                 split='train',  # 数据集分割类型，默认为训练集
                 conversation_types=None, # * default is simple_des, used for stage1 pre-train  # 对话类型，默认为简单描述，用于第一阶段预训练
                 use_color=True,  # 是否使用颜色信息，默认为True
                 data_args=None):  # 数据参数，默认为None

        """
        split: only considered when data_args.split_train_val is True.  # split参数：仅在data_args.split_train_val为True时考虑
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:  # conversation_types参数：元组，用于过滤数据，默认为简单描述
            "detailed_description", "single_round", "multi_round".  # 其他类型包括：详细描述、单轮对话、多轮对话
        tokenizer: load point clouds only if None  # tokenizer参数：如果为None则仅加载点云
        """
        super(ObjectPointCloudDataset, self).__init__()  # 调用父类Dataset的初始化方法

        """Initialize dataset with object point clouds and text"""  # 使用对象点云和文本初始化数据集
        self.data_path = data_path  # 保存数据路径
        self.anno_path = anno_path  # 保存标注文件路径
        self.tokenizer = tokenizer  # 保存分词器
        self.split = split   # 保存数据集分割类型
        if conversation_types is None:  # 如果对话类型为None
            self.conversation_types = ("simple_description",)  # 设置默认对话类型为简单描述
        else:  # 否则
            self.conversation_types = conversation_types  # 使用传入的对话类型

        self.data_args = data_args  # 保存数据参数
        self.normalize_pc = True  # 设置点云标准化标志为True
        self.use_color = use_color  # 保存是否使用颜色信息的标志

        self.pointnum = pointnum  # 保存点云中点的数量
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None  # 如果数据参数存在则获取点云骨干网络配置，否则为None
        self.point_indicator = '<point>'  # 设置点云指示符为'<point>'

        # Load the data list from JSON  # 从JSON文件加载数据列表
        print(f"Loading anno file from {anno_path}.")  # 打印加载标注文件的信息
        with open(anno_path, "r") as json_file:  # 打开标注文件
            self.list_data_dict = json.load(json_file)  # 从JSON文件加载数据字典列表
        
        # * print the conversations_type  # 打印对话类型
        print(f"Using conversation_type: {self.conversation_types}")   # 打印使用的对话类型
        # * print before filtering  # 打印过滤前的信息
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")  # 打印过滤前数据集的大小

        # * iterate the list and filter  # 遍历列表并过滤
        # * these two ids have corrupted colored point files, so filter them when use_color is True  # 这两个ID的彩色点文件已损坏，所以当use_color为True时过滤它们
        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []  # 如果使用颜色则设置过滤ID列表，否则为空列表

        # Iterate the list, filter those "converation_type" not in self.conversation_types  # 遍历列表，过滤那些不在self.conversation_types中的对话类型
        self.list_data_dict = [
            data for data in self.list_data_dict
            if data.get('conversation_type', 'simple_description') in self.conversation_types
            and data.get('object_id') not in filter_ids
            # and (
            #     isinstance(data.get("conversations"), list)
            #     and len(data["conversations"]) > 1
            #     and isinstance(data["conversations"][1], dict)
            #     and isinstance(data["conversations"][1].get("value", ""), str)
            #     and len(data["conversations"][1]["value"].split()) >= 10
            # )
        ]

        

        # * print after filtering  # 打印过滤后的信息
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")  # 打印过滤后数据集的大小
        # * print the size of different conversation_type  # 打印不同对话类型的数量
        for conversation_type in self.conversation_types:  # 遍历每种对话类型
            print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")  # 打印每种对话类型的数量

        if self.data_args is not None and self.data_args.data_debug_num > 0:  # 如果数据参数存在且设置了调试数据数量
            self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]  # 截取指定数量的数据用于调试
            # * print all the scan_id in debug mode, not using for loop  # 在调试模式下打印所有扫描ID，不使用for循环
            print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))  # 打印调试模式使用的对象ID
        elif self.data_args is not None and self.data_args.split_train_val:  # 否则如果数据参数存在且需要分割训练验证集
            # * split train and val with 9:1 ratios  # 按照9:1的比例分割训练集和验证集
            if self.split == 'train':  # 如果是训练集
                self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]  # 取前split_ratio比例的数据作为训练集
                print(f"Train set size: {len(self.list_data_dict)}")  # 打印训练集大小
            else:  # 如果是验证集
                self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]  # 取后面部分数据作为验证集
                print(f"Val set size: {len(self.list_data_dict)}")  # 打印验证集大小

    @staticmethod
    def get_question_template(task_type: str = "odLength"):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    def _load_point_cloud(self, object_id, type='objaverse'):  # 定义加载点云的私有方法，接收对象ID和类型参数
        if type == 'objaverse':  # 如果类型为objaverse
            return self._load_objaverse_point_cloud(object_id)   # 调用加载objaverse点云的方法

    def _load_objaverse_point_cloud(self, object_id):  # 定义加载objaverse点云的私有方法
        filename = f"{object_id}_{self.pointnum}.npy"  # 构造点云文件名，格式为对象ID_点数.npy
        point_cloud = np.load(os.path.join(self.data_path, filename))  # 从指定路径加载点云数据

        if not self.use_color:  # 如果不使用颜色信息
            point_cloud = point_cloud[:, :3]  # 只保留前三列（xyz坐标）

        return point_cloud  # 返回点云数据

    def pc_norm(self, pc):  # 定义点云标准化方法
        """ pc: NxC, return NxC """  # 方法文档：输入NxC的点云，返回NxC的标准化点云
        xyz = pc[:, :3]  # 提取xyz坐标（前三列）
        other_feature = pc[:, 3:]  # 提取其他特征（第四列及以后）

        centroid = np.mean(xyz, axis=0)  # 计算xyz坐标的质心（中心点）
        xyz = xyz - centroid  # 将xyz坐标减去质心，实现平移到原点
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))  # 计算点云中距离原点最远的点的距离
        xyz = xyz / m  # 将xyz坐标除以最大距离，实现缩放到单位球内

        pc = np.concatenate((xyz, other_feature), axis=1)  # 将标准化的xyz坐标与其他特征拼接
        return pc  # 返回标准化后的点云
    
    def __getitem__(self, index):  # 定义获取数据项的方法，实现Dataset接口
        sources = self.list_data_dict[index]  # 根据索引获取数据源
        if isinstance(index, int):  # 如果索引是整数
            sources = [sources]  # 将数据源包装成列表
        assert len(sources) == 1, "sources should be a list"  # 断言数据源列表长度为1
        
        object_id = sources[0]['object_id']

        # Get conversation data
        conversation = sources[0]['conversations']
        # Extract problem and solution from conversations
        assert len(conversation)==2, "conversation should be a list of length 2"
        problem = conversation[0]['value'] #.replace('<point>', '').strip()
        solution = conversation[1]['value']
        
        point_cloud_path = os.path.join(self.data_path, f"{object_id}_{self.pointnum}.npy")
        
        # # Default accu_reward_method
        # accu_reward_method = sources[0].get('accu_reward_method', 'default')
        
        # Create prompt structure for GRPO trainer
        content = []
        # content.append({'type': 'point'})
        content.append({'type': 'text', 'text': self.get_question_template().format(Question=problem)})
        
        point_cloud = self._load_point_cloud(object_id) # * N, C  # 加载点云数据，形状为N×C
        if self.normalize_pc:  # 如果需要标准化点云
            point_cloud = self.pc_norm(point_cloud) # * need to norm since point encoder is norm  # 标准化点云，因为点云编码器需要标准化数据

        return {
            'object_id': object_id,
            'point_cloud_path': point_cloud_path,
            'problem': problem,
            'solution': f"<answer> {solution} </answer>" if solution else "",
            'ground_truth': f"<answer> {solution} </answer>" if solution else "",
            # 'accu_reward_method': accu_reward_method,
            'prompt': [{
                'role': 'system',
                'content': "You are an expert in 3D point cloud analysis. Your task is to analyze point cloud data and provide clear, concise answer to the question.\n\nWhen analyzing a point cloud, you should:\n- Identify the main object or structure represented by the points\n- Describe its key visual characteristics (shape, color, size, notable features)\n- Use simple, direct language that anyone can understand\n- Keep descriptions brief but informative (typically 1-2 sentences)\n\nYou can encounter various types of objects including but not limited to: vehicles, buildings, furniture, animals, characters, tools, electronic devices, and abstract shapes. Focus on what the object IS and its most distinctive visual features.\n"
                },{
                'role': 'user',
                'content': content
            }],
            'point_cloud': torch.from_numpy(point_cloud.astype(np.float32))
        }
        
        # Original processing for training (with tokenizer)
        if self.point_indicator in sources[0]['conversations'][0]['value']:
            # Point cloud representation  # 点云表示
            
            sources = preprocess_multimodal_point_cloud(  # 预处理多模态点云数据
                copy.deepcopy([e["conversations"] for e in sources]), self.point_backbone_config, point_indicator=self.point_indicator)  # 深拷贝对话数据并处理
        else:  # 如果不包含点云指示符
            sources = copy.deepcopy([e["conversations"] for e in sources])  # 深拷贝对话数据

        data_dict = preprocess_v2(  # 使用v2版本预处理函数
            sources,  # 传入数据源
            self.tokenizer)  # 传入分词器

        if isinstance(index, int):  # 如果索引是整数
            data_dict = dict(input_ids=data_dict["input_ids"][0],  # 提取第一个输入ID序列
                             labels=data_dict["labels"][0])  # 提取第一个标签序列

        # point exist in the data  # 如果数据中存在点云
        if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:  # 检查第一个对话是否包含点云指示符
            data_dict['point_clouds'] = torch.from_numpy(point_cloud.astype(np.float32))  # 将点云数据添加到数据字典中

        return data_dict  # 返回数据字典

    def __len__(self):  # 定义获取数据集长度的方法，实现Dataset接口
        """Return number of utterances."""  # 方法文档：返回话语数量
        return len(self.list_data_dict)  # 返回数据字典列表的长度

if __name__ == '__main__':  # 如果作为主程序运行
    import argparse  # 导入命令行参数解析模块
    parser = argparse.ArgumentParser()  # 创建参数解析器

    parser.add_argument("--data_path", default="data/objaverse_data", type=str,  # 添加数据路径参数
                        help="Path to the data directory.")  # 参数帮助信息：数据目录路径
    parser.add_argument("--anno_path", default=None, type=str, required=True,  # 添加标注文件路径参数，必需
                        help="Path to the annotation file.")  # 参数帮助信息：标注文件路径
    parser.add_argument("--split", default='train', type=str,   # 添加数据集分割参数
                        help="Whether to use the train or validation dataset.")  # 参数帮助信息：使用训练集还是验证集
    parser.add_argument("--pointnum", default=8192, type=int,  # 添加点数参数
                        help="Number of points in the point cloud.")  # 参数帮助信息：点云中点的数量
    parser.add_argument("--data_debug_num", default=0, type=int,  # 添加调试数据数量参数
                        help="Number of data to debug with.")  # 参数帮助信息：用于调试的数据数量
    parser.add_argument("--split_train_val", default=False, type=bool,  # 添加是否分割训练验证集参数
                        help="Whether to split the dataset into training and validation.")  # 参数帮助信息：是否将数据集分割为训练集和验证集
    parser.add_argument("--split_ratio", default=0.9, type=float,  # 添加分割比例参数
                        help="The ratio of training to validation data.")  # 参数帮助信息：训练数据与验证数据的比例
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True,  # 添加分词器路径参数，必需
                        help="Path to the tokenizer config file.")  # 参数帮助信息：分词器配置文件路径
    
    args = parser.parse_args()  # 解析命令行参数

    # Initialize tokenizer  # 初始化分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)  # 从预训练模型加载分词器

    args.point_backbone_config = None  # 设置点云骨干网络配置为None

    # Initialize dataset  # 初始化数据集
    dataset = ObjectPointCloudDataset(  # 创建点云数据集对象
        data_path=args.data_path,  # 传入数据路径
        anno_path=args.anno_path,  # 传入标注文件路径
        pointnum=args.pointnum,  # 传入点数
        split=args.split,  # 传入数据集分割类型
        tokenizer=tokenizer,  # 传入分词器
        data_args=args  # 传入参数对象
    )

    # Example usage  # 示例用法
    print(f'Dataset length: {len(dataset)}')  # 打印数据集长度

