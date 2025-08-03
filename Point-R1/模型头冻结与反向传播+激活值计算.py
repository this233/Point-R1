import torch
import torch.nn as nn
import torch.optim as optim
import gc
import psutil
import os

# 定义一个两层MLP，简单实现一个神经网络
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

def get_memory_info():
    """获取GPU和CPU内存信息"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        gpu_cached = torch.cuda.memory_reserved() / 1024**2   # MB
        print(f"GPU显存: 已分配={gpu_memory:.2f}MB, 已缓存={gpu_cached:.2f}MB")
    
    # CPU内存
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**2  # MB
    print(f"CPU内存: {cpu_memory:.2f}MB")

def analyze_tensor_memory(tensor, name):
    """分析张量的内存占用"""
    element_size = tensor.element_size()  # 每个元素的字节数
    numel = tensor.numel()  # 元素总数
    memory_mb = (element_size * numel) / 1024**2
    print(f"{name}: 形状={tensor.shape}, 内存={memory_mb:.4f}MB, requires_grad={tensor.requires_grad}")
    return memory_mb

def analyze_grad_fn_details(tensor, name, depth=0):
    """详细分析grad_fn链"""
    indent = "  " * depth
    print(f"{indent}{name}:")
    print(f"{indent}  is_leaf: {tensor.is_leaf}")
    print(f"{indent}  requires_grad: {tensor.requires_grad}")
    print(f"{indent}  grad_fn: {tensor.grad_fn}")
    
    if tensor.grad_fn is not None:
        print(f"{indent}  grad_fn类型: {type(tensor.grad_fn).__name__}")
        
        # 获取next_functions（前驱节点）
        if hasattr(tensor.grad_fn, 'next_functions'):
            next_funcs = tensor.grad_fn.next_functions
            print(f"{indent}  前驱节点数量: {len(next_funcs)}")
            for i, (func, idx) in enumerate(next_funcs):
                if func is not None:
                    print(f"{indent}    前驱{i}: {type(func).__name__} (输入索引: {idx})")
                else:
                    print(f"{indent}    前驱{i}: None")
        
        # 获取variable（如果是叶子节点的AccumulateGrad）
        if hasattr(tensor.grad_fn, 'variable'):
            var = tensor.grad_fn.variable
            if var is not None:
                print(f"{indent}  关联变量形状: {var.shape}")
    
    print()

def visualize_computation_graph():
    """可视化计算图的构建过程"""
    print("=== 计算图构建过程 ===")
    
    # 1. 创建输入张量（叶子节点）
    x = torch.randn(4, 3, requires_grad=True)
    analyze_grad_fn_details(x, "输入张量 x", 0)
    
    # 2. 第一个操作：线性变换
    W1 = torch.randn(3, 5, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)
    analyze_grad_fn_details(W1, "权重 W1", 0)
    analyze_grad_fn_details(b1, "偏置 b1", 0)
    
    z1 = torch.mm(x, W1) + b1  # 矩阵乘法 + 加法
    analyze_grad_fn_details(z1, "第一层输出 z1", 0)
    
    # 3. 激活函数
    a1 = torch.relu(z1)
    analyze_grad_fn_details(a1, "激活后 a1", 0)
    
    # 4. 第二个操作
    W2 = torch.randn(5, 2, requires_grad=True)
    b2 = torch.randn(2, requires_grad=True)
    analyze_grad_fn_details(W2, "权重 W2", 0)
    analyze_grad_fn_details(b2, "偏置 b2", 0)
    
    z2 = torch.mm(a1, W2) + b2
    analyze_grad_fn_details(z2, "第二层输出 z2", 0)
    
    # 5. 损失函数
    target = torch.randn(4, 2)
    diff = z2 - target
    analyze_grad_fn_details(diff, "差值 (z2 - target)", 0)
    
    squared = diff ** 2
    analyze_grad_fn_details(squared, "平方 (diff ** 2)", 0)
    
    loss = torch.mean(squared)
    analyze_grad_fn_details(loss, "损失 loss", 0)
    
    print("=== 计算图结构链 ===")
    print("loss -> MeanBackward -> PowBackward -> SubBackward -> AddBackward -> MmBackward -> ReLUBackward -> AddBackward -> MmBackward -> [叶子节点]")
    
    return loss, [x, W1, b1, W2, b2]

def analyze_backward_process(loss, variables):
    """分析反向传播过程"""
    print("\n=== 反向传播详细分析 ===")
    
    print("反向传播前的梯度状态:")
    for i, var in enumerate(variables):
        var_name = ["x", "W1", "b1", "W2", "b2"][i]
        print(f"{var_name}.grad: {var.grad}")
    
    print("\n开始反向传播...")
    loss.backward()
    
    print("\n反向传播后的梯度状态:")
    for i, var in enumerate(variables):
        var_name = ["x", "W1", "b1", "W2", "b2"][i]
        if var.grad is not None:
            print(f"{var_name}.grad: 形状={var.grad.shape}, 范数={var.grad.norm():.6f}")
            print(f"  梯度统计: min={var.grad.min():.6f}, max={var.grad.max():.6f}, mean={var.grad.mean():.6f}")
        else:
            print(f"{var_name}.grad: None")
    
    print("\n=== 梯度函数链分析 ===")
    # 从loss开始追踪梯度函数链
    current_fn = loss.grad_fn
    depth = 0
    visited = set()
    
    def trace_grad_fn(grad_fn, depth=0, max_depth=10):
        if grad_fn is None or depth > max_depth or id(grad_fn) in visited:
            return
        
        visited.add(id(grad_fn))
        indent = "  " * depth
        print(f"{indent}梯度函数: {type(grad_fn).__name__}")
        
        if hasattr(grad_fn, 'next_functions'):
            for i, (next_fn, input_idx) in enumerate(grad_fn.next_functions):
                if next_fn is not None:
                    print(f"{indent}  -> 输入{input_idx}: {type(next_fn).__name__}")
                    trace_grad_fn(next_fn, depth + 1, max_depth)
    
    trace_grad_fn(current_fn)

def calculate_memory_breakdown():
    """详细计算显存占用"""
    print("\n=== 显存占用详细分析 ===")
    
    batch_size = 32
    input_dim = 1024
    hidden_dim = 2048
    output_dim = 512
    
    # 1. 模型参数内存
    print("1. 模型参数内存:")
    layer1_weight = input_dim * hidden_dim * 4  # float32 = 4 bytes
    layer1_bias = hidden_dim * 4
    layer2_weight = hidden_dim * output_dim * 4
    layer2_bias = output_dim * 4
    
    total_params = layer1_weight + layer1_bias + layer2_weight + layer2_bias
    print(f"   Layer1权重: {layer1_weight/1024**2:.2f}MB")
    print(f"   Layer1偏置: {layer1_bias/1024**2:.2f}MB")
    print(f"   Layer2权重: {layer2_weight/1024**2:.2f}MB")
    print(f"   Layer2偏置: {layer2_bias/1024**2:.2f}MB")
    print(f"   参数总计: {total_params/1024**2:.2f}MB")
    
    # 2. 梯度内存（如果requires_grad=True）
    print("\n2. 梯度内存:")
    grad_memory = total_params  # 梯度与参数大小相同
    print(f"   梯度总计: {grad_memory/1024**2:.2f}MB")
    
    # 3. 前向传播中间结果
    print("\n3. 前向传播激活值:")
    input_activation = batch_size * input_dim * 4
    hidden_activation = batch_size * hidden_dim * 4
    output_activation = batch_size * output_dim * 4
    
    print(f"   输入激活: {input_activation/1024**2:.2f}MB")
    print(f"   隐藏激活: {hidden_activation/1024**2:.2f}MB")
    print(f"   输出激活: {output_activation/1024**2:.2f}MB")
    
    activation_memory = input_activation + hidden_activation + output_activation
    print(f"   激活值总计: {activation_memory/1024**2:.2f}MB")
    
    # 4. 优化器状态（以Adam为例）
    print("\n4. 优化器状态内存 (Adam):")
    # Adam需要存储一阶矩和二阶矩
    optimizer_memory = total_params * 2  # m和v状态
    print(f"   优化器状态: {optimizer_memory/1024**2:.2f}MB")
    
    # 总内存
    print(f"\n=== 总内存估算 ===")
    total_memory = total_params + grad_memory + activation_memory + optimizer_memory
    print(f"模型参数: {total_params/1024**2:.2f}MB")
    print(f"梯度: {grad_memory/1024**2:.2f}MB")
    print(f"激活值: {activation_memory/1024**2:.2f}MB")
    print(f"优化器: {optimizer_memory/1024**2:.2f}MB")
    print(f"总计: {total_memory/1024**2:.2f}MB")

def memory_optimization_demo():
    """演示显存优化技术"""
    print("\n=== 显存优化技术演示 ===")
    
    # 1. 梯度检查点（Gradient Checkpointing）
    print("1. 梯度检查点:")
    print("   - 不保存中间激活值，反向传播时重新计算")
    print("   - 用计算换内存")
    
    # 2. 梯度累积
    print("\n2. 梯度累积:")
    print("   - 小批量多次前向传播，累积梯度")
    print("   - 减少单次的激活值内存")
    
    # 3. 混合精度训练
    print("\n3. 混合精度训练:")
    print("   - 使用FP16替代FP32")
    print("   - 内存减半，但需要梯度缩放")
    
    # 4. 参数冻结
    print("\n4. 参数冻结 (requires_grad=False):")
    print("   - 不计算梯度，节省梯度内存")
    print("   - 不更新参数，节省优化器状态内存")

def detailed_backward_analysis(model, input_data, target_data):
    """详细分析反向传播过程"""
    print("\n=== 详细反向传播分析 ===")
    
    # 前向传播
    print("1. 前向传播过程:")
    output = model(input_data)
    loss = nn.MSELoss()(output, target_data)
    
    print(f"输出张量: 形状={output.shape}, grad_fn={output.grad_fn}")
    print(f"损失张量: 值={loss.item():.6f}, grad_fn={loss.grad_fn}")
    
    # 分析所有中间张量的grad_fn
    print("\n2. 模型参数的grad_fn信息:")
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(f"  形状: {param.shape}")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  is_leaf: {param.is_leaf}")
        print(f"  grad_fn: {param.grad_fn}")
        print(f"  梯度状态: {'已计算' if param.grad is not None else '未计算'}")
    
    print("\n3. 开始反向传播...")
    loss.backward()
    
    print("\n4. 反向传播后的梯度信息:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}.grad:")
            print(f"  形状: {param.grad.shape}")
            print(f"  数据类型: {param.grad.dtype}")
            print(f"  设备: {param.grad.device}")
            print(f"  梯度范数: {param.grad.norm().item():.6f}")
            print(f"  梯度统计: min={param.grad.min().item():.6f}, max={param.grad.max().item():.6f}")
            print(f"  是否包含NaN: {torch.isnan(param.grad).any().item()}")
            print(f"  是否包含Inf: {torch.isinf(param.grad).any().item()}")
        else:
            print(f"{name}.grad: None")
    
    return loss

def demonstrate_layer2_freezing():
    """专门演示layer2冻结时的计算图、梯度计算和显存占用"""
    print("=" * 80)
    print("专门演示：Layer2冻结时的计算图构建、梯度计算、显存占用")
    print("=" * 80)
    
    # 创建模型
    model = TwoLayerMLP(input_dim=512, hidden_dim=1024, output_dim=256)
    input_data = torch.randn(16, 512)
    target_data = torch.randn(16, 256)
    criterion = nn.MSELoss()
    
    def analyze_model_state(title, model):
        print(f"\n=== {title} ===")
        print("模型参数状态:")
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            print(f"  {name}:")
            print(f"    形状: {param.shape}")
            print(f"    参数数量: {param_count:,}")
            print(f"    requires_grad: {param.requires_grad}")
            print(f"    is_leaf: {param.is_leaf}")
            print(f"    内存占用: {param.numel() * param.element_size() / 1024**2:.4f}MB")
        
        print(f"\n参数统计:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  冻结参数: {total_params - trainable_params:,}")
        print(f"  可训练比例: {trainable_params/total_params*100:.1f}%")
        
        return total_params, trainable_params
    
    def analyze_forward_pass(title, model, input_data, target_data):
        print(f"\n=== {title} - 前向传播分析 ===")
        
        # 钩子函数收集中间激活值
        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # 注册钩子
        model.layer1.register_forward_hook(get_activation('layer1'))
        model.activation.register_forward_hook(get_activation('activation'))
        model.layer2.register_forward_hook(get_activation('layer2'))
        
        # 前向传播
        output = model(input_data)
        loss = criterion(output, target_data)
        
        print("中间激活值分析:")
        activation_memory = 0
        for name, activation in activations.items():
            memory_mb = activation.numel() * activation.element_size() / 1024**2
            activation_memory += memory_mb
            print(f"  {name}: 形状={activation.shape}, 内存={memory_mb:.4f}MB")
            print(f"    requires_grad: {activation.requires_grad}")
            print(f"    grad_fn: {activation.grad_fn}")
        
        print(f"\n输出分析:")
        print(f"  输出形状: {output.shape}")
        print(f"  输出requires_grad: {output.requires_grad}")
        print(f"  输出grad_fn: {output.grad_fn}")
        print(f"  损失值: {loss.item():.6f}")
        print(f"  损失grad_fn: {loss.grad_fn}")
        
        print(f"\n激活值总内存: {activation_memory:.4f}MB")
        
        return loss, output, activations
    
    def analyze_computation_graph(title, loss):
        print(f"\n=== {title} - 计算图分析 ===")
        
        def trace_grad_fn_detailed(grad_fn, depth=0, max_depth=10, visited=None):
            if visited is None:
                visited = set()
            
            if grad_fn is None or depth > max_depth or id(grad_fn) in visited:
                return
            
            visited.add(id(grad_fn))
            indent = "  " * depth
            print(f"{indent}{type(grad_fn).__name__}")
            
            if hasattr(grad_fn, 'next_functions'):
                for i, (next_fn, input_idx) in enumerate(grad_fn.next_functions):
                    if next_fn is not None:
                        print(f"{indent}  └─ 输入{input_idx}: {type(next_fn).__name__}")
                        trace_grad_fn_detailed(next_fn, depth + 1, max_depth, visited)
            
            # 特殊处理AccumulateGrad
            if hasattr(grad_fn, 'variable') and grad_fn.variable is not None:
                var = grad_fn.variable
                print(f"{indent}  └─ 变量: 形状={var.shape}, requires_grad={var.requires_grad}")
        
        print("计算图结构:")
        trace_grad_fn_detailed(loss.grad_fn)
    
    def analyze_backward_pass(title, loss, model):
        print(f"\n=== {title} - 反向传播分析 ===")
        
        # 反向传播前的状态
        print("反向传播前梯度状态:")
        for name, param in model.named_parameters():
            print(f"  {name}.grad: {param.grad}")
        
        # 执行反向传播
        print("\n执行反向传播...")
        loss.backward()
        
        # 反向传播后的状态
        print("\n反向传播后梯度状态:")
        total_grad_memory = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_memory = param.grad.numel() * param.grad.element_size() / 1024**2
                total_grad_memory += grad_memory
                print(f"  {name}.grad:")
                print(f"    形状: {param.grad.shape}")
                print(f"    内存: {grad_memory:.4f}MB")
                print(f"    范数: {param.grad.norm().item():.6f}")
                print(f"    统计: min={param.grad.min().item():.6f}, max={param.grad.max().item():.6f}")
            else:
                print(f"  {name}.grad: None (参数被冻结)")
        
        print(f"\n梯度总内存: {total_grad_memory:.4f}MB")
        return total_grad_memory
    
    def analyze_memory_usage(title, model):
        print(f"\n=== {title} - 详细内存分析 ===")
        
        # 参数内存
        param_memory = 0
        grad_memory = 0
        
        print("参数内存分布:")
        for name, param in model.named_parameters():
            param_mem = param.numel() * param.element_size() / 1024**2
            param_memory += param_mem
            print(f"  {name}: {param_mem:.4f}MB")
            
            if param.requires_grad and param.grad is not None:
                grad_mem = param.grad.numel() * param.grad.element_size() / 1024**2
                grad_memory += grad_mem
                print(f"    对应梯度: {grad_mem:.4f}MB")
            elif param.requires_grad:
                print(f"    对应梯度: 将分配{param_mem:.4f}MB")
            else:
                print(f"    对应梯度: 0MB (冻结)")
        
        print(f"\n内存总结:")
        print(f"  参数内存: {param_memory:.4f}MB")
        print(f"  梯度内存: {grad_memory:.4f}MB")
        print(f"  总计: {param_memory + grad_memory:.4f}MB")
        
        return param_memory, grad_memory
    
    # ========== 开始演示 ==========
    
    # 1. 正常状态（未冻结）
    print("\n" + "="*50)
    print("第一阶段：正常状态 (所有参数可训练)")
    print("="*50)
    
    total_params, trainable_params = analyze_model_state("正常状态", model)
    loss1, output1, activations1 = analyze_forward_pass("正常状态", model, input_data, target_data)
    analyze_computation_graph("正常状态", loss1)
    grad_memory1 = analyze_backward_pass("正常状态", loss1, model)
    param_memory1, actual_grad_memory1 = analyze_memory_usage("正常状态", model)
    
    # 2. 冻结layer2
    print("\n" + "="*50)
    print("第二阶段：冻结Layer2参数")
    print("="*50)
    
    # 重新创建模型以清除之前的梯度
    model2 = TwoLayerMLP(input_dim=512, hidden_dim=1024, output_dim=256)
    
    # 冻结layer2
    print("冻结layer2参数...")
    for param in model2.layer2.parameters():
        param.requires_grad = False
    
    frozen_total_params, frozen_trainable_params = analyze_model_state("冻结Layer2", model2)
    loss2, output2, activations2 = analyze_forward_pass("冻结Layer2", model2, input_data, target_data)
    analyze_computation_graph("冻结Layer2", loss2)
    grad_memory2 = analyze_backward_pass("冻结Layer2", loss2, model2)
    param_memory2, actual_grad_memory2 = analyze_memory_usage("冻结Layer2", model2)
    
    # 3. 对比分析
    print("\n" + "="*50)
    print("第三阶段：对比分析")
    print("="*50)
    
    print("参数对比:")
    print(f"  正常状态可训练参数: {trainable_params:,}")
    print(f"  冻结后可训练参数: {frozen_trainable_params:,}")
    print(f"  减少的参数数量: {trainable_params - frozen_trainable_params:,}")
    print(f"  参数减少比例: {(trainable_params - frozen_trainable_params)/trainable_params*100:.1f}%")
    
    print(f"\n内存对比:")
    print(f"  正常状态梯度内存: {actual_grad_memory1:.4f}MB")
    print(f"  冻结后梯度内存: {actual_grad_memory2:.4f}MB")
    print(f"  节省的梯度内存: {actual_grad_memory1 - actual_grad_memory2:.4f}MB")
    print(f"  内存节省比例: {(actual_grad_memory1 - actual_grad_memory2)/actual_grad_memory1*100:.1f}%")
    
    print(f"\n计算图对比:")
    print(f"  正常状态：梯度会传播到所有参数")
    print(f"  冻结后：梯度不会累积到layer2参数，但仍会传播到layer1")
    print(f"  关键观察：计算图结构基本相同，只是部分AccumulateGrad节点被跳过")
    
    # 4. 验证梯度传播路径
    print("\n" + "="*50)
    print("第四阶段：验证梯度传播路径")
    print("="*50)
    
    # 创建简单的测试用例
    test_model = TwoLayerMLP(input_dim=3, hidden_dim=5, output_dim=2)
    test_input = torch.randn(2, 3)
    test_target = torch.randn(2, 2)
    
    print("测试用例 - 正常状态:")
    test_output = test_model(test_input)
    test_loss = criterion(test_output, test_target)
    test_loss.backward()
    
    for name, param in test_model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: 梯度范数={param.grad.norm().item():.6f}")
    
    # 冻结layer2并重新测试
    test_model2 = TwoLayerMLP(input_dim=3, hidden_dim=5, output_dim=2)
    test_model2.layer2.requires_grad_(False)
    
    print("\n测试用例 - 冻结layer2:")
    test_output2 = test_model2(test_input)
    test_loss2 = criterion(test_output2, test_target)
    test_loss2.backward()
    
    for name, param in test_model2.named_parameters():
        if param.grad is not None:
            print(f"  {name}: 梯度范数={param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: 梯度为None (冻结)")
    
    print("\n关键结论:")
    print("1. 冻结layer2后，layer1仍然能接收到梯度")
    print("2. 计算图结构基本保持不变，只是跳过了被冻结参数的梯度累积")
    print("3. 显存占用减少了冻结参数对应的梯度存储空间")
    print("4. 反向传播路径：loss → layer2输出 → layer1输出 → layer1参数")
    print("5. layer2参数不参与梯度计算，但layer2的激活值仍参与梯度传播")

# 示例使用
if __name__ == "__main__":
    print("=== 计算图和显存分析 ===\n")
    
    # 1. 计算图可视化
    loss, variables = visualize_computation_graph()
    
    # 2. 反向传播详细分析
    analyze_backward_process(loss, variables)
    
    # 3. 显存详细分析
    calculate_memory_breakdown()
    
    # 4. 显存优化技术
    memory_optimization_demo()
    
    # 5. 专门演示layer2冻结
    demonstrate_layer2_freezing()
    
    # 6. 实际内存监控
    print("\n=== 实际内存监控 ===")
    print("初始状态:")
    get_memory_info()
    
    # 创建模型和数据
    model = TwoLayerMLP(input_dim=1024, hidden_dim=2048, output_dim=512)
    input_data = torch.randn(32, 1024)
    target_data = torch.randn(32, 512)
    
    print("\n模型创建后:")
    get_memory_info()
    
    # 详细的反向传播分析
    loss = detailed_backward_analysis(model, input_data, target_data)
    
    print("\n反向传播后:")
    get_memory_info()
    
    # 分析各张量内存
    print("\n=== 张量内存分析 ===")
    total_memory = 0
    for name, param in model.named_parameters():
        memory = analyze_tensor_memory(param, f"参数-{name}")
        total_memory += memory
        if param.grad is not None:
            grad_memory = analyze_tensor_memory(param.grad, f"梯度-{name}")
            total_memory += grad_memory
    
    print(f"\n模型总内存: {total_memory:.2f}MB")
    
    # 额外的grad_fn分析
    print("\n=== 额外的grad_fn链分析 ===")
    print("损失函数的完整grad_fn链:")
    current = loss.grad_fn
    step = 0
    while current is not None and step < 20:
        print(f"步骤{step}: {type(current).__name__}")
        if hasattr(current, 'next_functions') and current.next_functions:
            current = current.next_functions[0][0]
        else:
            break
        step += 1
