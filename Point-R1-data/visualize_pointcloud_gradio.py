"""
点云和 3D 模型可视化 Gradio 应用

功能：
1. Objaverse 点云可视化（Plot 点云可视化）
2. GLB 文件可视化（Model3D 3D 模型可视化）
3. ModelNet40 数据集可视化（Plot 点云可视化）

使用方法：
    python visualize_pointcloud_gradio.py --port 7860
"""

import argparse
import os
import sys
import shutil
import tempfile
import numpy as np
import gradio as gr
import plotly.graph_objects as go
try:
    from dataloader import load_objaverse_point_cloud, ModelNet
    from dataloader.utils import pc_norm
except ImportError as e:
    print(f"警告: 无法导入 PointLLM 模块: {e}")
    print("请确保 PointLLM 已正确安装或路径正确")
    load_objaverse_point_cloud = None
    ModelNet = None

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None
    TRIMESH_AVAILABLE = False
    print("警告: trimesh 未安装，GLB 可视化功能将受限")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False
    print("警告: open3d 未安装，某些高级可视化功能将不可用")


def points_to_plotly(points, colors=None, title="点云可视化", point_size=1.5):
    """
    将点云转换为 plotly figure
    
    参数:
        points: numpy array, shape (N, 3) 或 (N, 6)
        colors: numpy array, shape (N, 3) 或 None
        title: 图表标题
        point_size: 点的大小（半径）
    返回:
        plotly figure
    """
    # 提取坐标和颜色
    if points.shape[1] >= 6:
        xyz = points[:, :3]
        if colors is None:
            colors = points[:, 3:6]
            # 归一化颜色到 [0, 1]
            if colors.max() > 1.0:
                colors = colors / 255.0
    else:
        xyz = points[:, :3]
        if colors is None:
            colors = np.ones((xyz.shape[0], 3)) * 0.5  # 默认灰色
    
    # 确保颜色在 [0, 1] 范围内
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # 转换为 RGB 字符串格式
    color_data = (colors * 255).astype(int)
    color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in color_data]
    
    # 创建 plotly figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_strings,
                    opacity=0.8,
                )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True, title='X'),
                yaxis=dict(visible=True, title='Y'),
                zaxis=dict(visible=True, title='Z'),
                aspectmode='data',
            ),
            title=title,
            paper_bgcolor='rgb(255,255,255)',
            height=600,
        ),
    )
    
    return fig


def load_objaverse_pc(data_path, object_id, pointnum=8192, use_color=True, point_size=1.5):
    """
    加载 Objaverse 点云并返回 plotly figure
    
    参数:
        data_path: 数据路径
        object_id: 对象 ID
        pointnum: 点数
        use_color: 是否使用颜色
        point_size: 点的大小（半径）
    返回:
        plotly figure 或 None（如果出错）
    """
    if load_objaverse_point_cloud is None:
        error_msg = "错误: 无法导入 load_objaverse_point_cloud，请确保 PointLLM 已正确安装"
        print(error_msg)
        return None, error_msg
    
    try:
        point_cloud = load_objaverse_point_cloud(data_path, object_id, pointnum=pointnum, use_color=use_color)
        print(f"成功加载点云: {object_id}, 点数: {point_cloud.shape[0]}")

        # ===== 坐标重定向 & 自身归一化（与 extract_dino_features.py 保持一致）=====
        # 原始点云坐标系说明（来自 extract_dino_features.py）：
        #   - 原始点云的 Y 轴正方向 = mesh 采样点的 Z 轴负方向
        #   - 原始点云的 Z 轴正方向 = mesh 的 Y 轴正方向
        #
        # 设原始点 (x_p, y_p, z_p)，对应到 mesh 坐标 (x_m, y_m, z_m) 为：
        #   x_m = x_p
        #   y_m = z_p
        #   z_m = -y_p
        # 即：
        #   [x_m, y_m, z_m] = [x_p, z_p, -y_p]
        #
        # 同时使用点云自身做归一化，使其与 GLB 归一化后坐标范围一致
        if point_cloud.shape[1] >= 3:
            points = point_cloud[:, :3]

            # 坐标轴对齐
            points_aligned = np.empty_like(points)
            points_aligned[:, 0] = points[:, 0]          # X 保持不变
            points_aligned[:, 1] = points[:, 2]          # Y <- 原始 Z
            points_aligned[:, 2] = -points[:, 1]         # Z <- - 原始 Y

            # 自身归一化（与 extract_dino_features.py 中一致）
            min_bound = points_aligned.min(axis=0)
            max_bound = points_aligned.max(axis=0)
            center = (min_bound + max_bound) / 2.0
            extent = max_bound - min_bound
            max_extent = np.max(extent)

            if max_extent < 1e-6:
                print("警告：Objaverse 点云范围过小，跳过归一化，仅做坐标轴对齐")
                points_norm = points_aligned.astype(np.float32)
            else:
                scale = 1.0 / max_extent
                points_norm = (points_aligned - center) * scale
                points_norm = points_norm.astype(np.float32)
                print("已对 Objaverse 点云进行了坐标轴对齐并归一化（与 extract_dino_features.py 一致）")

            # 将归一化后的坐标写回 point_cloud，保留颜色等其它信息
            point_cloud = point_cloud.copy()
            point_cloud[:, :3] = points_norm
        
        fig = points_to_plotly(point_cloud, title=f"Objaverse 点云: {object_id}", point_size=point_size)
        return fig, f"成功加载点云: {object_id}, 点数: {point_cloud.shape[0]}"
    except Exception as e:
        error_msg = f"错误: 无法加载点云数据: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_glb_model(glb_path):
    """
    加载 GLB 文件并返回文件路径用于 Model3D 可视化
    将文件复制到临时目录以符合 Gradio 的安全要求
    
    参数:
        glb_path: GLB 文件路径
    返回:
        临时目录中的 GLB 文件路径或 None（如果出错）
    """
    if not TRIMESH_AVAILABLE:
        error_msg = "错误: trimesh 未安装，无法加载 GLB 文件"
        print(error_msg)
        return None, error_msg
    
    try:
        # 验证 GLB 文件是否可以加载
        scene = trimesh.load(glb_path)
        
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
        elif isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        else:
            error_msg = "错误: 无法识别的 GLB 文件格式"
            print(error_msg)
            return None, error_msg
        
        print(f"成功加载 GLB 文件: {glb_path}")
        print(f"包含 {len(meshes)} 个网格")
        
        # 将文件复制到临时目录（Gradio 允许的路径）
        temp_dir = '/mnt/extra/tmp'
        # 确保临时目录存在
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = os.path.basename(glb_path)
        temp_file_path = os.path.join(temp_dir, filename)
        
        # 如果临时文件已存在，先删除
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # 复制文件到临时目录
        shutil.copy2(glb_path, temp_file_path)
        print(f"文件已复制到临时目录: {temp_file_path}")
        
        info_msg = f"成功加载 GLB 文件，包含 {len(meshes)} 个网格"
        return temp_file_path, info_msg
        
    except Exception as e:
        error_msg = f"错误: 无法加载 GLB 文件: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_modelnet_pc(index=0, split='test', use_color=False, config_path=None, point_size=1.5):
    """
    加载 ModelNet40 点云并返回 plotly figure
    
    参数:
        index: 数据索引
        split: 'train' 或 'test'
        use_color: 是否使用颜色
        config_path: 配置文件路径
        point_size: 点的大小（半径）
    返回:
        plotly figure 或 None（如果出错）
    """
    if ModelNet is None:
        error_msg = "错误: 无法导入 ModelNet，请确保 PointLLM 已正确安装"
        print(error_msg)
        return None, error_msg
    
    try:
        # 使用默认配置文件路径
        if config_path is None or config_path == "":
            config_path = 'dataloader/modelnet_config/ModelNet40.yaml'
        if not os.path.exists(config_path):
            error_msg = f"警告: 配置文件不存在: {config_path}\n请手动指定 config_path 或确保 PointLLM 路径正确"
            print(error_msg)
            return None, error_msg
        
        dataset = ModelNet(config_path=config_path, split=split, subset_nums=-1, use_color=use_color)
        
        index = int(index)  # 确保索引是整数
        if index < 0 or index >= len(dataset):
            error_msg = f"错误: 索引 {index} 超出范围，数据集大小为 {len(dataset)}"
            print(error_msg)
            return None, error_msg
        
        data_dict = dataset[index]
        point_cloud = data_dict['point_clouds'].numpy()
        label = data_dict['labels']
        label_name = data_dict['label_names']
        
        print(f"成功加载 ModelNet40 数据:")
        print(f"  索引: {index}")
        print(f"  类别: {label_name} (ID: {label})")
        print(f"  点数: {point_cloud.shape[0]}")
        
        title = f"ModelNet40 - {label_name} (索引: {index})"
        fig = points_to_plotly(point_cloud, title=title, point_size=point_size)
        info_msg = f"成功加载 ModelNet40 数据:\n索引: {index}\n类别: {label_name} (ID: {label})\n点数: {point_cloud.shape[0]}"
        return fig, info_msg
        
    except Exception as e:
        error_msg = f"错误: 无法加载 ModelNet40 数据: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_pca_pointcloud(ply_path, point_size=1.5):
    """
    加载PCA点云文件（.ply格式）并返回 plotly figure
    
    参数:
        ply_path: .ply文件路径
        point_size: 点的大小（半径）
    返回:
        plotly figure 或 None（如果出错）
    """
    if not OPEN3D_AVAILABLE:
        error_msg = "错误: open3d 未安装，无法加载 .ply 文件"
        print(error_msg)
        return None, error_msg
    
    try:
        if not os.path.exists(ply_path):
            error_msg = f"错误: 文件不存在: {ply_path}"
            print(error_msg)
            return None, error_msg
        
        if not ply_path.lower().endswith('.ply'):
            error_msg = "错误: 文件必须是 .ply 格式"
            print(error_msg)
            return None, error_msg
        
        # 使用open3d加载点云
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if len(pcd.points) == 0:
            error_msg = "错误: 点云文件为空"
            print(error_msg)
            return None, error_msg
        
        # 提取点云坐标
        points = np.asarray(pcd.points)
        
        # 提取颜色（如果有）
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # open3d的颜色范围是[0, 1]，plotly也需要[0, 1]
        else:
            # 如果没有颜色，使用默认灰色
            colors = np.ones((points.shape[0], 3)) * 0.5
        
        print(f"成功加载PCA点云:")
        print(f"  文件: {ply_path}")
        print(f"  点数: {points.shape[0]}")
        print(f"  有颜色: {pcd.has_colors()}")
        
        filename = os.path.basename(ply_path)
        title = f"PCA点云可视化: {filename}"
        fig = points_to_plotly(points, colors=colors, title=title, point_size=point_size)
        info_msg = f"成功加载PCA点云:\n文件: {filename}\n点数: {points.shape[0]}\n有颜色: {pcd.has_colors()}"
        return fig, info_msg
        
    except Exception as e:
        error_msg = f"错误: 无法加载PCA点云: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def change_mode(mode):
    """根据模式切换显示不同的输入组件和可视化组件"""
    if mode == 'Objaverse':
        return [
            gr.update(visible=True),   # objaverse inputs
            gr.update(visible=False),  # glb inputs
            gr.update(visible=False),  # modelnet inputs
            gr.update(visible=False),  # pca inputs
            gr.update(visible=True),   # plot_output (点云可视化)
            gr.update(visible=False),  # model3d_output (3D模型可视化)
        ]
    elif mode == 'GLB':
        return [
            gr.update(visible=False),  # objaverse inputs
            gr.update(visible=True),   # glb inputs
            gr.update(visible=False),  # modelnet inputs
            gr.update(visible=False),  # pca inputs
            gr.update(visible=False),  # plot_output (点云可视化)
            gr.update(visible=True),   # model3d_output (3D模型可视化)
        ]
    elif mode == 'ModelNet40':
        return [
            gr.update(visible=False),  # objaverse inputs
            gr.update(visible=False),  # glb inputs
            gr.update(visible=True),   # modelnet inputs
            gr.update(visible=False),  # pca inputs
            gr.update(visible=True),   # plot_output (点云可视化)
            gr.update(visible=False),  # model3d_output (3D模型可视化)
        ]
    elif mode == 'PCA点云':
        return [
            gr.update(visible=False),  # objaverse inputs
            gr.update(visible=False),  # glb inputs
            gr.update(visible=False),  # modelnet inputs
            gr.update(visible=True),   # pca inputs
            gr.update(visible=True),   # plot_output (点云可视化)
            gr.update(visible=False),  # model3d_output (3D模型可视化)
        ]


def main():
    parser = argparse.ArgumentParser(description='点云和 3D 模型可视化 Gradio 应用')
    
    parser.add_argument('--port', type=int, default=7860,
                       help='Gradio 服务器端口（默认: 7860）')
    parser.add_argument('--server_name', type=str, default='0.0.0.0',
                       help='服务器地址（默认: 0.0.0.0）')
    parser.add_argument('--share', action='store_true',
                       help='是否创建公共链接')
    
    args = parser.parse_args()
    
    with gr.Blocks(title="点云可视化工具") as demo:
        gr.Markdown(
            """
            # 点云和 3D 模型可视化工具 🎨
            
            支持四种模式：
            1. **Objaverse 点云** - 通过对象 ID 加载点云（点云可视化）
            2. **GLB 文件** - 输入服务器上的 GLB 文件路径（3D 模型可视化）
            3. **ModelNet40** - 浏览 ModelNet40 数据集（点云可视化）
            4. **PCA点云** - 加载DINO特征PCA可视化点云（.ply格式）
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ['Objaverse', 'GLB', 'ModelNet40', 'PCA点云'],
                    value='Objaverse',
                    label='选择模式',
                    info='选择要可视化的点云类型'
                )
                
                # Objaverse 输入
                with gr.Group(visible=True) as objaverse_inputs:
                    objaverse_data_path = gr.Textbox(
                        label='数据路径',
                        placeholder='输入 Objaverse 数据路径',
                        value='data/objaverse_data'
                    )
                    objaverse_object_id = gr.Textbox(
                        label='对象 ID',
                        placeholder='输入对象 ID',
                        value=''
                    )
                    objaverse_pointnum = gr.Slider(
                        minimum=1024,
                        maximum=16384,
                        value=8192,
                        step=1024,
                        label='点数'
                    )
                    objaverse_use_color = gr.Checkbox(
                        label='使用颜色',
                        value=True
                    )
                    objaverse_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小（半径）'
                    )
                    objaverse_btn = gr.Button('加载 Objaverse 点云', variant='primary')
                
                # GLB 输入
                with gr.Group(visible=False) as glb_inputs:
                    glb_file_path = gr.Textbox(
                        label='GLB 文件路径',
                        placeholder='输入服务器上的 GLB 文件路径',
                        value=''
                    )
                    glb_btn = gr.Button('加载 GLB 文件', variant='primary')
                
                # ModelNet40 输入
                with gr.Group(visible=False) as modelnet_inputs:
                    modelnet_index = gr.Number(
                        label='数据索引',
                        value=0,
                        precision=0
                    )
                    modelnet_split = gr.Radio(
                        ['train', 'test'],
                        value='test',
                        label='数据集分割'
                    )
                    modelnet_use_color = gr.Checkbox(
                        label='使用颜色',
                        value=False
                    )
                    modelnet_config_path = gr.Textbox(
                        label='配置文件路径（可选）',
                        placeholder='留空使用默认路径',
                        value=''
                    )
                    modelnet_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小（半径）'
                    )
                    modelnet_btn = gr.Button('加载 ModelNet40 点云', variant='primary')
                
                # PCA点云输入
                with gr.Group(visible=False) as pca_inputs:
                    pca_file_path = gr.Textbox(
                        label='PCA点云文件路径 (.ply)',
                        placeholder='输入服务器上的 .ply 文件路径',
                        value=''
                    )
                    pca_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小（半径）'
                    )
                    pca_btn = gr.Button('加载 PCA 点云', variant='primary')
                
                info_output = gr.Textbox(
                    label='信息',
                    interactive=False,
                    lines=3
                )
            
            with gr.Column(scale=2):
                plot_output = gr.Plot(label='点云可视化', visible=True)
                model3d_output = gr.Model3D(label='3D 模型可视化', visible=False)
        
        # 模式切换
        mode.change(
            change_mode,
            inputs=[mode],
            outputs=[objaverse_inputs, glb_inputs, modelnet_inputs, pca_inputs, plot_output, model3d_output]
        )
        
        # Objaverse 按钮事件
        objaverse_btn.click(
            load_objaverse_pc,
            inputs=[objaverse_data_path, objaverse_object_id, objaverse_pointnum, objaverse_use_color, objaverse_point_size],
            outputs=[plot_output, info_output]
        )
        
        # GLB 按钮事件
        def load_glb_wrapper(file_path):
            if file_path is None or file_path.strip() == "":
                return None, "错误: 请输入 GLB 文件路径"
            file_path = file_path.strip()
            if not os.path.exists(file_path):
                return None, f"错误: 文件不存在: {file_path}"
            if not file_path.lower().endswith('.glb'):
                return None, "错误: 文件必须是 .glb 格式"
            return load_glb_model(file_path)
        
        glb_btn.click(
            load_glb_wrapper,
            inputs=[glb_file_path],
            outputs=[model3d_output, info_output]
        )
        
        # ModelNet40 按钮事件
        modelnet_btn.click(
            load_modelnet_pc,
            inputs=[modelnet_index, modelnet_split, modelnet_use_color, modelnet_config_path, modelnet_point_size],
            outputs=[plot_output, info_output]
        )
        
        # PCA点云按钮事件
        def load_pca_wrapper(file_path, point_size):
            if file_path is None or file_path.strip() == "":
                return None, "错误: 请输入 PCA 点云文件路径"
            file_path = file_path.strip()
            return load_pca_pointcloud(file_path, point_size)
        
        pca_btn.click(
            load_pca_wrapper,
            inputs=[pca_file_path, pca_point_size],
            outputs=[plot_output, info_output]
        )
        
        gr.Markdown(
            """
            ### 使用说明：
            - **Objaverse 模式**: 输入数据路径和对象 ID，点击加载按钮（点云可视化）
            - **GLB 模式**: 输入服务器上的 GLB 文件路径，点击加载按钮（3D 模型可视化）
            - **ModelNet40 模式**: 输入数据索引和数据集分割，点击加载按钮（点云可视化）
            - **PCA点云模式**: 输入 .ply 文件路径（例如：`example_material/dino_features/xxx_pca.ply`），点击加载按钮（点云可视化）
            
            ### 提示：
            - 点云可视化支持鼠标交互（旋转、缩放、平移）
            - 3D 模型可视化支持完整的 3D 交互（旋转、缩放、平移、材质查看）
            - 如果点云没有颜色信息，将显示为灰色
            - PCA点云的颜色表示特征的PCA降维结果（RGB对应前3个主成分）
            """
        )
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        allowed_paths=['/mnt/extra/tmp']
    )


if __name__ == '__main__':
    main()

