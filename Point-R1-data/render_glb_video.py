"""
GLB 模型环绕视频生成工具
输入 GLB 路径，输出 360 度环绕视频 (GIF 或 MP4)
"""
import argparse
import os
import numpy as np
import imageio
from renderer_o3d import Open3DRenderer

def get_viewpoint_from_angles(azimuth, elevation, radius):
    """根据方位角、仰角和半径计算相机位置"""
    theta = np.radians(90 - elevation)
    phi = np.radians(azimuth)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.sin(phi)
    
    return np.array([x, y, z])

def render_turntable_video(glb_path, output_path, width=512, height=512, frames=60, elevation=20, fps=30):
    """
    生成 GLB 模型的环绕视频
    
    参数:
        glb_path: GLB 文件路径
        output_path: 输出视频路径 (.gif 或 .mp4)
        width, height: 分辨率
        frames: 总帧数
        elevation: 俯仰角度
        fps: 帧率
    """
    if not os.path.exists(glb_path):
        print(f"错误: 文件不存在: {glb_path}")
        return

    print(f"初始化渲染器 ({width}x{height})...")
    renderer = Open3DRenderer(width=width, height=height)
    
    try:
        renderer.setup()
        
        print(f"加载模型: {glb_path}")
        model = renderer.load_model(glb_path)
        model, _ = renderer.normalize_model(model)
        renderer.upload_model(model)
        
        print(f"渲染 {frames} 帧...")
        images = []
        radius = 1.3  # 更近的相机距离
        
        for i in range(frames):
            azimuth = (i / frames) * 360.0
            eye = get_viewpoint_from_angles(azimuth, elevation, radius)
            
            img, _ = renderer.render_view(eye, center=np.array([0, 0, 0]), return_depth=False)
            images.append(img)
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{frames}")
        
        # 保存视频
        print(f"保存视频: {output_path}")
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext == '.gif':
            imageio.mimsave(output_path, images, fps=fps, loop=0)
        else:
            imageio.mimsave(output_path, images, fps=fps)
            
        print("完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        renderer.cleanup()

def main():
    parser = argparse.ArgumentParser(description='GLB 模型环绕视频生成')
    parser.add_argument('glb_path', type=str, help='GLB 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出路径 (.gif 或 .mp4)')
    parser.add_argument('--size', '-s', type=int, default=512, help='分辨率')
    parser.add_argument('--frames', '-f', type=int, default=60, help='帧数')
    parser.add_argument('--elevation', '-e', type=float, default=20, help='俯仰角')
    parser.add_argument('--fps', type=int, default=24, help='帧率')

    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.glb_path))[0]
        args.output = f"{base}_turntable.gif"
        
    render_turntable_video(
        args.glb_path, 
        args.output, 
        width=args.size, 
        height=args.size, 
        frames=args.frames, 
        elevation=args.elevation,
        fps=args.fps
    )

if __name__ == "__main__":
    main()
