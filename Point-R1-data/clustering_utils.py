
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_linkage_matrix(model, n_samples):
    """
    从已拟合的 sklearn AgglomerativeClustering 模型构建 scipy linkage 矩阵
    """
    # 创建 counts 数组 (每个节点包含的样本数)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    # children_ 中的索引：
    # 0..n_samples-1 是原始样本 (叶节点)
    # >= n_samples 是非叶节点
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶节点数量为 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([
        model.children_, 
        model.distances_,
        counts
    ]).astype(float)

    return linkage_matrix

def perform_hierarchical_clustering(points, features, k_neighbors, betas):
    """
    执行基于 KNN 空间约束和特征相似度的层级聚类
    使用相对阈值策略，使 Beta 参数更直观
    
    参数:
        points: (N, 3) 点云坐标
        features: (N, D) 特征向量
        k_neighbors: KNN 的 K 值
        betas: 包含4个特征相似度阈值的列表 (0.0 - 1.0)
               Beta 越大 = 要求越相似 = 阈值越小 = 聚类越细碎
        
    返回:
        clustering_results: 字典，key为层级索引(0-3)，value为对应的聚类标签数组 (N,)
    """
    N = points.shape[0]
    
    # 1. 特征预处理
    # Ward Linkage 最小化方差增量，使用欧氏距离。
    # 对归一化特征使用欧氏距离等价于考虑方向相似性。
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_features = features / norms
    
    # 2. 构建空间约束图
    k = int(k_neighbors)
    # mode='connectivity' 返回 0/1 矩阵
    connectivity = kneighbors_graph(points, n_neighbors=k, include_self=False)
    
    # 3. 训练全树 (Full Tree)
    # distance_threshold=0, n_clusters=None 强制构建完整的树
    # 使用 Ward Linkage 以获得紧凑的聚类，解决"簇内差异大"问题
    print(f"Computing clustering tree (Ward) with K={k} spatial constraints...")
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        metric='euclidean',  # Ward requires euclidean
        connectivity=connectivity,
        linkage='ward',
        compute_full_tree=True  # 确保计算完整的树用于提取 distances_
    )
    
    try:
        model.fit(normalized_features)
    except Exception as e:
        print(f"聚类构建失败: {e}")
        return {i: np.arange(N) for i in range(len(betas))}
        
    # 4. 构建 Linkage Matrix
    # 这样我们可以使用 scipy 的 fcluster 进行快速切割，无需重复 fit
    Z = get_linkage_matrix(model, N)
    
    # 获取树的最大高度 (最大合并距离)
    max_dist = Z[:, 2].max()
    print(f"Tree max distance: {max_dist:.4f}")
    
    clustering_results = {}
    
    # 5. 根据相对阈值切割
    for idx, beta in enumerate(betas):
        # Beta 是相似度 (0~1)，我们需要将其转换为距离阈值
        # Beta=1.0 (最相似) -> Ratio=0.0 -> Threshold=0.0 (每个点一类)
        # Beta=0.0 (最不相似) -> Ratio=1.0 -> Threshold=MaxDist (全合并)
        # 
        # 考虑到用户习惯，通常 Beta=0.9 希望是很细的聚类，Beta=0.3 是很粗的。
        # 为了让控制更线性，我们可以直接用线性映射：
        # Threshold = MaxDist * (1 - Beta)
        # 但是为了防止 Beta=1 时阈值完全为0导致计算错误，加一个小 epsilon
        
        threshold_ratio = 1.0 - beta
        if threshold_ratio < 1e-4: threshold_ratio = 1e-4
        
        threshold = max_dist * threshold_ratio
        
        # 使用 fcluster 提取扁平聚类
        labels = fcluster(Z, t=threshold, criterion='distance')
        
        # fcluster 返回的标签是从 1 开始的，为了统一习惯改为从 0 开始
        labels = labels - 1
        
        n_clusters = len(np.unique(labels))
        print(f"  Level {idx+1}: Beta={beta:.2f} -> Thresh={threshold:.4f} (Ratio={threshold_ratio:.2f}) -> Clusters={n_clusters}")
        
        clustering_results[idx] = labels
        
    return clustering_results

def generate_cluster_colors(labels):
    """
    为聚类标签生成颜色
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 随机颜色生成器
    np.random.seed(42)
    
    # 生成颜色: Hue 随机, Saturation/Value 高
    hues = np.random.rand(n_clusters)
    sats = np.random.uniform(0.6, 0.95, n_clusters)
    vals = np.random.uniform(0.7, 1.0, n_clusters)
    
    colors_lookup = np.zeros((n_clusters, 3))
    for i in range(n_clusters):
        colors_lookup[i] = mcolors.hsv_to_rgb([hues[i], sats[i], vals[i]])
    
    point_colors = np.zeros((len(labels), 3))
    
    # 构建 label -> index 映射
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    for i, lbl in enumerate(labels):
        if lbl in label_to_idx:
            idx = label_to_idx[lbl]
            point_colors[i] = colors_lookup[idx]
        else:
            point_colors[i] = [0.5, 0.5, 0.5] # Fallback grey
        
    return point_colors
