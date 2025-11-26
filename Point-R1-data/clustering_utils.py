
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy.sparse.csgraph import connected_components
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
    print(f"Building KNN graph with K={k}...")
    connectivity = kneighbors_graph(points, n_neighbors=k, include_self=False)
    
    # 2.1 检查连通性
    # 如果图不是连通的，AgglomerativeClustering 会尝试自动补全连接（并发出警告），或者需要分开处理。
    # 用户明确要求不要自动连接，因此我们按连通分量分别聚类。
    n_comps, comp_labels = connected_components(connectivity, directed=False)
    
    sub_results = [] # 存储每个分量的 (Z, mask)
    global_max_dist = 0.0
    
    if n_comps > 1:
        print(f"检测到 {n_comps} 个独立的连通分量。将分别进行聚类，保持分量间独立。")
        
        for i in range(n_comps):
            mask = comp_labels == i
            n_sub = np.sum(mask)
            if n_sub < 2: # 极少情况，孤立点
                continue
                
            # 提取子集数据
            sub_features = normalized_features[mask]
            # 重新构建子集的 connectivity，因为切片稀疏矩阵可能会有问题且索引需要重置
            # 注意：这里使用原始 points 的子集重新计算 KNN，确保图是干净的
            sub_points = points[mask]
            sub_connectivity = kneighbors_graph(sub_points, n_neighbors=min(k, n_sub-1), include_self=False)
            
            # 训练子树
            try:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0,
                    metric='euclidean',
                    connectivity=sub_connectivity,
                    linkage='ward',
                    compute_full_tree=True
                )
                model.fit(sub_features)
                
                Z = get_linkage_matrix(model, n_sub)
                current_max = Z[:, 2].max()
                if current_max > global_max_dist:
                    global_max_dist = current_max
                    
                sub_results.append({'mask': mask, 'Z': Z, 'n_samples': n_sub})
                
            except Exception as e:
                print(f"分量 {i} 聚类失败: {e}")
                # 即使失败，也可以当作所有点各自为一类，或者统一为一类，这里暂时跳过
                
    else:
        # 只有一个分量，走标准流程
        print(f"Computing clustering tree (Ward) with K={k} spatial constraints...")
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0,
                metric='euclidean',
                connectivity=connectivity,
                linkage='ward',
                compute_full_tree=True
            )
            model.fit(normalized_features)
            Z = get_linkage_matrix(model, N)
            global_max_dist = Z[:, 2].max()
            sub_results.append({'mask': np.ones(N, dtype=bool), 'Z': Z, 'n_samples': N})
            
        except Exception as e:
            print(f"聚类构建失败: {e}")
            return {i: np.arange(N) for i in range(len(betas))}

    print(f"Global Tree max distance: {global_max_dist:.4f}")
    
    clustering_results = {}
    
    # Level 0: 整体点云 (强制为一个 Cluster)
    # 即使 KNN 导致物理不连通，逻辑上也视为一个整体
    clustering_results[0] = np.zeros(N, dtype=int)
    
    # 5. 根据相对阈值切割并合并结果
    for idx, beta in enumerate(betas):
        threshold_ratio = 1.0 - beta
        if threshold_ratio < 1e-4: threshold_ratio = 1e-4
        
        threshold = global_max_dist * threshold_ratio
        
        # 最终的标签数组
        final_labels = np.zeros(N, dtype=int)
        label_offset = 0
        
        # 遍历所有子树的结果
        for res in sub_results:
            Z = res['Z']
            mask = res['mask']
            
            # 切割
            sub_labels = fcluster(Z, t=threshold, criterion='distance')
            
            # fcluster 返回 1-based，转为 0-based
            sub_labels = sub_labels - 1
            
            # 加上偏移量，确保不同分量的标签不冲突
            final_labels[mask] = sub_labels + label_offset
            
            # 更新偏移量 (当前分量的最大标签 + 1)
            if len(sub_labels) > 0:
                label_offset += (sub_labels.max() + 1)
        
        n_clusters = len(np.unique(final_labels))
        # Level 1, 2, 3, 4...
        level_idx = idx + 1
        print(f"  Level {level_idx}: Beta={beta:.2f} -> Thresh={threshold:.4f} -> Clusters={n_clusters}")
        
        clustering_results[level_idx] = final_labels
        
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
