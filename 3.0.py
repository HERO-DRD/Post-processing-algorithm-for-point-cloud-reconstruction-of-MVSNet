import open3d as o3d
import numpy as np
import os
import gc
from scipy.spatial import KDTree
from tqdm import tqdm


# ----------------------------
# 预处理模块
# ----------------------------
def radius_outlier_removal(pcd, radius=1.0, min_points=3):
    """优化的半径滤波"""
    print(f"\n▶ 半径滤波 (半径={radius}, 最小邻域={min_points})")
    cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    filtered_pcd = pcd.select_by_index(ind)
    print(f"✔ 完成滤波: 原始 {len(pcd.points):,} → 保留 {len(filtered_pcd.points):,}")
    return filtered_pcd


# ----------------------------
# 边界检测函数（来自 test.py）
# ----------------------------
def point_cloud_boundary(cloud, radius=0.8, k_neighbors=30, angle_threshold=None):
    """
    基于AC方法的点云边界检测

    参数:
        cloud: open3d点云对象
        radius: 法向量估计半径
        k_neighbors: 邻域点数
        angle_threshold: 边界判定角度阈值（弧度），如果为None则自适应

    返回:
        boundary_indices: 边界点索引列表
    """
    # 1. 计算法向量
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    cloud.orient_normals_consistent_tangent_plane(15)  # 统一法向量方向

    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    kdtree = o3d.geometry.KDTreeFlann(cloud)

    boundary_indices = []

    # 2. 遍历每个点
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], k_neighbors + 1)

        if k <= 1:
            continue
        neighbor_points = points[idx[1:]]
        neighbor_normals = normals[idx[1:]]

        # 动态计算角度阈值
        if angle_threshold is None:
            density = k / (np.pi * radius ** 2)
            angle_threshold = 0.05 * np.log(density + 1) + 1.5

        normal = normals[i]
        if abs(normal[2]) > 0.999:
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = np.array([0.0, 0.0, 1.0])

        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)

        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        vectors = neighbor_points - points[i]
        x_coords = np.dot(vectors, u)
        y_coords = np.dot(vectors, v)

        angles = np.arctan2(y_coords, x_coords)
        angles = np.mod(angles, 2 * np.pi)
        angles.sort()

        if len(angles) > 1:
            gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
            max_gap = np.max(gaps)

            if max_gap > angle_threshold:
                boundary_indices.append(i)

    print(f"✔ 检测到边界点数量：{len(boundary_indices)}")
    return boundary_indices


# ----------------------------
# 替换 find_exterior_points 函数
# ----------------------------
def find_exterior_points(pcd, radius=0.8, min_points=5, angle_threshold=2.5):
    """使用AC方法改进的边界点检测"""
    print("\n▶ 使用AC方法进行边界检测")
    return point_cloud_boundary(pcd, radius=radius, k_neighbors=min_points, angle_threshold=angle_threshold)


# ----------------------------
# 核心处理模块
# ----------------------------
def oriented_dilate_points(pcd, exterior_indices, dilation_radius=1.0, n_samples=600, defect_direction=True):
    """支持缺陷方向膨胀的点云扩展"""
    if not exterior_indices:
        print("⚠ 无边界点可膨胀")
        return pcd

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    centers = points[exterior_indices]
    n_centers = len(centers)

    print("\n⏳ 计算法向量...")
    pcd_for_normals = o3d.geometry.PointCloud()
    pcd_for_normals.points = o3d.utility.Vector3dVector(points)
    pcd_for_normals.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=dilation_radius * 2, max_nn=30)
    )
    pcd_for_normals.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asarray(pcd_for_normals.normals, dtype=np.float32)[exterior_indices]

    # 如果是缺陷方向，则反转法向量
    if defect_direction:
        normals = -normals

    # 新增：计算每个中心点的局部密度
    kdtree = KDTree(points)
    _, nn_indices = kdtree.query(centers, k=30)  # K=30 作为示例值
    densities = []
    for idx in range(n_centers):
        distances = np.linalg.norm(centers[idx] - points[nn_indices[idx][1:]], axis=1)
        density = 30 / (29 * np.sum(distances))
        densities.append(density)
    densities = np.array(densities)

    # 新增：根据局部密度计算自适应膨胀半径
    alpha = 0.5  # 示例值
    beta = 0.2   # 示例值
    gamma = 0.5  # 示例值
    adaptive_radii = alpha * (1 / densities) ** gamma + beta

    chunk_size = 5000
    total_chunks = int(np.ceil(n_centers / chunk_size))
    new_pts_list = []
    new_colors_list = []

    print(f"\n▶ 缺陷方向膨胀 (采样/点={n_samples})")
    for chunk_idx in tqdm(range(total_chunks), desc="处理分块"):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, n_centers)
        chunk_centers = centers[start:end]
        chunk_normals = normals[start:end]
        chunk_size_real = end - start

        # 使用自适应膨胀半径
        chunk_adaptive_radii = adaptive_radii[start:end]

        theta = np.random.uniform(0, 2 * np.pi, (chunk_size_real, n_samples))
        phi = np.arccos(np.random.uniform(0.9, 1.0, (chunk_size_real, n_samples)))

        # 根据自适应半径生成 r
        r = np.array([np.sqrt(np.random.uniform(0, rad**2, n_samples)) for rad in chunk_adaptive_radii])

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        local_pts = np.stack([x, y, z], axis=2).reshape(-1, 3)

        rot_matrices = []
        for normal in chunk_normals:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.cross([0, 0, 1], normal) * np.arccos(normal[2])
            )
            rot_matrices.append(R)
        rot_matrices = np.array(rot_matrices)

        rotated_pts = np.einsum('ijk,ik->ij',
                                np.repeat(rot_matrices, n_samples, axis=0),
                                local_pts)

        expanded_centers = np.repeat(chunk_centers, n_samples, axis=0)
        chunk_new_pts = expanded_centers + rotated_pts

        kdtree = KDTree(points)
        _, nn_indices = kdtree.query(chunk_new_pts, k=3)
        chunk_new_colors = np.mean(colors[nn_indices], axis=1)

        new_pts_list.append(chunk_new_pts.astype(np.float32))
        new_colors_list.append(chunk_new_colors.astype(np.float32))

        del rotated_pts, chunk_new_pts, chunk_new_colors
        if chunk_idx % 10 == 0:
            gc.collect()

    print("\n⏳ 合并结果...")
    dilated_pcd = o3d.geometry.PointCloud()
    dilated_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([points] + new_pts_list)
    )
    dilated_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([colors] + new_colors_list)
    )

    return dilated_pcd.voxel_down_sample(dilation_radius / 5)


# ----------------------------
# 文件管理模块
# ----------------------------
def save_processed_cloud(pcd, input_path, output_dir, suffix="_processed"):
    """版本化保存"""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    version = 1
    while True:
        output_path = os.path.join(output_dir, f"{base}{suffix}_v{version}.ply")
        if not os.path.exists(output_path):
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"\n💾 保存成功: {output_path}")
            return output_path
        version += 1


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    # 输入输出配置
    input_dir = r"F:\本科\毕设\cloud\input"  # 输入文件夹路径
    output_dir = r"F:\本科\毕设\cloud\results"  # 输出文件夹路径

    # 参数配置（可根据需要调整）
    params = {
        'filter': {'radius': 1.2, 'min_points': 4},
        'detection': {'radius': 0.02, 'min_points': 30, 'angle_threshold': None},
        'dilation': {'dilation_radius': 0.8, 'n_samples': 500, 'defect_direction': True}
    }

    # 获取所有PLY文件
    ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    print(f"发现 {len(ply_files)} 个点云文件待处理")

    # 批量处理流程
    for idx, ply_file in enumerate(ply_files, 1):
        try:
            print(f"\n🔍 正在处理文件 ({idx}/{len(ply_files)}): {ply_file}")
            input_path = os.path.join(input_dir, ply_file)

            # 1. 加载点云
            pcd = o3d.io.read_point_cloud(input_path)
            print(f"原始点数: {len(pcd.points):,}")

            # 2. 预处理滤波
            filtered_pcd = radius_outlier_removal(pcd, **params['filter'])
            if not filtered_pcd.has_colors():
                filtered_pcd.paint_uniform_color([0.6, 0.6, 0.6])

            # 3. 边界检测（使用AC方法）
            exterior_ids = find_exterior_points(filtered_pcd, **params['detection'])

            # 4. 法向量膨胀（支持缺陷方向）
            dilated_pcd = oriented_dilate_points(
                filtered_pcd,
                exterior_ids,
                dilation_radius=params['dilation']['dilation_radius'],
                n_samples=params['dilation']['n_samples'],
                defect_direction=params['dilation'].get('defect_direction', False)
            )

            # 5. 保存结果
            save_processed_cloud(dilated_pcd, input_path, output_dir)

            # 6. 内存清理
            del pcd, filtered_pcd, dilated_pcd
            gc.collect()

        except Exception as e:
            print(f"\n❌ 处理文件 {ply_file} 时发生错误: {str(e)}")
            continue

    print("\n🎉 批量处理完成！")
