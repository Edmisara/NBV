from pathlib import Path
import json
import numpy as np
import open3d as o3d
from collections import Counter

# ==== 你的可见性计算函数（完全不动） ====
##from capture_depth_calculate_camera import compute_visible_points, blender_camera_extrinsic, visualize_visibility_sequence

def visualize_visibility_sequence(model_pcd, env_pcd, visible_ids_sequence):
    """
    可视化多轮新增可见点：
    - 每轮颜色不同（红、橙、黄、绿、蓝、紫）
    - 环境点为灰色
    - 未被观测的点不显示

    参数:
        model_pcd: 原始物体点云 (o3d.geometry.PointCloud)
        env_pcd: 环境点云 (o3d.geometry.PointCloud)
        visible_ids_sequence: List[np.ndarray[int]]
            - 每轮新增的可见点索引（非累计）
    """
    points_world_orig = np.asarray(model_pcd.points)
    color_list = [
        [1.0, 0.0, 0.0],  # 红
        [1.0, 0.5, 0.0],  # 橙
        [1.0, 1.0, 0.0],  # 黄
        [0.0, 1.0, 0.0],  # 绿
        [0.0, 0.5, 1.0],  # 蓝
        [0.5, 0.0, 1.0],  # 紫
    ]

    pcds = []

    # 环境点云：灰色
    env_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcds.append(env_pcd)

    for i, visible_ids in enumerate(visible_ids_sequence):
        color = color_list[i % len(color_list)]
        pcd_visible = o3d.geometry.PointCloud()
        pcd_visible.points = o3d.utility.Vector3dVector(points_world_orig[visible_ids])
        pcd_visible.paint_uniform_color(color)
        pcds.append(pcd_visible)

    o3d.visualization.draw_geometries(
        pcds,
        window_name="NBV Visibility Sequence (Red→Purple = Earlier→Later)",
        width=1280, height=720,
    )


def blender_camera_extrinsic(cam_pos, lookat, up_vector=np.array([0, 0, 1])):
    """
    生成 Blender 相机外参矩阵（T_world2cam），用于可见性计算等。

    参数:
        cam_pos: 相机位置，np.array([x, y, z])
        lookat: 相机观察点，np.array([x, y, z])
        up_vector: 相机的世界上方向（默认 Z 轴朝上）

    返回:
        4x4 外参矩阵 (np.ndarray)
    """
    # Z轴: 从相机指向目标
    forward = cam_pos - lookat
    forward = forward / np.linalg.norm(forward)

    # X轴: 右方向
    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)

    # Y轴: 上方向（重新计算）
    up = np.cross(forward, right)

    # 相机旋转矩阵
    R = np.stack([right, up, forward], axis=1)  # 列向量是右、上、前

    # 转为相机坐标系 → 相当于逆变换（转置 + 平移）
    R_inv = R.T
    t_inv = -R_inv @ cam_pos

    # 构造外参矩阵
    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv
    return T


def compute_visible_points(model_pcd, env_pcd, cam):
    points_world_orig = np.asarray(model_pcd.points)
    points_env = np.asarray(env_pcd.points)

    fx, fy = cam["intrinsic"]["fx"], cam["intrinsic"]["fy"]
    cx, cy = cam["intrinsic"]["cx"], cam["intrinsic"]["cy"]
    width, height = cam["intrinsic"]["width"], cam["intrinsic"]["height"]
    T_world2cam = np.array(cam["extrinsic"])

    # ======= 自动加密 =======
    pcd_tree = o3d.geometry.KDTreeFlann(model_pcd)
    distances = []
    for i in range(len(points_world_orig)):
        [_, idx, dists] = pcd_tree.search_knn_vector_3d(model_pcd.points[i], 6)
        distances.extend(np.sqrt(dists[1:]))
    mean_dist = np.mean(distances)
    target_spacing = mean_dist / 2.0

    density = 4
    r = np.arange(-density // 2 + 1, density // 2 + 1)
    offsets = np.array([[dx, dy, dz] for dx in r for dy in r for dz in r]) * target_spacing / (density // 2)

    expanded_model = (points_world_orig[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    orig_indices = np.repeat(np.arange(len(points_world_orig)), len(offsets))

    expanded_env = (points_env[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

    # 合并膨胀后的模型和环境点
    points_all = np.vstack([expanded_env, expanded_model])

    # ======= 投影与遮挡剔除 =======
    points_homo = np.concatenate([points_all, np.ones((len(points_all), 1))], axis=1)
    points_cam = (T_world2cam @ points_homo.T).T[:, :3]
    z_cam = points_cam[:, 2]
    valid = z_cam < 0
    points_cam = points_cam[valid]
    z_cam = -z_cam[valid]
    u = (fx * points_cam[:, 0] / -points_cam[:, 2] + cx).astype(np.int32)
    v = (fy * points_cam[:, 1] / -points_cam[:, 2] + cy).astype(np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    index_map = np.full((height, width), -1, dtype=np.int32)
    valid_indices = np.nonzero(valid)[0]

    for i in range(len(z_cam)):
        if 0 <= u[i] < width and 0 <= v[i] < height:
            if z_cam[i] < depth[v[i], u[i]]:
                depth[v[i], u[i]] = z_cam[i]
                index_map[v[i], u[i]] = valid_indices[i]

    # ======= 投票找原始点可见性 =======
    visible_expanded_indices = np.unique(index_map[index_map >= 0])

    model_offset = len(expanded_env)
    visible_model_indices = visible_expanded_indices[visible_expanded_indices >= model_offset]
    visible_model_indices = visible_model_indices.astype(np.int32)

    # 映射：visible_model_indices 中是哪些膨胀模型点被看到（相对于 expanded_model 起点）

    # 找出哪些膨胀模型点是有效的（相机前方）
    valid_model_mask = valid[model_offset:]
    assert valid_model_mask.shape[0] == orig_indices.shape[0], "❌ 长度不匹配，模型膨胀点数异常"

    # 映射膨胀索引 → valid 中压缩后的索引（只对 valid 为 True 的点保留位置）
    valid_index_map = -np.ones(valid_model_mask.shape[0], dtype=np.int32)
    valid_index_map[valid_model_mask] = np.arange(np.sum(valid_model_mask))

    # 可见膨胀点的相对索引（相对于 expanded_model）
    relative_indices = visible_model_indices - model_offset

    # 在压缩后的 valid 索引映射中查找这些可见点的位置
    compressed_indices = valid_index_map[relative_indices]

    # 过滤掉 -1（即 invalid）
    valid_mask = compressed_indices >= 0
    visible_orig_ids_all = orig_indices[compressed_indices[valid_mask]]


    vote_count = Counter(visible_orig_ids_all)

    min_visible_votes = len(offsets) * 0.12  # 阈值，支持参数化
    visible_mask = np.array([vote_count[i] >= min_visible_votes for i in range(len(points_world_orig))])

    # ===== 👇 插入法线方向过滤逻辑 👇 =====
    if model_pcd.has_normals():
        normals = np.asarray(model_pcd.normals)
        cam_pos_world = np.linalg.inv(T_world2cam)[:3, 3]  # 相机位置 in world

        vectors_to_cam = cam_pos_world - points_world_orig  # [N, 3]
        vectors_to_cam /= np.linalg.norm(vectors_to_cam, axis=1, keepdims=True)

        normals_unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        cos_angles = np.einsum("ij,ij->i", vectors_to_cam, normals_unit)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))  # 弧度
        angle_threshold = np.deg2rad(100)  # 可调：设为 120 或 135 均可
        direction_mask = angles < angle_threshold

        visible_mask = visible_mask & direction_mask
    # ===== 👆 插入法线方向过滤逻辑 👆 =====

    visible_orig_ids = np.where(visible_mask)[0]

    return visible_orig_ids

# === 路径配置 ===
BASE = Path("D:/NBV/nbv_simulation/results")
CLOUD_DIR = BASE / "ReferenceCloud"
MODEL_PATH = CLOUD_DIR / "DiningTable6Seat_038_reference_points.ply"
ENV_PATH = CLOUD_DIR / "VicSideChair_038_reference_points.ply"
INIT_CAM_PATH = BASE / "camera.json"
NBV_CAM_DIR = BASE / "VisualCameras"

# === 读取点云 ===
model_pcd = o3d.io.read_point_cloud(str(MODEL_PATH))
env_pcd = o3d.io.read_point_cloud(str(ENV_PATH))


# === 可选：体素下采样（调试阶段加速）
voxel_size = 0.3  
model_pcd = model_pcd.voxel_down_sample(voxel_size=voxel_size)
env_pcd = env_pcd.voxel_down_sample(voxel_size=voxel_size)




# === 初始化点索引 ===
points = np.asarray(model_pcd.points)
total_ids = set(range(len(points)))
lookat = np.mean(points, axis=0)

# === 第一个相机（初始）
with open(INIT_CAM_PATH, 'r') as f:
    cam = json.load(f)
visible_ids =  compute_visible_points(model_pcd, env_pcd, cam)

T_world2cam = np.array(cam["extrinsic"])
R = T_world2cam[:3, :3]
t = T_world2cam[:3, 3]
cam_pos = -R.T @ t  # 由外参矩阵推得相机位置



all_visible_ids = set(visible_ids)
visible_ids_sequence = [np.array(visible_ids)]
visualize_visibility_sequence(model_pcd, env_pcd, visible_ids_sequence)
# === 后续4个 NBV 相机
nbv_cameras = [
    NBV_CAM_DIR / "camera_00.json",
    NBV_CAM_DIR / "camera_01.json",
    NBV_CAM_DIR / "camera_02.json",
    NBV_CAM_DIR / "camera_03.json"
]

for idx, cam_path in enumerate(nbv_cameras, start=1):
    if not cam_path.exists():
        print(f"❌ 缺失文件: {cam_path}")
        continue

    with open(cam_path, 'r') as f:
        cam_pose = json.load(f)

    cam_pos = np.array(cam_pose["position"])
    extrinsic = blender_camera_extrinsic(cam_pos, lookat)
    cam_struct = {
        "intrinsic": cam["intrinsic"],
        "extrinsic": extrinsic.tolist()
    }

    visible_ids =  compute_visible_points(model_pcd, env_pcd, cam_struct)
    new_ids = np.setdiff1d(visible_ids, list(all_visible_ids))
    visible_ids_sequence.append(new_ids)
    all_visible_ids.update(new_ids.tolist())

    print(f"📷 相机 {idx}: 新增 {len(new_ids)} 个点，累计 {len(all_visible_ids)}")
    visualize_visibility_sequence(model_pcd, env_pcd, visible_ids_sequence)

# === 最终可视化和统计 ===
coverage = len(all_visible_ids) / len(points)
print(f"\n✅ 五个相机合并后总覆盖率：{coverage:.2%}（{len(all_visible_ids)} / {len(points)}）")

