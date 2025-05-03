from pathlib import Path
import numpy as np
import json
from collections import Counter
import open3d as o3d

import os
import numpy as np

def look_at_rotation(camera_pos, target_pos, up_vector=np.array([0, 0, 1])):
    """
    生成 Blender 用的欧拉角 (XYZ 顺序)，使相机从 camera_pos 看向 target_pos。
    """
    from scipy.spatial.transform import Rotation as R

    forward = np.array(target_pos) - np.array(camera_pos)
    forward /= np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    # 注意 forward 要取负以符合 Blender 相机 -Z 看向目标
    rot_matrix = np.stack([right, up, -forward], axis=1)

    r = R.from_matrix(rot_matrix)
    return r.as_euler('xyz', degrees=False)




def compute_valid_camera_distance(center, direction, original_camera_pos):
    """
    沿给定方向从 center 出发，计算在不穿地、不超天花、不超距离的约束下的最大合法相机距离 t。
    
    限制：
    - 相机 z 必须在 [0, 1.5 * original_camera_z] 范围内
    - 相机距离不能超过 r0（原始相机与中心点的距离）
    
    返回：
    - 合法距离 t（float）
    """
    cz = center[2]
    dz = direction[2]
    z0 = original_camera_pos[2]
    r0 = np.linalg.norm(center - original_camera_pos)

    t_candidates = []

    # 地板限制
    if dz < 0:
        t_floor = (0 - cz) / dz
        if t_floor > 0:
            t_candidates.append(t_floor)

    # 天花板限制
    if dz > 0:
        t_ceiling = (1.5 * z0 - cz) / dz
        if t_ceiling > 0:
            t_candidates.append(t_ceiling)

    # 距离限制
    t_candidates.append(r0)

    if not t_candidates or min(t_candidates) <= 0:
        return None  # 表示该方向完全非法

    return min(t_candidates)



def generate_layered_directions(base_direction, angle_deg, num_directions):
    """
    在以 base_direction 为中心的球面锥体（夹角 angle_deg）上均匀采样方向向量。
    返回单位向量列表，长度为 num_directions。
    """
    base_direction = base_direction / np.linalg.norm(base_direction)
    directions = []

    # 找一个与 base_direction 不共线的向量
    if abs(base_direction[2]) < 0.99:
        ortho = np.array([0, 0, 1])
    else:
        ortho = np.array([0, 1, 0])

    # 构造正交基底
    u = np.cross(base_direction, ortho)
    u /= np.linalg.norm(u)
    v = np.cross(base_direction, u)

    # 球面锥体采样
    angle_rad = np.radians(angle_deg)
    for i in range(num_directions):
        theta = 2 * np.pi * i / num_directions
        direction = (
            np.cos(angle_rad) * base_direction +
            np.sin(angle_rad) * (np.cos(theta) * u + np.sin(theta) * v)
        )
        direction /= np.linalg.norm(direction)
        directions.append(direction)

    return directions

def generate_fine_directions(found_angle_deg, angle_step_deg, base_direction, num=24):
    """
    根据外层找到合法方向的夹角 found_angle_deg 和步进角 angle_step_deg，
    自动回退一圈角度并在该夹角上以 base_direction 为中心生成 num 个方向。
    
    示例：若在 45° 层找到合法视角，步进为 15°，则回退至 30° 做精细采样。
    """
    refine_angle = found_angle_deg - angle_step_deg
    if refine_angle <= 0:
        # 防御：如果已经是最内圈，不再回退，使用当前角度
        refine_angle = found_angle_deg
    return generate_layered_directions(base_direction, refine_angle, num_directions=num)



def save_blender_camera_position(view_result, save_dir, idx):
    """
    保存 Blender 相机位置和朝向向量（来自视角搜索结果）。

    参数:
        view_result: 包含 'location' 和 'rotation_euler' 的字典
        save_dir: 保存目录
        idx: 当前相机编号
    """
    data = {
        "position": view_result["location"],
        "rotation_euler": view_result["rotation_euler"]
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = Path(save_dir) / f"camera_{idx:02d}.json"
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"📸 保存 Blender 相机位置到: {save_path}")


def estimate_radius_from_pointcloud(pcd, factor=2.0):
    import numpy as np
    import open3d as o3d

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    min_dist = np.inf

    for i in range(len(points)):
        [_, idx, dist_sq] = pcd_tree.search_knn_vector_3d(points[i], 2)
        if len(dist_sq) >= 2:
            dist = np.sqrt(dist_sq[1])
            if dist > 0:
                min_dist = min(min_dist, dist)

    return factor * min_dist if np.isfinite(min_dist) else 0.1  # fallback


def ray_hits_kdtree(ray_origin, ray_target, kdtree, point_cloud, step=0.5, factor=2.0):
    import numpy as np

    radius = estimate_radius_from_pointcloud(point_cloud, factor=factor)
    ray_origin = np.array(ray_origin)
    ray_target = np.array(ray_target)
    direction = ray_target - ray_origin
    distance = np.linalg.norm(direction)
    if distance == 0:
        return False

    direction = direction / distance
    num_steps = int(distance / step)

    for i in range(1, num_steps + 1):
        point = ray_origin + i * step * direction
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if len(idx) > 0:
            return True

    return False

def open3d_to_blender_vec(v):
    # v: np.array([x, y, z]) from Open3D
    return np.array([v[0], -v[2], v[1]])

def estimate_unobserved_main_direction(unobserved_points, center, original_camera_pos):
    """
    利用点投影到球面、与原相机连心线比较方向，返回建议的观察方向（单位向量）。
    """
    vectors = unobserved_points - center
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    dir_mean = np.mean(unit_vectors, axis=0)
    dir_mean /= np.linalg.norm(dir_mean)

    ref_vec = original_camera_pos - center
    ref_vec /= np.linalg.norm(ref_vec)

    if np.dot(dir_mean, ref_vec) > 0:
        dir_mean = -dir_mean  # 翻转方向 ➤ 走另一面去观察

    return dir_mean
  

def find_best_viewpoint(
    unobserved_points,
    cloud_clean_pcd,
    target_pcd,
    original_camera_pos,
    angle_step_deg=15,
    max_angle_deg=135,
):
    """
    寻找一个合法且尽可能接近主法向的相机视角，返回 Blender 可用的位姿（位置 + 欧拉角）。
    使用粗扰动 + 精细扰动 + fallback 三层策略。
    """

    cloud_clean_kdtree = o3d.geometry.KDTreeFlann(cloud_clean_pcd)
    target_kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    center = np.mean(np.asarray(target_pcd.points), axis=0)
    z0 = original_camera_pos[2]

    normal_open3d = estimate_unobserved_main_direction(unobserved_points, center, original_camera_pos)

    # ✅ 转换为 Blender 坐标系下的方向向量
    normal_mean = open3d_to_blender_vec(normal_open3d)
    best_coarse_cam = None

    for angle_deg in range(angle_step_deg, max_angle_deg + 1, angle_step_deg):
        directions = generate_layered_directions(normal_mean, angle_deg, num_directions=4)

        for dir_vec in directions:
            t = compute_valid_camera_distance(center, dir_vec, original_camera_pos)
            if t is None:
                print(f"❌ 距离失败: direction = {dir_vec}, angle = {angle_deg}")
                continue

            cam_pos = center + t * dir_vec
            if cam_pos[2] < 1e-2:
                cam_pos[2] = 1e-2
            if not (0 < cam_pos[2] < 1.5 * z0):
                continue


            blocked_by_env = ray_hits_kdtree(cam_pos, center, cloud_clean_kdtree, cloud_clean_pcd)
            # 相机杆子是否撞到了地面或其他环境障碍
            rod_hits_env = ray_hits_kdtree(cam_pos, [cam_pos[0], cam_pos[1], 0], cloud_clean_kdtree, cloud_clean_pcd)

            if blocked_by_env or rod_hits_env:
                reason = []
                if blocked_by_env:
                    reason.append("❌ 环境遮挡")
                if rod_hits_env:
                    reason.append("❌ 相机杆碰撞")
                print(f"⚠️ 非法视角: angle={angle_deg}°, pos={cam_pos.tolist()} | 原因: {', '.join(reason)}")
                continue

            best_coarse_cam = cam_pos

            refine_angle = angle_deg - angle_step_deg
            if refine_angle > 0:
                fine_dirs = generate_layered_directions(normal_mean, refine_angle, num_directions=24)

                for fine_dir in fine_dirs:
                    t_fine = compute_valid_camera_distance(center, fine_dir, original_camera_pos)
                    cam_pos_fine = center + t_fine * fine_dir

                    if not (0 < cam_pos_fine[2] < 1.5 * z0):
                        continue

                    if (
                        ray_hits_kdtree(cam_pos_fine, center, cloud_clean_kdtree, cloud_clean_pcd) or
                        ray_hits_kdtree(cam_pos_fine, [cam_pos_fine[0], cam_pos_fine[1], 0], cloud_clean_kdtree, cloud_clean_pcd) or
                        ray_hits_kdtree(cam_pos_fine, [cam_pos_fine[0], cam_pos_fine[1], 0], target_kdtree, target_pcd)
                    ):
                        continue

                    return {
                        'location': cam_pos_fine.tolist(),
                        'rotation_euler': look_at_rotation(cam_pos_fine, center).tolist()
                    }
            else:
                # 没有 refine，但已经是第一个合法 coarse 视角了
                print(f"🟢 使用 coarse 视角（无 refine）：{cam_pos.tolist()}")
                return {
                    'location': cam_pos.tolist(),
                    'rotation_euler': look_at_rotation(cam_pos, center).tolist()
                }
    if best_coarse_cam is not None:
        print(f"🟡 使用粗扰动 fallback 视角：{best_coarse_cam.tolist()}")
        return {
            'location': best_coarse_cam.tolist(),
            'rotation_euler': look_at_rotation(best_coarse_cam, center).tolist()
        }

    t_fallback = compute_valid_camera_distance(center, normal_mean, original_camera_pos)
    cam_pos_fallback = center + t_fallback * normal_mean
    print(f"🔴 所有方向非法，使用主法向 fallback 视角。")
    print(f"    ➤ fallback_cam = {cam_pos_fallback.tolist()}")
    return {
        'location': cam_pos_fallback.tolist(),
        'rotation_euler': look_at_rotation(cam_pos_fallback, center).tolist()
    }




# ================== 可见性计算函数 ==================
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

    density = 5
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

    min_visible_votes = len(offsets) * 0.15  # 阈值，支持参数化
    visible_mask = np.array([vote_count[i] >= min_visible_votes for i in range(len(points_world_orig))])

    visible_orig_ids = np.where(visible_mask)[0]
    coverage = len(visible_orig_ids) / len(points_world_orig)
    print(f"📊 当前视角覆盖率: {coverage:.2%} ({len(visible_orig_ids)} / {len(points_world_orig)})")

    return visible_orig_ids

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




# 预设路径和文件位置（根据你的 Blender 输出目录）
BASE = Path("D:/NBV/nbv_simulation/results")
PLY_PATH = BASE / "model_full_transformed.ply"
ENV_PATH = BASE / "cloud_clean.ply"
CAM_PATH = BASE / "camera.json"

# 加载点云和初始相机配置
model_pcd = o3d.io.read_point_cloud(str(PLY_PATH))
env_pcd = o3d.io.read_point_cloud(str(ENV_PATH))
cloud_clean_pcd = env_pcd  # 用于遮挡检测
target_pcd = model_pcd     # 若为裁剪目标，也可替换为子集

with open(CAM_PATH, "r") as f:
    cam = json.load(f)  

# === 初始化 ===
points = np.asarray(model_pcd.points)
total_ids = set(range(len(points)))
lookat = np.mean(points, axis=0)  # 所有视角统一朝向目标中心

# 使用 Blender 相机外参算第一轮可见点（不保存这个相机）
visible_ids = compute_visible_points(model_pcd, env_pcd, cam)
visible_ids = np.array(visible_ids, dtype=int)

# 初始化所有已知可见点
all_visible_ids = set(visible_ids)
visible_ids_sequence = [visible_ids]  # 每一轮新增点，用于可视化
camera_poses = []                     # 只记录 NBV 生成的相机
visualize_visibility_sequence(model_pcd, env_pcd, visible_ids_sequence)

# 相机位置初始化为 Blender 相机的位置
T_world2cam = np.array(cam["extrinsic"])
R = T_world2cam[:3, :3]
t = T_world2cam[:3, 3]
cam_pos = -R.T @ t

# === 开始 NBV 迭代 ===
max_rounds = 5
for i in range(max_rounds - 1):  # 第一轮是 Blender 提供，后续最多 4 次
    # 判断剩余未观测点
    unobserved_ids = np.array(list(total_ids - all_visible_ids))
    if len(unobserved_ids) == 0:
        print(f"✅ 所有点已被观测，终止于第 {i+1} 轮。")
        break

    unobserved_points = points[unobserved_ids]

    # 调用 find_best_viewpoint 获取下一最佳视角
    next_cam = find_best_viewpoint(
        unobserved_points,
        cloud_clean_pcd,
        target_pcd,
        cam_pos
    )
    save_blender_camera_position(
    view_result=next_cam,
    save_dir="D:/NBV/nbv_simulation/results/VisualCameras",
    idx=i
    )  
    print(f"已经保存{i+2}号相机信息")
    # 更新相机位置（用于下一轮迭代）
    cam_pos = np.array(next_cam["location"])
    camera_poses.append(next_cam)  # 记录 NBV 生成的相机信息

    extrinsic = blender_camera_extrinsic(cam_pos, lookat)
    cam_struct = {
    "intrinsic": cam["intrinsic"],   # 保持原始相机的内参
    "extrinsic": extrinsic.tolist()  # 当前新视角的外参
    }

    # 计算当前相机视角的可见点
    visible_ids = compute_visible_points(model_pcd, env_pcd, cam_struct)
    visible_ids = np.array(visible_ids, dtype=int)

    # 提取“新增”的点
    new_ids = np.setdiff1d(visible_ids, list(all_visible_ids))
    visible_ids_sequence.append(new_ids)
    all_visible_ids.update(new_ids.tolist())

    print(f"🔁 Round {i+2}: 新增 {len(new_ids)} 个点，累计 {len(all_visible_ids)}")
    # === 可视化最终结果 ===
    visualize_visibility_sequence(model_pcd, env_pcd, visible_ids_sequence)


