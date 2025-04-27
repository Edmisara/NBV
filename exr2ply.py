import numpy as np
import json
import os
import cv2  # 可选，只用于读取 RGB 图像

# === 路径设置 ===
base_path = "D:/NBV/nbv_simulation/results"
depth_path = os.path.join(base_path, "depth_meter.npy")
camera_path = os.path.join(base_path, "camera.json")
rgb_path = os.path.join(base_path, "rgb.png")
ply_output_path = os.path.join(base_path, "cloud.ply")

# === 读取深度图（单位：米） ===
depth = np.load(depth_path)

# === 读取相机参数（你的自定义格式） ===
with open(camera_path, "r") as f:
    cam = json.load(f)

fx = cam["intrinsic"]["fx"]
fy = cam["intrinsic"]["fy"]
cx = cam["intrinsic"]["cx"]
cy = cam["intrinsic"]["cy"]
width = cam["intrinsic"]["width"]
height = cam["intrinsic"]["height"]
T_world2cam = np.array(cam["extrinsic"])
T_cam2world = np.linalg.inv(T_world2cam)

# === 反投影为点云（单位米） ===
points = []
colors = []

if os.path.exists(rgb_path):
    rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
else:
    rgb_img = None

for v in range(height):
    for u in range(width):
        z = depth[v, u]
        if z == 0:
            continue
        # ✅ 使用方向单位向量 × 欧几里得距离（正确反投影）
        x = (u - cx) / fx
        y = -(v - cy) / fy
        ray = np.array([x, y, -1.0])
        ray /= np.linalg.norm(ray)
        point_cam = ray * z
        
        # 转为齐次坐标再应用 T_cam2world
        cam_point_h = np.concatenate([point_cam, [1.0]])
        world_point = T_cam2world @ cam_point_h
        points.append(world_point[:3])
        # **反转Z轴**，使得它符合Blender的Z-up标准
        #world_point[2] = -world_point[2]  # 反转Z轴

        # **修正点云位置**，添加相机的偏移量（例如，原相机位置偏移）
        ##camera_offset = np.array([-120.0, 120.0, 80.0])  # 相机原位置偏移
        #world_point[:3] += camera_offset  # 修正点云位置

        if rgb_img is not None:
            r, g, b = rgb_img[v, u]
            colors.append((r, g, b))

# === 写入 PLY 文件 ===
with open(ply_output_path, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    if rgb_img is not None:
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")

    for i, pt in enumerate(points):
        line = f"{pt[0]} {pt[1]} {pt[2]}"
        if rgb_img is not None:
            r, g, b = colors[i]
            line += f" {r} {g} {b}"
        f.write(line + "\n")

print(f"✅ PLY 点云已保存：{ply_output_path}（共 {len(points)} 点）")
