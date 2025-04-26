import numpy as np
import open3d as o3d

# === 文件路径 ===
pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.pcd"
intrinsic_path = "D:/NBV/nbv_simulation/results/intrinsic_matrix.npy"
extrinsic_path = "D:/NBV/nbv_simulation/results/extrinsic_matrix.npy"

# === 加载点云 ===
pcd = o3d.io.read_point_cloud(pcd_path)

# OpenGL → Open3D 坐标变换（上下 + 前后翻转）
flip_transform = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])
pcd.transform(flip_transform)

# 如果是毫米单位，转为米
pcd.scale(0.001, center=(0, 0, 0))
pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

# === 加载相机参数 ===
K = np.load(intrinsic_path)
extrinsic = np.load(extrinsic_path)

# ✅ 使用实际截图尺寸
width = 1389
height = 768

# === 构造内参对象 ===
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width, height,
                         fx=K[0, 0], fy=K[1, 1],
                         cx=K[0, 2], cy=K[1, 2])

# === 打包为相机参数对象 ===
camera_param = o3d.camera.PinholeCameraParameters()
camera_param.intrinsic = intrinsic
camera_param.extrinsic = extrinsic

# === 可视化窗口 ===
vis = o3d.visualization.Visualizer()
vis.create_window("Camera View Reproduction", width=width, height=height)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(camera_param)

vis.run()
vis.destroy_window()
