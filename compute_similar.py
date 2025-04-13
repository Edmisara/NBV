import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
import open3d as o3d

# === 文件路径 ===
mesh_path = "D:/NBV/nbv_simulation/data/table_001.obj"
intrinsic_path = "D:/NBV/nbv_simulation/results/intrinsic_matrix.npy"
extrinsic_path = "D:/NBV/nbv_simulation/results/extrinsic_matrix.npy"
target_png_path = "D:/NBV/nbv_simulation/results/objects/fused_table.png"
output_silhouette = "D:/NBV/nbv_simulation/results/objects/matched_silhouette.png"
output_rgb = "D:/NBV/nbv_simulation/results/objects/matched_rgb.png"

# === 获取渲染图像尺寸 ===
target_image = Image.open(target_png_path)
width, height = target_image.size

# === 加载相机参数 ===
intrinsic = np.load(intrinsic_path)
extrinsic = np.load(extrinsic_path)
T_cw = np.linalg.inv(extrinsic)
cam_pos = T_cw[:3, 3]
view_dir = -T_cw[:3, 2]
model_center = cam_pos + view_dir * 1.5

# === 使用 Open3D 加载 .obj 并转换为 trimesh ===
o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
o3d_mesh.compute_vertex_normals()
vertices = np.asarray(o3d_mesh.vertices)
faces = np.asarray(o3d_mesh.triangles)
trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# === 缩放归一 + 居中 + 绕 X 轴旋转 + 平移 ===
bbox = trimesh_obj.bounds
extent = np.linalg.norm(bbox[1] - bbox[0])
scale = 1.0 / extent
trimesh_obj.apply_scale(scale)
trimesh_obj.apply_translation(-trimesh_obj.centroid)
R_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])  # ✅ 只 Rx
trimesh_obj.apply_transform(R_x)
trimesh_obj.apply_translation(model_center)

# === 构造相机 ===
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]
camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=5.0)

# === 构建场景 ===
scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[1.0, 1.0, 1.0])
mesh = pyrender.Mesh.from_trimesh(trimesh_obj, smooth=False)
scene.add(mesh)
scene.add(camera, pose=T_cw)
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=T_cw)

# === 渲染 ===
renderer = pyrender.OffscreenRenderer(width, height)
color, _ = renderer.render(scene)

# === 保存 RGB 彩图（左右镜像还原 GUI 行为）===
color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
color_bgr = cv2.flip(color_bgr, 1)
cv2.imwrite(output_rgb, color_bgr)

# === 提取轮廓图并保存（白色前景 + 透明背景）===
mask = (np.sum(color[:, :, :3], axis=2) > 10).astype(np.uint8) * 255
mask = cv2.flip(mask, 1)
rgba = np.zeros((height, width, 4), dtype=np.uint8)
rgba[mask > 0] = [255, 255, 255, 255]
Image.fromarray(rgba).save(output_silhouette)

print("✅ 渲染完成！")
print(f"🎨 彩色图保存于: {output_rgb}")
print(f"⚪ 轮廓图保存于: {output_silhouette}")
