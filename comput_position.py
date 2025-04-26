import numpy as np
import open3d as o3d
import trimesh
import cv2
from PIL import Image

# === 文件路径 ===
intrinsic_path = "D:/NBV/nbv_simulation/results/intrinsic_matrix.npy"
fused_img_path = "D:/NBV/nbv_simulation/results/objects/fused_table.png"
rendered_img_path = "D:/NBV/nbv_simulation/results/objects/matched_silhouette.png"
obj_path = "D:/NBV/nbv_simulation/data/table_001.obj"
fused_pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.pcd"

# === 加载图像并生成轮廓 ===
def load_mask(path):
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    return (alpha > 10).astype(np.uint8) * 255, img.size

fused_mask, fused_size = load_mask(fused_img_path)
rendered_mask, rendered_size = load_mask(rendered_img_path)

# === 图像匹配参数 ===
target_size = (512, 512)
fused_resized = cv2.resize(fused_mask, target_size, interpolation=cv2.INTER_NEAREST)
rendered_resized = cv2.resize(rendered_mask, target_size, interpolation=cv2.INTER_NEAREST)

def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, w, h

x1, y1, w1, h1 = get_bounding_box(fused_resized)
x2, y2, w2, h2 = get_bounding_box(rendered_resized)
du_px = (x1 + w1 / 2) - (x2 + w2 / 2)
dv_px = (y1 + h1 / 2) - (y2 + h2 / 2)
scale_factor = w1 / w2

# === 相机内参处理 ===
intrinsic = np.load(intrinsic_path)
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
orig_w, orig_h = fused_size
fx_scaled = fx * (target_size[0] / orig_w)
fy_scaled = fy * (target_size[1] / orig_h)

Z0 = 1.5
dx = du_px * Z0 / fx_scaled
dy = dv_px * Z0 / fy_scaled
Z1 = Z0 / scale_factor
dz = Z1 - Z0

print(f"\n📐 图像反推平移:")
print(f"dx = {dx:.4f}, dy = {dy:.4f}, dz = {dz:.4f}")

# === 加载并修正点云 ===
fused_pcd = o3d.io.read_point_cloud(fused_pcd_path)
fused_pcd.scale(0.001, center=(0, 0, 0))  # 毫米 → 米单位
fused_pcd.paint_uniform_color([0.5, 0.5, 0.5])

pts = np.asarray(fused_pcd.points)
x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
z_min, z_max = np.min(pts[:, 2]), np.max(pts[:, 2])
fused_width = x_max - x_min

print(f"\n🌐 点云范围:")
print(f"X: {x_min:.4f} → {x_max:.4f}  ΔX = {fused_width:.4f}m")
print(f"Y: {y_min:.4f} → {y_max:.4f}")
print(f"Z: {z_min:.4f} → {z_max:.4f}")

# === 加载 obj 模型 ===
mesh_o3d = o3d.io.read_triangle_mesh(obj_path)
mesh_o3d.compute_vertex_normals()
vertices = np.asarray(mesh_o3d.vertices)
faces = np.asarray(mesh_o3d.triangles)
mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

obj_width = mesh_tri.bounds[1][0] - mesh_tri.bounds[0][0]
real_scale = fused_width / obj_width
print(f"\n🔁 模型缩放比例: {real_scale:.4f}")

# === 模型变换流程 ===
mesh_tri.apply_scale(real_scale)
mesh_tri.apply_translation(-mesh_tri.centroid)
R_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
mesh_tri.apply_transform(R_x)
mesh_tri.apply_translation([dx, dy, dz])

# === 转为 Open3D 可视化 ===
mesh_final = o3d.geometry.TriangleMesh()
mesh_final.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
mesh_final.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
mesh_final.compute_vertex_normals()
mesh_final.paint_uniform_color([1.0, 0.0, 0.0])  # 红色 = 预测模型

v = np.asarray(mesh_final.vertices)
print(f"\n📍 最终模型范围:")
print(f"X: {np.min(v[:, 0]):.3f} → {np.max(v[:, 0]):.3f}")
print(f"Y: {np.min(v[:, 1]):.3f} → {np.max(v[:, 1]):.3f}")
print(f"Z: {np.min(v[:, 2]):.3f} → {np.max(v[:, 2]):.3f}")

# === 可视化结果 ===
o3d.visualization.draw_geometries([fused_pcd, mesh_final])
