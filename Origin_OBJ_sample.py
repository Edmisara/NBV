import blenderproc as bproc  # 必须为首行
import bpy
import random
import numpy as np
from mathutils import Vector
from pathlib import Path

# === 初始化并加载场景 ===
bproc.init()
bproc.loader.load_blend("D:/NBV/nbv_simulation/view.blend")

# === 参数设置 ===
num_points = 100000
output_base = Path("D:/NBV/nbv_simulation/results/ReferenceCloud")
output_base.mkdir(parents=True, exist_ok=True)

# ✅ 三角化所有 MESH
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')

# ✅ 采样并写PLY（带法线）
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue

    print(f"🎯 正在采样: {obj.name}")
    mesh = obj.data
    faces = [f for f in mesh.polygons if len(f.vertices) == 3]
    verts = mesh.vertices

    sampled_points = []
    sampled_normals = []

    face_data = []
    face_areas = []

    for f in faces:
        v0, v1, v2 = [verts[i].co for i in f.vertices]
        n0, n1, n2 = [verts[i].normal for i in f.vertices]
        area = ((v1 - v0).cross(v2 - v0)).length / 2.0
        face_data.append((v0, v1, v2, n0, n1, n2))
        face_areas.append(area)

    area_sum = sum(face_areas)
    probabilities = [a / area_sum for a in face_areas]

    for _ in range(num_points):
        v0, v1, v2, n0, n1, n2 = random.choices(face_data, weights=probabilities, k=1)[0]
        r1, r2 = random.random(), random.random()
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v

        point_local = u * v0 + v * v1 + w * v2
        normal_local = u * n0 + v * n1 + w * n2

        point_world = obj.matrix_world @ Vector(point_local)
        normal_world = obj.matrix_world.to_3x3() @ Vector(normal_local)
        normal_world.normalize()

        sampled_points.append(point_world[:])
        sampled_normals.append(normal_world[:])

    obj_safe_name = obj.name.replace(" ", "_").replace(".", "_")
    ply_path = output_base / f"{obj_safe_name}_reference_points.ply"

    # === 写PLY（带法线）===
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(sampled_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for pt, n in zip(sampled_points, sampled_normals):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {n[0]} {n[1]} {n[2]}\n")

    print(f"✅ 已完成导出: {ply_path}")

print("🎉 所有对象的带法线点云采样已完成并写入 PLY 文件！")
