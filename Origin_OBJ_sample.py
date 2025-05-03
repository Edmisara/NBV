import blenderproc as bproc  # BlenderProc 必须为首行
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

# ✅ 先统一对所有 MESH 执行一次三角化（只做一次）
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')

# ✅ 对每个对象采样并导出PLY
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue

    print(f"🎯 正在采样: {obj.name}")
    mesh = obj.data
    faces = [f for f in mesh.polygons if len(f.vertices) == 3]
    verts = mesh.vertices

    sampled_points = []

    for _ in range(num_points):
        face = random.choice(faces)
        v0, v1, v2 = [verts[i].co for i in face.vertices]

        r1, r2 = random.random(), random.random()
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v

        point_local = u * v0 + v * v1 + w * v2
        point_world = obj.matrix_world @ Vector(point_local)
        sampled_points.append(point_world[:])

    # 写出PLY文件
    obj_safe_name = obj.name.replace(" ", "_").replace(".", "_")
    ply_path = output_base / f"{obj_safe_name}_reference_points.ply"
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(sampled_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for pt in sampled_points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

    print(f"✅ 已完成导出: {ply_path}")

print("🎉 所有对象的参考点云已采样并写入 PLY 文件！保存路径：", output_base)
