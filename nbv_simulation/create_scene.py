import open3d as o3d
import numpy as np

def create_scene():
    # 文件路径
    table_path = "D:/NBV/nbv_simulation/data/DiningTable6Seat.stl"
    chair_path = "D:/NBV/nbv_simulation/data/VicSideChair.stl"

    # 加载模型
    table_mesh = o3d.io.read_triangle_mesh(table_path)
    chair_mesh = o3d.io.read_triangle_mesh(chair_path)

    # 设置颜色
    table_mesh.paint_uniform_color([0.6, 0.3, 0.1])
    chair_mesh.paint_uniform_color([0.9, 0.8, 0.1])

    # 缩放桌子
    table_mesh.scale(2.2, center=table_mesh.get_center())

    # 对齐高度
    table_bottom = np.min(np.asarray(table_mesh.vertices)[:, 2])
    chair_bottom = np.min(np.asarray(chair_mesh.vertices)[:, 2])
    height_diff = table_bottom - chair_bottom
    chair_mesh.translate([0, 24, height_diff])

    # 创建地面
    ground_size = 150
    ground = o3d.geometry.TriangleMesh.create_box(width=ground_size, height=0.1, depth=ground_size)
    ground.paint_uniform_color([0.7, 0.7, 0.7])
    R = table_mesh.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0])
    ground.rotate(R)
    ground.translate([-75, 0, table_bottom - 75])

    return [ground, table_mesh, chair_mesh]

