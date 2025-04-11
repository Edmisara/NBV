import open3d as o3d
import numpy as np

def create_scene():
    # 文件路径
    table_path = "D:/NBV/nbv_simulation/data/table.obj"
    chair_path = "D:/NBV/nbv_simulation/data/chair.obj"

    # 加载模型
    table_mesh = o3d.io.read_triangle_mesh(table_path)
    chair_mesh = o3d.io.read_triangle_mesh(chair_path)
    
    table_mesh.compute_vertex_normals()
    chair_mesh.compute_vertex_normals()

    N_table = len(table_mesh.vertices)
    N_chair = len(chair_mesh.vertices)

    # 创建UV坐标，这里为每个顶点分配一个默认的UV坐标 [0, 0]
    table_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.zeros((N_table, 2)))
    chair_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.zeros((N_chair, 2)))
    

    table_texture = o3d.io.read_image("D:/NBV/nbv_simulation/data/texture_table.jpeg")
    chair_texture = o3d.io.read_image("D:/NBV/nbv_simulation/data/texture_chair.jpeg")

    # 将纹理图像附加到模型
    table_mesh.textures = [table_texture]
    chair_mesh.textures = [chair_texture]

    # 设置颜色
    table_mesh.paint_uniform_color([0.6, 0.3, 0.1])
    chair_mesh.paint_uniform_color([0.9, 0.8, 0.1])

    # 缩放桌子
    table_mesh.scale(2.2, center=table_mesh.get_center())

    # 对齐高度
    chair_mesh.translate([0, -9, -28])

    return [table_mesh, chair_mesh]

