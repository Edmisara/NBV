import open3d as o3d
import numpy as np
import copy
from PIL import Image
from create_scene import create_scene



def render_scene(geometry_list, camera_position, camera_euler):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Scene with Camera", visible=True)

    # 添加场景几何体
    for g in geometry_list:
        vis.add_geometry(g)  # 不使用 deepcopy，直接添加



    # 设置初始视角（可以根据需要调整）
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, 0, 1])
    vis.get_view_control().set_front([0, -1, 0])
    vis.get_view_control().set_zoom(0.5)


    vis.run()
    vis.destroy_window()


# 主程序入口
if __name__ == "__main__":
    scene = create_scene()
    camera_position = np.array([0, 0, 0])  # 相机位置
    camera_euler = np.radians([90, 0, 0])  # 相机欧拉角 (Pitch, Yaw, Roll)
    render_scene(scene, camera_position, camera_euler)
