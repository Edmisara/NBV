import os
import numpy as np
import open3d as o3d
from PIL import Image
import json

def save_image_from_camera(vis, path="D:/NBV/nbv_simulation/results"):
    # 确保保存路径存在
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"目录 {path} 创建成功。")
    else:
        print(f"目录 {path} 已存在。")

    # 获取相机参数
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic
    extrinsic = params.extrinsic

    # 保存内外参
    intrinsic_matrix = np.array([
        [intrinsic.get_focal_length()[0], 0, intrinsic.get_principal_point()[0]],
        [0, intrinsic.get_focal_length()[1], intrinsic.get_principal_point()[1]],
        [0, 0, 1]
    ])
    np.save(os.path.join(path, "intrinsic_matrix.npy"), intrinsic_matrix)
    np.save(os.path.join(path, "extrinsic_matrix.npy"), extrinsic)

    print(f"✅ 相机参数已保存到 {path}")

    # 捕获 RGB 图像
    rgb_image = vis.capture_screen_float_buffer(do_render=True)
    rgb_image_np = np.asarray(rgb_image) * 255
    rgb_image_pil = Image.fromarray(rgb_image_np.astype(np.uint8))
    rgb_image_pil.save(os.path.join(path, "camera_view.png"))

    # 捕获深度图像
    depth_image = vis.capture_depth_float_buffer(do_render=True)
    depth_image_np = np.asarray(depth_image)
    depth_image_pil = Image.fromarray(depth_image_np.astype(np.uint8))
    depth_image_pil.save(os.path.join(path, "depth_image.png"))

    # 生成 RGBD 图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image_np.astype(np.uint8)),
        o3d.geometry.Image(depth_image_np.astype(np.uint8)),
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    # 相机内参构造
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(
        width=rgb_image_np.shape[1],
        height=rgb_image_np.shape[0],
        fx=intrinsic.get_focal_length()[0],
        fy=intrinsic.get_focal_length()[1],
        cx=intrinsic.get_principal_point()[0],
        cy=intrinsic.get_principal_point()[1]
    )

    # 点云生成与坐标修正
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(os.path.join(path, "point_cloud.pcd"), pcd)
    print(f"✅ 点云已保存为 {os.path.join(path, 'point_cloud.pcd')}")
