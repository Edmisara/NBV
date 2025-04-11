import os

def batch_rename_obj_files(folder, prefix="cabinet_"):
    for filename in os.listdir(folder):
        if filename.endswith(".obj") and not filename.startswith(prefix):
            old_path = os.path.join(folder, filename)
            new_name = prefix + filename
            new_path = os.path.join(folder, new_name)
            
            if os.path.exists(new_path):
                print(f"❌ 已存在: {new_name}，跳过重命名")
                continue

            os.rename(old_path, new_path)
            print(f"✅ 重命名: {filename} → {new_name}")

if __name__ == "__main__":
    folder_path = "D:/NBV/AM009_OBJ"
    batch_rename_obj_files(folder_path)
