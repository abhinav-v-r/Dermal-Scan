import os
base_dir="/Dermal_Scan/dataset"
classes = os.listdir(base_dir)
print("Classes found:", classes)

for cls in classes:
    folder = os.path.join(base_dir, cls)
    print(cls, ":", len(os.listdir(folder)), "images")