import kagglehub
import os
import shutil

# Download the dataset using kagglehub
print("Downloading dataset...")
path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")

print("Path to dataset files:", path)

# Copy the files to the current directory for easier access
target_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, target_dir)
        print(f"Copied {file_name} to {target_dir}")

print("Dataset ready in ./data directory.")
