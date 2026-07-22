import glob
from pathlib import Path
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dir1 = Path(current_dir, "Uniform")
dir2 = Path(current_dir, "Zonal")

#all folders in dir1
folders1 = [f for f in dir1.iterdir() if f.is_dir()]
folders2 = [f for f in dir2.iterdir() if f.is_dir()]
found_pts = []

for folder in folders1:
    #folders in folder
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    for subfolder in subfolders:
        ptPath = Path(subfolder, "Trained_models")
        #glob for .pt files in ptPath
        pt_files = glob.glob(str(ptPath) + "/*.pt")
        found_pts.extend(pt_files)

for folder in folders2:
    #folders in folder
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    for subfolder in subfolders:
        ptPath = Path(subfolder, "Trained_models")
        #glob for .pt files in ptPath
        pt_files = glob.glob(str(ptPath) + "/*.pt")
        found_pts.extend(pt_files)

print(f"Found {len(found_pts)} .pt files to delete.")
print("these are the files:")
for pt_file in found_pts:
    print(pt_file)
input("Press Enter to delete all .pt files in found_pts or Ctrl+C to cancel...")
#delete all .pt files in found_pts
for pt_file in found_pts:
    os.remove(pt_file)
    
