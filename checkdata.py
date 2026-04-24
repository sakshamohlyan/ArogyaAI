# check.py
import os

print("=== PNEUMONIA ===")
for cls in ["NORMAL", "PNEUMONIA"]:
    path = f"datasets/pneumonia/train/{cls}"
    count = len(os.listdir(path)) if os.path.exists(path) else "MISSING"
    print(f"  {cls}: {count}")

print("\n=== SKIN ===")
for cls in ["akiec","bcc","bkl","df","mel","nv","vasc"]:
    path = f"datasets/skin/train/{cls}"
    count = len(os.listdir(path)) if os.path.exists(path) else "MISSING"
    print(f"  {cls}: {count}")

print("\n=== MODEL FILES ===")
for f in ["pneumonia_resnet.keras","skin_resnet.keras","pneumonia_model.h5","skin_model.h5"]:
    exists = os.path.exists(f)
    size = f"{os.path.getsize(f)/1e6:.1f} MB" if exists else "-"
    print(f"  {f}: {'EXISTS' if exists else 'MISSING'} {size}")