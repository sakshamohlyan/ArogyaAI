import os
import pandas as pd
import shutil

base = "datasets/skin"
meta = pd.read_csv(os.path.join(base, "HAM10000_metadata.csv"))

src1 = os.path.join(base, "HAM10000_images_part_1")
src2 = os.path.join(base, "HAM10000_images_part_2")
dest = os.path.join(base, "train")

os.makedirs(dest, exist_ok=True)

for cls in meta['dx'].unique():
    os.makedirs(os.path.join(dest, cls), exist_ok=True)

for _, row in meta.iterrows():
    img = row['image_id'] + ".jpg"
    label = row['dx']

    src_path = os.path.join(src1, img)
    if not os.path.exists(src_path):
        src_path = os.path.join(src2, img)

    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(dest, label, img))

print("DONE")