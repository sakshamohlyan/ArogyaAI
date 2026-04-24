import os
print("Class counts:")
for cls in classes:
    folder = os.path.join(base, cls)
    print(cls, len(os.listdir(folder)))