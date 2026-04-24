# pretrain.py  —  run once: python pretrain.py
from app import train_pneumonia, train_skin
print("Training pneumonia...")
train_pneumonia()
print("Training skin...")
train_skin()
print("Done. Models saved. You can now run python app.py safely.")