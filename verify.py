import torch
import torch.nn.functional as F
from hwmnet_model import load_hwnet, get_embedding

# Зургийн зам
img1 = "D:\\Data\\Auth_001\\3.png"
img2 = "D:\\Data\\Auth_001\\4.png"

# Модель ачаалах
model = load_hwnet()

# Эмбеддинг гаргах
f1 = get_embedding(model, img1)
f2 = get_embedding(model, img2)

# Cosine similarity → probability
cos = F.cosine_similarity(f1, f2, dim=0).item()
prob = (cos + 1) / 2  # 0..1 болгож хувиргана

print(f"Cosine similarity: {cos:.4f}")
print(f"Same writer probability: {prob*100:.2f}%")
