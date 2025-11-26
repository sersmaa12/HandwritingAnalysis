import torch
from hwmnet_model import HWNetV2

# HWNet-v2 model
model = HWNetV2()

# PyTorch 1.13-д татсан оригинал файл
orig_file = "hwnetv2.pth"

# PyTorch 1.13 environment-д load
state_dict = torch.load(orig_file, map_location="cpu")

# Шинэ PyTorch 2.x-д нийцүүлэх файл руу хадгалах
torch.save(state_dict, "hwnetv2_compat.pth")
print("✅ HWNet-v2 weights converted successfully to 'hwnetv2_compat.pth'")
