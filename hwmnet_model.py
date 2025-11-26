import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ---------------------------
# HWNet-v2 architecture
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)

class HWNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.fc = nn.Linear(256 * 8 * 2, 512)  # final embedding

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

# ---------------------------
# Load converted weights
# ---------------------------
def load_hwnet():
    model = HWNetV2()
    # state_dict = torch.load("hwnetv2_compat.pth", map_location="cpu")
    state_dict = torch.load("hwnetv2.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------------------
# Transform + Extract embedding
# ---------------------------
transform = T.Compose([
    T.Grayscale(),
    T.Resize((64, 256)),
    T.ToTensor(),
])

def get_embedding(model, image_path):
    img = Image.open(image_path)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(x).squeeze(0)
    return feat
