# predict.py
import torch
import torch.nn.functional as F
from skimage import io, color
from ResnetSiamese import ResnetSiamese
from AuthorsDataset import Pad, Threshold, ShiftAndCrop, Downsample


MAXWIDTH = 2260
MAXHEIGHT = 337
THRESHOLD_VALUE = 177
CROP_SIZE = 700
RANDOM_CROP = False
DOWNSAMPLE_RATE = 0.75

# =========================
# Preprocessing pipeline
# =========================
class TransformPipeline:
    def __init__(self):
        self.pad = Pad((MAXWIDTH, MAXHEIGHT))
        self.thresh = Threshold(THRESHOLD_VALUE)
        self.shift = ShiftAndCrop(CROP_SIZE, random=RANDOM_CROP)
        self.down = Downsample(DOWNSAMPLE_RATE)

    def __call__(self, imgs):
        imgs, labels = imgs
        imgs, _ = self.pad((imgs, labels))
        imgs, _ = self.thresh((imgs, labels))
        imgs, _ = self.shift((imgs, labels))
        imgs, _ = self.down((imgs, labels))
        return imgs, _

transform = TransformPipeline()

# =========================
# Load and process image
# =========================
def load_and_process(img_path):
    img = io.imread(img_path)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)  # 2D болгож хувиргана
    img_proc, _ = transform((img, img))
    img_proc = torch.reshape(torch.from_numpy(img_proc), (1, 1, img_proc.shape[0], img_proc.shape[1])).float()
    return img_proc

# =========================
# Model
# =========================
RESNET_LAYERS = [1,1,1,1]
RESNET_OUTSIZE = 10
model = ResnetSiamese(RESNET_LAYERS, RESNET_OUTSIZE)

# Load checkpoint
checkpoint = torch.load("Model_Checkpoints/11-13-2025_12-42-15_epoch20", map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# =========================
# Provide your image paths here
# =========================
IMG1_PATH = "D:\\Data\\Auth_002\\1.png"
IMG2_PATH = "D:\\Data\\Auth_001\\4.png"

img1 = load_and_process(IMG1_PATH)
img2 = load_and_process(IMG2_PATH)

# =========================
# Predict
# =========================
with torch.no_grad():
    output = model(img1, img2)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    same_prob = probs[1]  # "same author"
    diff_prob = probs[0]

print("---- Siamese Authorship Verification ----")
print(f"Image 1: {IMG1_PATH}")
print(f"Image 2: {IMG2_PATH}")
print(f"Same author probability: {same_prob*100:.2f}%")
print(f"Different author probability: {diff_prob*100:.2f}%")

if same_prob > 0.5:
    print("\n✅ These handwriting samples are likely from the SAME author.")
else:
    print("\n❌ These handwriting samples are likely from DIFFERENT authors.")










# from AuthorsDataset import Pad, Threshold, ShiftAndCrop, Downsample
# from skimage import io
# import torch

# MAXWIDTH = 2260
# MAXHEIGHT = 337
# THRESHOLD_VALUE = 177
# CROP_SIZE = 700
# RANDOM_CROP = False
# DOWNSAMPLE_RATE = 0.75

# # List биш callable pipeline
# class TransformPipeline:
#     def __init__(self):
#         self.pad = Pad((MAXWIDTH, MAXHEIGHT))
#         self.thresh = Threshold(THRESHOLD_VALUE)
#         self.shift = ShiftAndCrop(CROP_SIZE, random=RANDOM_CROP)
#         self.down = Downsample(DOWNSAMPLE_RATE)

#     def __call__(self, imgs):
#         imgs, labels = imgs
#         imgs, _ = self.pad((imgs, labels))
#         imgs, _ = self.thresh((imgs, labels))
#         imgs, _ = self.shift((imgs, labels))
#         imgs, _ = self.down((imgs, labels))
#         return imgs, _
        
# transform = TransformPipeline()

# from skimage import io, color


# def load_and_process(img_path):
#     img = io.imread(img_path)
#     if len(img.shape) == 3:
#         img = color.rgb2gray(img)  # 2D болгож хувиргана
#     img_proc, _ = transform((img, img))
#     img_proc = torch.reshape(torch.from_numpy(img_proc), (1, 1, img_proc.shape[0], img_proc.shape[1])).float()
#     return img_proc













# from ResnetSiamese import ResnetSiamese

# # Load checkpoint
# model = ResnetSiamese([1,1,1,1], 10)  # ResNet layer config
# checkpoint = torch.load("Model_Checkpoints/best", map_location='cpu')
# model.load_state_dict(checkpoint['state_dict'])
# model.eval() 

# # Load images
# img1 = load_and_process("D:\\Data\\Auth_001\\2.png")
# img2 = load_and_process("D:\\Data\\Auth_002\\1.png")

# # Predict
# with torch.no_grad():
#     output = model(img1, img2)
#     probs = torch.softmax(output, dim=1).cpu().numpy()[0]
#     same_prob = probs[1]  # "same author"
#     diff_prob = probs[0]

# print(f"Same author probability: {same_prob*100:.2f}%")
# print(f"Different author probability: {diff_prob*100:.2f}%")
