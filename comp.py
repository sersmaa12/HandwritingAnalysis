# import cv2
# from skimage.metrics import structural_similarity as ssim
# import numpy as np

# # Зургийн захын цагаан орон зайг хасах функц
# def removeWhiteSpace(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return img
#     x, y, w, h = cv2.boundingRect(contours[0])
#     return img[y:y+h, x:x+w]

# # Хоёр үгийн зургийн төстэй байдлыг тооцоолох функц
# def compare_words(image_path1, image_path2):
#     img1 = cv2.imread(image_path1)
#     img2 = cv2.imread(image_path2)

#     img1 = removeWhiteSpace(img1)
#     img2 = removeWhiteSpace(img2)

#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

#     score, diff = ssim(img1, img2, full=True)
#     return score

# # Жишээ хэрэглээ
# image1 = "D:\\Data\\Auth_001\\2.png"
# image2 = "D:\\Data\\Auth_001\\1.png"
# similarity_score = compare_words(image1, image2)
# print(f'Үгсийн төстэй байдал: {similarity_score:.3f}')


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import cv2

class SiamesePretrained(nn.Module):
    def __init__(self):
        super(SiamesePretrained, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # fc layer-гүй
        self.fc = nn.Linear(512, 256)

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img = img.repeat(1,3,1,1)  # grayscale to 3 channel for ResNet
    return img

def calculate_similarity(output1, output2):
    distance = F.pairwise_distance(output1, output2)
    similarity = torch.exp(-distance)
    return similarity

if __name__ == "__main__":
    net = SiamesePretrained()
    img1 = preprocess_image('D:\\Data\\Auth_001\\3.png')
    img2 = preprocess_image('D:\\Data\\Auth_001\\2.png')
    out1, out2 = net(img1, img2) 
    sim = calculate_similarity(out1, out2)
    print("Similarity:", sim.item())
