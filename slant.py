# import cv2
# import numpy as np
# from skimage.transform import radon

# def handwriting_slant_angle(image_path):

#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError("Зургаа зөв зааж өгнө үү.")

#     _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     bw = bw.astype(np.float32)

#     theta = np.arange(-45, 46, 1)
#     sinogram = radon(bw, theta=theta, circle=False)
    
#     projection_sums = np.sum(sinogram, axis=0)
    
#     slant_angle = theta[np.argmax(projection_sums)]
    
#     return slant_angle


# slant = handwriting_slant_angle("D:\\Data\\Auth_004\\1.png")
# print("Бичгийн налуугийн хэм:", slant, "градус")

# slant1 = handwriting_slant_angle("D:\\Data\\Auth_004\\2.png")
# print("Бичгийн налуугийн хэм:", slant1, "градус") 

# slant2 = handwriting_slant_angle("D:\\Data\\Auth_004\\3.png")
# print("Бичгийн налуугийн хэм:", slant2, "градус") 



import cv2
import numpy as np
from skimage.transform import radon

def handwriting_slant_angle(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Зургаа зөв зааж өгнө үү.")
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = bw.astype(np.float32)
    theta = np.arange(-45, 46, 1)
    sinogram = radon(bw, theta=theta, circle=False)
    projection_sums = np.sum(sinogram, axis=0)
    slant_angle = theta[np.argmax(projection_sums)]
    # Чиглэл тодорхойлох
    if slant_angle > 0:
        direction = "[translate:Баруун тийш хазайсан]"
    elif slant_angle < 0:
        direction = "[translate:Зүүн тийш хазайсан]"
    else:
        direction = "[translate:Тэгш бичиг]"
    return slant_angle, direction

slant, dirn = handwriting_slant_angle("D:\\Data\\Auth_004\\1.png")
print("Бичгийн налуугийн хэм:", slant, "градус,", dirn)

slant1, dirn1 = handwriting_slant_angle("D:\\Data\\Auth_004\\2.png")
print("Бичгийн налуугийн хэм:", slant1, "градус,", dirn1) 

slant2, dirn2 = handwriting_slant_angle("D:\\Data\\Auth_004\\5.png")
print("Бичгийн налуугийн хэм:", slant2, "градус,", dirn2)  
