import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Зураг унших
# =========================
img = cv2.imread("D:\\front\\front\\public\\first.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

# =========================
# 2. Harris corner detection
# =========================
dst = cv2.cornerHarris(src=gray_f, blockSize=6, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

# =========================
# 3. Threshold-аас дээш corner-уудыг авах
# =========================
threshold = 0.031 * dst.max()

points = []
responses = []

h, w = dst.shape
for y in range(h):
    for x in range(w):
        if dst[y, x] > threshold:
            points.append((x, y))
            responses.append(dst[y, x])

print("Raw Harris keypoints:", len(points))

# =========================
# 4. NMS + Min-distance filtering
# =========================
def nms_min_dist_filter(points, responses, min_dist=12):
    pts = [(x, y, r) for (x, y), r in zip(points, responses)]
    pts.sort(key=lambda x: x[2], reverse=True)

    filtered = []
    for (x, y, r) in pts:
        too_close = False
        for (fx, fy, fr) in filtered:
            if np.hypot(x - fx, y - fy) < min_dist:
                too_close = True
                break
        if not too_close:
            filtered.append((x, y, r))

    return [(x, y) for (x, y, r) in filtered]

filtered_points = nms_min_dist_filter(points, responses, min_dist=12)

print("Filtered keypoints:", len(filtered_points))

# =========================
# 5. Хадгалах (patch.py эндээс авна)
# =========================
np.save("filtered_keypoints.npy", np.array(filtered_points))
print("filtered_keypoints.npy хадгаллаа.")

# =========================
# 6. Corner visualization (optional)
# =========================
corner_img = img.copy()
for (x, y) in filtered_points:
    cv2.circle(corner_img, (x, y), 3, (0, 0, 255), -1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Filtered keypoints")
plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()













# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # --- 1. Зураг унших ---
# # img = cv2.imread('D:\Hicheel\Diplom\S-0008.bmp')   
# # "D:\CSCI5922-Siamese-CNN-For-Authorship-Verification\Dataset\Authors\001\3.png"
# img = cv2.imread("D:\\front\\front\public\\first.jpg") 

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_f = np.float32(gray)

# # --- 2. Harris detection ---
# dst = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
# dst = cv2.dilate(dst, None)

# # --- 3. Threshold сонгох ---
# threshold = 0.031 * dst.max()

# # --- 4. Keypoint-уудыг дугуйтайгаар тэмдэглэх ---
# vis = img.copy() 

# # (dst > threshold) нөхцөлд true болсон pixel-үүдийн координатыг авах
# ys, xs = np.where(dst > threshold)
 
# for x, y in zip(xs, ys): 
#     # Гадна дугуй (хар хүрээ) 
#     cv2.circle(vis, (x, y), 4, (0, 0, 0), 2)
#     # Дотор улаан цэг
#     cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

# # --- 5. Харагдац ---
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title("Эх бичвэр")
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Harris Keypoints (улаан цэг + дугуй хүрээ)")
# plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.tight_layout()
# plt.show()











# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # --- 1. Зураг унших ---
# img = cv2.imread("D:\\front\\front\public\\first.jpg")
# if img is None:
#     raise FileNotFoundError("handwriting.png олдсонгүй")

# # --- 2. Grayscale + Blur (илүү жигд corner гаргах) ---
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
# gray_f = np.float32(gray_blur)

# # --- 3. Harris detection ---
# dst = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.05)
# dst = cv2.dilate(dst, None)

# # --- 4. Threshold өндөр (цэгийн тоог багасгана) ---
# threshold = 0.1 * dst.max()     

# # --- 5. NON-MAX SUPPRESSION -- давхцсан цэгүүдийг багасгана ---
# dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
# dst_norm = np.uint8(dst_norm)

# # Энд 5x5 цонх ашиглаж ойр байрлах corner-уудыг нэгтгэж байна
# kernel = np.ones((5,5), np.uint8)
# dst_dilated = cv2.dilate(dst_norm, kernel)
# mask = (dst_norm == dst_dilated)

# # Final keypoints
# ys, xs = np.where((dst > threshold) & mask)

# # --- 6. Хар дугуй + улаан цэгээр дүрслэх ---
# vis = img.copy()
# for x, y in zip(xs, ys):
#     cv2.circle(vis, (x, y), 6, (0,0,0), 2)   # гадна дугуй
#     cv2.circle(vis, (x, y), 3, (0,0,255), -1) # дотор улаан цэг

# # --- 7. Харагдац ---
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.title("Эх бичвэр")
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Harris Keypoints (судалгааны хэв маяг)")
# plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.tight_layout()
# plt.show()




