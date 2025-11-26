import cv2
import numpy as np
import os

def extract_patches_from_keypoints(
        image_path,
        keypoint_file="filtered_keypoints.npy",
        patch_size=20,
        save_patches=False,
        save_dir="patches"):

    # --------------------------
    # 1) Load image
    # --------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Зураг олдсонгүй: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --------------------------
    # 2) Load filtered keypoints
    # --------------------------
    keypoints = np.load(keypoint_file)
    print(f"Filtered keypoints: {len(keypoints)}")

    half = patch_size // 2
    patches = []
    centers = []

    # --------------------------
    # 3) Extract patches using filtered Harris points
    # --------------------------
    for (x, y) in keypoints:
        x = int(x)
        y = int(y)

        if x - half < 0 or x + half > w:
            continue
        if y - half < 0 or y + half > h:
            continue

        patch = gray[y-half:y+half, x-half:x+half]
        patches.append(patch)
        centers.append((x, y))

    # --------------------------
    # 4) Optional save
    # --------------------------
    if save_patches:
        os.makedirs(save_dir, exist_ok=True)
        for i, p in enumerate(patches):
            cv2.imwrite(os.path.join(save_dir, f"patch_{i:04d}.png"), p)

    print(f"Нийт {len(patches)} patch гарлаа.")
    return patches, centers



# -----------------------------------
# ЖИШЭЭ АШИГЛАЛТ
# -----------------------------------

patches, centers = extract_patches_from_keypoints(
    image_path="D:\\front\\front\\public\\first.jpg",
    keypoint_file="filtered_keypoints.npy",  # Harris.py-гаас гарсан файл
    patch_size=20,
    save_patches=True,
    save_dir="patches_20px"
)


























# import cv2
# import numpy as np
# import os

# def extract_patches(image_path, patch_size=20, threshold_ratio=0.05,
#                     save_patches=False, save_dir="patches"):
#     """
#     Гар бичвэрийн зургаас Harris keypoint ашиглан n×n patch гаргах функц.

#     params:
#         image_path      - Зургийн зам
#         patch_size      - n×n хэмжээ
#         threshold_ratio - Harris threshold (dst.max() * ratio)
#         save_patches    - True бол patch-уудыг зураг болгон хадгална
#         save_dir        - хадгалах хавтас

#     return:
#         patches (list) - numpy array хэлбэртэй patch-ууд
#         centers (list) - (x, y) төв координатууд
#     """

#     # --------------------------
#     # 1) Load image
#     # --------------------------
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Зураг олдсонгүй: {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     # --------------------------
#     # 2) Harris corner detection
#     # --------------------------
#     gray_f = np.float32(gray)
#     dst = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
#     dst = cv2.dilate(dst, None)

#     # threshold
#     threshold = threshold_ratio * dst.max()
#     ys, xs = np.where(dst > threshold)

#     # --------------------------
#     # 3) Extract patches
#     # --------------------------
#     patches = []
#     centers = []
#     half = patch_size // 2

#     for (x, y) in zip(xs, ys):
#         # check boundaries
#         if x - half < 0 or x + half > w or y - half < 0 or y + half > h:
#             continue

#         patch = gray[y-half:y+half, x-half:x+half]
#         patches.append(patch)
#         centers.append((x, y))

#     # --------------------------
#     # 4) Save to folder (optional)
#     # --------------------------
#     if save_patches:
#         os.makedirs(save_dir, exist_ok=True)
#         for i, p in enumerate(patches):
#             cv2.imwrite(os.path.join(save_dir, f"patch_{i:04d}.png"), p)

#     return patches, centers



# # -----------------------------------
# # Жишээ ашиглалт
# # -----------------------------------

# patches, centers = extract_patches(
#     image_path="D:\\front\\front\public\\first.jpg",
#     patch_size=20,
#     threshold_ratio=0.05,
#     save_patches=True,       # хүсвэл False болго
#     save_dir="patches_20px"
# )

# print(f"Нийт {len(patches)} patch гарлаа.")  
