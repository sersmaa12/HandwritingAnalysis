# crop_single_page.py
import os
import cv2
import numpy as np 

# ХЭРЭГЛЭГЧ ТОХИРУУЛГА
INPUT_FILE = "RawPages/S-0135.bmp"                    
OUTPUT_DIR = "Dataset/Authors/076"          
MIN_LINE_HEIGHT = 25 
LINE_MARGIN = 5
TARGET_WIDTH = 1500                          

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_lines_from_page(image_path, outdir):
    ensure(outdir)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot read:", image_path)
        return

    # Бинар болгох
    _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - thr  # бичвэр цагаан, фон хар

    # Хэвтээ проекц
    proj = np.sum(inv, axis=1)
    threshold = np.max(proj) * 0.15

    h, w = img.shape
    lines = []
    in_line = False

    for y in range(h):
        if proj[y] > threshold and not in_line:
            in_line = True
            y_start = y
        elif proj[y] <= threshold and in_line:
            in_line = False
            y_end = y
            if y_end - y_start >= MIN_LINE_HEIGHT:
                lines.append((y_start, y_end))

    if in_line:
        y_end = h - 1
        if y_end - y_start >= MIN_LINE_HEIGHT:
            lines.append((y_start, y_end))

    # Мөрүүдийг хадгалах
    idx = 0
    for (a, b) in lines:
        a = max(0, a - LINE_MARGIN)
        b = min(h, b + LINE_MARGIN)
        line_img = img[a:b, :]

        # resize
        h0, w0 = line_img.shape
        scale = TARGET_WIDTH / w0
        new_h = int(h0 * scale)
        line_img = cv2.resize(line_img, (TARGET_WIDTH, new_h))

        outfile = os.path.join(outdir, f"line_{idx:04d}.png")
        cv2.imwrite(outfile, line_img)
        idx += 1

    print(f"{os.path.basename(image_path)} → {idx} мөр салгалаа.")


if __name__ == "__main__":
    crop_lines_from_page(INPUT_FILE, OUTPUT_DIR)









# import os
# import cv2
# import numpy as np

# # ---------------- CONFIG ----------------
# INPUT_DIR = "RawPages"               # Бүтэн хуудсууд
# OUTPUT_DIR = "Dataset/Authors/A001"  # Мөрүүд хадгалах фолдер
# MIN_LINE_HEIGHT = 25                 # 20–40px бол ихэнх handwriting-д OK
# LINE_MARGIN = 5                      # мөрийн дээр, доор бага хэмжээний buffer
# TARGET_WIDTH = 1500                  # Бүх мөрийг энэхүү өргөнд resize хийнэ
# # ----------------------------------------


# def ensure(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


# def segment_lines(image_path, save_dir, start_index=0):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print("Cannot read:", image_path)
#         return start_index

#     # Бинар болгох
#     _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Инверт хийх — бичвэрийг 255 болгож
#     inv = 255 - thr

#     # Хэвтээ проекц
#     proj = np.sum(inv, axis=1)

#     # босго утга
#     threshold = np.max(proj) * 0.15

#     h, w = img.shape
#     lines = []
#     in_line = False
#     y_start = 0

#     for y in range(h):
#         if proj[y] > threshold and not in_line:
#             in_line = True
#             y_start = y
#         elif proj[y] <= threshold and in_line:
#             in_line = False
#             y_end = y
#             if y_end - y_start >= MIN_LINE_HEIGHT:
#                 lines.append((y_start, y_end))

#     # төгсгөл нь таслагдаагүй бол
#     if in_line:
#         y_end = h - 1
#         if y_end - y_start >= MIN_LINE_HEIGHT:
#             lines.append((y_start, y_end))

#     # Мөр бүрийг тасдаж хадгалах
#     idx = start_index
#     for (a, b) in lines:
#         a = max(0, a - LINE_MARGIN)
#         b = min(h, b + LINE_MARGIN)
#         line_img = img[a:b, :]

#         # өргөн normalize хийх
#         h0, w0 = line_img.shape
#         scale = TARGET_WIDTH / w0
#         new_h = int(h0 * scale)
#         line_img = cv2.resize(line_img, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

#         out_path = os.path.join(save_dir, f"line_{idx:04d}.png")
#         cv2.imwrite(out_path, line_img)
#         idx += 1

#     print(f"{os.path.basename(image_path)} → {idx - start_index} мөр")
#     return idx


# def main():
#     ensure(OUTPUT_DIR)

#     files = [f for f in os.listdir(INPUT_DIR)
#              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

#     idx = 0
#     for f in sorted(files):
#         path = os.path.join(INPUT_DIR, f)
#         idx = segment_lines(path, OUTPUT_DIR, idx)


# if __name__ == "__main__":
#     main()
