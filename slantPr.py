import cv2
import numpy as np
import math

def load_binary_word(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Зураг алга.")
    
    _, bw = cv2.threshold(img, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)
    return bw 

def horizontal_black_run_profile(bw):
    Iy, Ix = bw.shape  # height, width
    H = np.zeros(Iy, dtype=np.float64)

    for y in range(Iy):
        row = bw[y, :]
        # black runs (1-үүдийн тасралтгүй сегмент)
        runs = []
        in_run = False
        start = 0
        for x in range(Ix):
            if row[x] == 1 and not in_run:
                in_run = True
                start = x
            elif row[x] == 0 and in_run:
                in_run = False
                runs.append((start, x-1))
        if in_run:
            runs.append((start, Ix-1))

        B = len(runs)
        if B == 0:
            H[y] = 0
            continue

        # H(y) = [B(y)]^2 * sum_i sum_{j=0}^{L(i,y)} j
        total = 0
        for (s, e) in runs:
            L = e - s + 1
            # sum_{j=0}^{L} j = L*(L+1)/2
            total += L * (L + 1) / 2.0

        H[y] = (B ** 2) * total

    return H  # H(y)


def detect_core_region(H):
    Iy = len(H)
    Th = 0.15 * (np.sum(H) / Iy)  

    HB = (H > Th).astype(np.uint8)  

    
    segments = []
    in_seg = False
    s = 0
    for y in range(Iy):
        if HB[y] == 1 and not in_seg:
            in_seg = True
            s = y
        elif (HB[y] == 0 or y == Iy-1) and in_seg:
            e = y-1 if HB[y] == 0 else y
            segments.append((s, e))
            in_seg = False

    if not segments:
        # fallback: бүх мөрийг core гэж үзэх
        return 0, Iy-1, HB

    # Eq. (4) – хамгийн их Σ H(y) бүхий сегмент сонгох
    best_idx = -1
    best_sum = -1
    for i, (s, e) in enumerate(segments):
        seg_sum = np.sum(H[s:e+1])
        if seg_sum > best_sum:
            best_sum = seg_sum
            best_idx = i

    UB, LB = segments[best_idx]  # upper baseline, lower baseline
    return UB, LB, HB


def compute_slant(bw, UB, LB):
    Iy, Ix = bw.shape

    # 1) бүх мөрүүдийн black run length-ээс stroke өргөн λ (mode) олох
    run_lengths = []
    for y in range(Iy):
        row = bw[y, :]
        in_run = False
        start = 0
        for x in range(Ix):
            if row[x] == 1 and not in_run:
                in_run = True
                start = x
            elif row[x] == 0 and in_run:
                in_run = False
                L = x - start
                if L > 0:
                    run_lengths.append(L)
        if in_run:
            L = Ix - start
            if L > 0:
                run_lengths.append(L)

    if not run_lengths:
        return 0.0

    # modal λ-ийг ойролцоогоор histogram-аар
    hist, bin_edges = np.histogram(run_lengths, bins=20)
    mode_idx = np.argmax(hist)
    lam = (bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2.0

    M = 2.5 * lam  # Eq. (5)

    # 2) M-с урт black run агуулсан мөрүүдийг устгах
    keep = np.ones(Iy, dtype=bool)
    for y in range(Iy):
        row = bw[y, :]
        in_run = False
        start = 0
        long_run = False
        for x in range(Ix):
            if row[x] == 1 and not in_run:
                in_run = True
                start = x
            elif row[x] == 0 and in_run:
                in_run = False
                L = x - start
                if L > M:
                    long_run = True
                    break
        if in_run:
            L = Ix - start
            if L > M:
                long_run = True
        if long_run:
            keep[y] = False

    bw2 = bw.copy()
    for y in range(Iy):
        if not keep[y]:
            bw2[y, :] = 0

    # 3) vertical-аар таслагдсан box-ууд үүсгэх
    boxes = []
    visited = np.zeros_like(bw2, dtype=bool)

    for y in range(Iy):
        for x in range(Ix):
            if bw2[y, x] == 1 and not visited[y, x]:
                # flood fill / BFS ашиглаж connected component авах
                stack = [(y, x)]
                min_y, max_y = y, y
                min_x, max_x = x, x
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < Iy and 0 <= nx < Ix:
                                if bw2[ny, nx] == 1 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))
                boxes.append((min_x, min_y, max_x, max_y))

    # 4) box бүрийн slant s_i, height h_i, weight c_i
    s_list = []
    h_list = []
    c_list = []

    for (min_x, min_y, max_x, max_y) in boxes:
        h = max_y - min_y + 1
        if h < 3:
            continue

        # box-ыг дотор нь хуваагаад upper/lower half centroid
        box = bw2[min_y:max_y+1, min_x:max_x+1]
        bh, bw_ = box.shape
        mid = bh // 2

        upper = box[:mid, :]
        lower = box[mid:, :]

        if upper.sum() == 0 or lower.sum() == 0:
            continue

        # centroid (x, y) – image координат руу буцаан шилжүүлэх
        uy, ux = np.argwhere(upper).mean(axis=0)
        ly, lx = np.argwhere(lower).mean(axis=0)

        uy_global = min_y + uy
        ux_global = min_x + ux
        ly_global = min_y + mid + ly
        lx_global = min_x + lx

        # slant s_i = angle between vertical and line (ux->lx, uy->ly)
        dy = ly_global - uy_global
        dx = lx_global - ux_global
        if dy == 0:
            continue
        angle_rad = math.atan2(dx, dy)  # dx/dy – vertical-ээс хазайлт
        s_i = angle_rad * 180.0 / math.pi  # degrees

        # weight c_i (Eq. (7))
        yti = min_y
        ybi = max_y
        if (ybi < LB) and (yti > UB):
            c_i = 1.0
        else:
            c_i = 2.0

        s_list.append(s_i)
        h_list.append(h)
        c_list.append(c_i)

    if not s_list:
        return 0.0

    s_arr = np.array(s_list)
    h_arr = np.array(h_list)
    c_arr = np.array(c_list)

    # Eq. (6)
    S = np.sum(s_arr * h_arr * c_arr) / np.sum(h_arr * c_arr)
    return S, boxes


def word_slant_papandreou(image_path):
    bw = load_binary_word(image_path)
    H = horizontal_black_run_profile(bw)
    UB, LB, HB = detect_core_region(H)
    S, boxes = compute_slant(bw, UB, LB)

    if S > 0:
        direction = "Зүүн хазайсан"
    elif S < 0:
        direction = "Баруун хазайсан"
    else:
        direction = "Тэгш"

    return S, direction, UB, LB, boxes 



import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_ub_lb_boxes(image_path, UB, LB, boxes):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w = img_gray.shape

    # Baseline зурах 
    cv2.line(img_vis, (0, UB), (w-1, UB), (255, 0, 0), 2)   # UB (blue)
    cv2.line(img_vis, (0, LB), (w-1, LB), (0, 0, 255), 2)   # LB (red)
    # Box-ууд зурах
    for (min_x, min_y, max_x, max_y) in boxes:
        cv2.rectangle(img_vis, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# path = "D:\\Hicheel\\Diplom\\S-0004.BMP" 
path = "D:\\Data\\Auth_006\\6.png"
S, dirn, UB, LB, boxes = word_slant_papandreou(path)
print(f"Slant S = {S:.2f} градус, {dirn}, UB={UB}, LB={LB}")
draw_ub_lb_boxes(path, UB, LB, boxes)


# path1 = "D:\\Data\\Authors\\001\\1.png"
# S1, dirn1, UB1, LB1 = word_slant_papandreou(path1)
# print(f"Slant1 S = {S1:.2f} градус, {dirn1}, UB={UB1}, LB={LB1}")    