import numpy as np
import matplotlib.pyplot as plt
from patch import extract_patches_from_keypoints
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------
# 1. Compute NGC similarity
# ---------------------------------------------------------
def compute_ngc_matrix(patches):
    vectors = [p.flatten().astype(np.float32) for p in patches]

    # normalize vectors
    norm_vecs = []
    for v in vectors:
        v_norm = v - np.mean(v)
        denom = np.std(v)
        if denom < 1e-6:
            denom = 1e-6
        v_norm = v_norm / denom
        norm_vecs.append(v_norm)

    norm_vecs = np.array(norm_vecs)
    N = len(norm_vecs)

    sim = np.zeros((N, N), dtype=np.float32)

    # Compute similarity (NGC)
    for i in range(N):
        for j in range(i, N):
            c = np.dot(norm_vecs[i], norm_vecs[j]) / len(norm_vecs[i])
            sim[i, j] = c
            sim[j, i] = c

    return sim



# ---------------------------------------------------------
# 2. Agglomerative clustering (complete-linkage)
# ---------------------------------------------------------
def cluster_patches(similarity_matrix, n_clusters=12):
    distance_matrix = 1 - similarity_matrix

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="complete"
    )

    labels = model.fit_predict(distance_matrix)
    return labels



# ---------------------------------------------------------
# 3. Visualizing patch samples per cluster (Figure 6 style)
# ---------------------------------------------------------
def visualize_clusters(patches, labels, max_per_cluster=10):

    unique_clusters = np.unique(labels)
    C = len(unique_clusters)

    plt.figure(figsize=(15, C * 2))
    idx = 1

    for c in unique_clusters:
        members = np.where(labels == c)[0]
        selected = members[:max_per_cluster]

        for m in selected:
            plt.subplot(C, max_per_cluster, idx)
            plt.imshow(patches[m], cmap="gray")
            plt.title(f"C{c}")
            plt.axis("off")
            idx += 1

    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

# 1) Filtered keypoints → patches
patches, centers = extract_patches_from_keypoints(
    image_path="D:\\front\\front\\public\\first.jpg",
    keypoint_file="filtered_keypoints.npy",
    patch_size=20
)

print(f"Patch count: {len(patches)}")

# 2) NGC similarity
sim = compute_ngc_matrix(patches)

# 3) Clustering
labels = cluster_patches(similarity_matrix=sim, n_clusters=12)

# 4) Visualization
visualize_clusters(patches, labels)

















# import numpy as np

# def compute_ngc_matrix(patches):

#     # patches -> vector
#     vectors = [p.flatten().astype(np.float32) for p in patches]

#     # normalize: remove mean, divide by std
#     norm_vecs = []
#     for v in vectors:
#         v_norm = (v - np.mean(v))
#         denom = np.std(v)
#         if denom < 1e-6: denom = 1e-6
#         v_norm = v_norm / denom
#         norm_vecs.append(v_norm)

#     norm_vecs = np.array(norm_vecs)

#     # NGC similarity = dot product / length
#     N = len(norm_vecs)
#     sim = np.zeros((N, N), dtype=np.float32)

#     for i in range(N):
#         for j in range(i, N):
#             c = np.dot(norm_vecs[i], norm_vecs[j]) / len(norm_vecs[i])
#             sim[i, j] = c
#             sim[j, i] = c

#     return sim




# from sklearn.cluster import AgglomerativeClustering

# def cluster_patches(similarity_matrix, n_clusters=10, linkage='complete'):
#     # similarity → distance
#     distance_matrix = 1 - similarity_matrix

#     # Agglomerative clustering (Bennoura complete linkage ашигласан)
#     model = AgglomerativeClustering( 
#         metric='precomputed',
#         linkage=linkage,
#         n_clusters=n_clusters 
#     )
#     labels = model.fit_predict(distance_matrix)
#     return labels





# import matplotlib.pyplot as plt
# from patch import extract_patches

# def visualize_clusters(patches, labels, max_per_cluster=10):
#     """
#     patches: list of patch images (2D)
#     labels: cluster labels
#     max_per_cluster: кластер бүрээс хэдэн patch харуулах вэ
#     """

#     unique_clusters = np.unique(labels)
#     cluster_count = len(unique_clusters)

#     plt.figure(figsize=(15, cluster_count * 2))

#     plot_idx = 1

#     for c in unique_clusters:
#         cluster_indices = np.where(labels == c)[0]

#         # max_per_cluster хүртэл patch сонгоно
#         selected = cluster_indices[:max_per_cluster]

#         for idx in selected:
#             plt.subplot(cluster_count, max_per_cluster, plot_idx)
#             plt.imshow(patches[idx], cmap='gray')
#             plt.title(f"C{c}")
#             plt.axis("off")
#             plot_idx += 1

#     plt.tight_layout()
#     plt.show()




# # 1. Patch-уудыг гаргаж авсан гэж үзье
# patches, centers = extract_patches("D:\\front\\front\public\\first.jpg", patch_size=20)

# # 2. NGC similarity matrix
# sim = compute_ngc_matrix(patches)

# # 3. Clustering
# labels = cluster_patches(similarity_matrix=sim, n_clusters=12)

# # 4. Visualization (Figure 6 хэлбэрээр)
# visualize_clusters(patches, labels)

