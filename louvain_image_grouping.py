import numpy as np
import re
from collections import defaultdict
import networkx as nx
from community import community_louvain
import random
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import math
from itertools import combinations

from vggt.models.vggt_classification import VGGT

img_name2id = {}
random.seed(42)

def parse_image_map_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            img_id, image_name = line.split(' ')
            img_name2id.update({image_name.strip(): int(img_id)})

def parse_image_pair_file(file_path):
    # 用于存储边信息
    edges = []
    G = nx.Graph()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            img1, img2 = parts[2], parts[3]
            inliers = int(parts[4])
            matches = int(parts[5])

            # quaternion 部分在 parts[5:11]，里面有 "+"，需要过滤
            q_tokens = [p for p in parts[6:13] if p != '+']
            if len(q_tokens) != 4:
                raise ValueError(f"无法解析 quaternion: {parts[6:12]}")

            # 去掉 i,j,k
            def clean_token(tok):
                return float(re.sub(r'[ijk]', '', tok))

            qx, qy, qz, qw = map(clean_token, q_tokens)
            # 顺序 (x, y, z, w)
            q = np.array([qx, qy, qz, qw])

            # 平移向量
            tx, ty, tz = map(float, parts[13:16])
            t = np.array([tx, ty, tz])

            edges.append((img1, img2, inliers, inliers / matches, q, t))
            G.add_edge(img1, img2, weight=inliers)
    print("Graph info: nodes edges", G.number_of_nodes(), G.number_of_edges())
    return G

def load_images_as_tensor_from_list(image_list=[], interval=1, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    

    for i in range(0, len(image_list)):
        img_path = image_list[i]
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {image_list[i]}: {e}")


    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    # print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    # print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)

def get_connected_components(subgraph, cluster):
    # 统一边表示（无向图要排序）
    def normalize_edge(u, v):
        return tuple(sorted((u, v)))

    all_edges = {normalize_edge(u, v) for u, v in subgraph.edges()}

    uncovered = set(all_edges)
    samples = []

    all_4sets = list(combinations(cluster, 4))

    while uncovered:
        best_set = None
        best_cover = set()

        for nodes4 in all_4sets:
            edges_in_set = {
                tuple(sorted(e))
                for e in combinations(nodes4, 2)
            }
            cover = edges_in_set & uncovered

            if len(cover) > len(best_cover):
                best_cover = cover
                best_set = nodes4

        samples.append(best_set)
        uncovered -= best_cover
    
    return samples

def recursive_split(subgraph, cluster_nodes, name, model, device, dtype, depth=0):
    global final_clusters

    indent = "  " * depth
    n = len(cluster_nodes)

    if n < 4:
        print(f"{indent}Cluster too small, stop: {cluster_nodes}")
        final_clusters.append(sorted(cluster_nodes))
        return

    samples = get_connected_components(subgraph, cluster_nodes)

    print(f"{indent}Cluster size: {n}, samples: {len(samples)}, edges: {subgraph.number_of_edges()}")
    print(f"{indent}density:", nx.density(subgraph))

    edge_modified = False

    for sample_nodes in samples:

        image_list = [f'{name}/input/{img}' for img in sample_nodes]

        images_tensor = load_images_as_tensor_from_list(image_list)
        images_tensor = images_tensor.to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None])

        pred = res['logits']
        pred = torch.sigmoid(pred).float().cpu().numpy()
        pred = pred.squeeze(0)

        print(indent, sample_nodes, pred)

        for j in range(1, len(sample_nodes)):
            img1, img2 = sample_nodes[0], sample_nodes[j]

            if subgraph.has_edge(img1, img2) and pred[j] < 0.5:
                subgraph.remove_edge(img1, img2)
                edge_modified = True

    # 如果VGGT没有发现错误边 → 终止
    if not edge_modified:
        print(f"{indent}No wrong edges, stop split.")
        final_clusters.append(sorted(cluster_nodes))
        return

    # 重新做Louvain
    sub_communities = community_louvain.best_partition(subgraph, random_state=42)

    sub_clusters = defaultdict(list)
    for node, cid in sub_communities.items():
        sub_clusters[cid].append(node)

    sub_clusters_sorted = sorted(
        sub_clusters.values(),
        key=lambda x: len(x),
        reverse=True
    )

    print(f"{indent}Split into {len(sub_clusters_sorted)} clusters")

    for c in sub_clusters_sorted:
        c = sorted(c)
        new_subgraph = subgraph.subgraph(c).copy()

        recursive_split(new_subgraph, c, name, model, device, dtype, depth+1)



if __name__ == '__main__':
    name = '/home/disk3_SSD/ylx/dataset_vggt_classification/street'
    parse_image_map_file(f'{name}/sparse/image_map.txt')
    # input()
    G = parse_image_pair_file(f'{name}/sparse/image_pair_inliers_relpose_final.txt')
    communitys = community_louvain.best_partition(G, random_state=42)

    clusters = defaultdict(list)
    for node, cid in communitys.items():
        clusters[cid].append(node)
    clusters = dict(clusters)
    clusters_sorted = sorted(
        clusters.values(),
        key=lambda x: len(x),
        reverse=True
    )
    for idx, c in enumerate(clusters_sorted):
        print(f'Cluster {idx}, size: {len(c)}', sorted(c))

    # Prepare model
    print(f"Loading model...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    model = VGGT(enable_point=False, enable_track=False)
    model.load_state_dict(torch.load('ckpt/checkpoint_fourimg_add13_layer012-1_epoch2.pt', map_location=device)['model'])
    model.to(device)
    model.eval()
    print(f"Model loaded")

    final_clusters = []

    for idx, cluster in enumerate(clusters_sorted):

        cluster = sorted(cluster)

        if len(cluster) < 4:
            print(f"Cluster {idx} too small, skipping...")
            final_clusters.append(cluster)
            continue

        subgraph = G.subgraph(cluster).copy()

        print(f"\n==== Start Cluster {idx} ====")

        recursive_split(subgraph, cluster, name, model, device, dtype)

    print("\nFinal clusters:")
    for c in final_clusters:
        print(len(c), c)

    input()



    # final_clusters = [['0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0009.jpg', '0010.jpg', '0011.jpg'], 
    #                   ['0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg', '0016.jpg', '0017.jpg', '0018.jpg'], 
    #                   ['0000.jpg', '0001.jpg', '0002.jpg', '0003.jpg', '0004.jpg']]
    
    # print(f"Final clusters: ", final_clusters)

    with open(f'{name}/image_clusters_louvain.txt', 'w') as f:
        for idx, c in enumerate(final_clusters):
            f.write(f'# Cluster {idx}, size: {len(c)}\n')
            c = sorted(c)
            for img in c:
                f.write(f'{img} ')
            f.write('\n')

    all_node4 = []

    for group_1, group_2 in tqdm(combinations(final_clusters, 2)):
        for img1 in group_1:
            for img2 in group_2:
                if G.has_edge(img1, img2):
                    # print(f"Inter-cluster edge: ({img1}, {img2}), weight: {G[img1][img2]['weight']}")

                    img1_neighbors_in_group1 = set(G.neighbors(img1)) & set(group_1)
                    img1_neighbors_in_group1_weight = {neighbor: G[img1][neighbor]['weight'] for neighbor in img1_neighbors_in_group1}
                    node1 = sorted(img1_neighbors_in_group1_weight.items(), key=lambda x: x[1], reverse=True)[0][0]
                    # node2 = sorted(img1_neighbors_in_group1_weight.items(), key=lambda x: x[1], reverse=True)[1][0]
                    
                    img2_neighbors_in_group2 = set(G.neighbors(img2)) & set(group_2)
                    img2_neighbors_in_group2_weight = {neighbor: G[img2][neighbor]['weight'] for neighbor in img2_neighbors_in_group2}
                    # node2 = sorted(img2_neighbors_in_group2_weight.items(), key=lambda x: x[1], reverse=True)[0][0]

                    if len(img1_neighbors_in_group1) > 1:
                        node2 = sorted(img1_neighbors_in_group1_weight.items(), key=lambda x: x[1], reverse=True)[1][0]
                    else:
                        node2 = sorted(img2_neighbors_in_group2_weight.items(), key=lambda x: x[1], reverse=True)[0][0]

                    # node4 = [node1, img1, img2, node2]
                    node4 = [node1, node2, img1, img2]
                    all_node4.append(node4)

    delete_edges = []
    reserve_edges = []

    for node4 in tqdm(all_node4):
        image_list = [f'{name}/input/{img}' for img in node4]
        images_tensor = load_images_as_tensor_from_list(image_list)
        images_tensor = images_tensor.to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None]) # Add batch dimension
        # print(res['logits'].shape)
        pred = res['logits']   # [B, N]
        pred = torch.sigmoid(pred).float().cpu().numpy() # [B, N] (0, 1)
        pred = pred.squeeze(0)
        print(node4, pred)
        if pred[3] < 0.9:
            delete_edges.append((node4[2], node4[3]))
        else:
            reserve_edges.append((node4[2], node4[3]))


    with open(f'{name}/delete_edges_name.txt', 'w') as f:
        print("writing delete_edges_name.txt")
        for edge in delete_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    with open(f'{name}/delete_edges.txt', 'w') as f:
        print("writing delete_edges.txt")
        for img1, img2 in delete_edges:
            f.write(f"{img_name2id[img1]} {img_name2id[img2]}\n")

    print(reserve_edges)
    print(len(reserve_edges))