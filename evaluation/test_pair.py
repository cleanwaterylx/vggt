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
import os
from itertools import combinations
from vggt.models.vggt_classification import VGGT


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


if __name__  == '__main__':

    # Prepare model
    print(f"Loading model...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    model = VGGT(enable_point=False, enable_track=False)
    model.load_state_dict(torch.load('ckpt/checkpoint_fourimg_add13_layer05111723_epoch12_focalloss_dopp_adam.pt', map_location=device)['model'])
    model.to(device)
    model.eval()
    print(f"Model loaded")


    name = '/home/disk3_SSD/ylx/dataset_vggt_classification'
    all_node4 = [['26/input/00024.jpg', '26/input/00025.jpg', '26/input/00026.jpg', '26/input/00083.jpg'],
                 ['26/input/00024.jpg', '26/input/00025.jpg', '26/input/00026.jpg', '26/input/00043.jpg'],
                 ['26/input/00024.jpg', '26/input/00025.jpg', '26/input/00026.jpg', '26/input/00039.jpg'],
                 ['26/input/00086.jpg', '26/input/00089.jpg', '26/input/00090.jpg', '26/input/00123.jpg'],
                 ['street/input/0000.jpg', 'street/input/0001.jpg', 'street/input/0002.jpg', 'street/input/0016.jpg'],
                 ['street/input/0000.jpg', 'street/input/0001.jpg', 'street/input/0002.jpg', 'street/input/0017.jpg'],
                 ['street/input/0003.jpg', 'street/input/0004.jpg', 'street/input/0005.jpg', 'street/input/0012.jpg'],
                 ['street/input/0003.jpg', 'street/input/0004.jpg', 'street/input/0005.jpg', 'street/input/0014.jpg'],
                 ['street/input/0003.jpg', 'street/input/0004.jpg', 'street/input/0005.jpg', 'street/input/0016.jpg'],
                 ['street/input/0004.jpg', 'street/input/0005.jpg', 'street/input/0006.jpg', 'street/input/0016.jpg'],
                 ['books/input/0001.jpg', 'books/input/0002.jpg', 'books/input/0003.jpg', 'books/input/0014.jpg'],
                 ['books/input/0001.jpg', 'books/input/0002.jpg', 'books/input/0003.jpg', 'books/input/0015.jpg'],
                 ['books/input/0002.jpg', 'books/input/0003.jpg', 'books/input/0004.jpg', 'books/input/0015.jpg'],
                 ['indoor/input/0033.jpg', 'indoor/input/0047.jpg', 'indoor/input/0077.jpg', 'indoor/input/0015.jpg'],
                 ['indoor/input/0033.jpg', 'indoor/input/0047.jpg', 'indoor/input/0077.jpg', 'indoor/input/0067.jpg'],
                 ['indoor/input/0033.jpg', 'indoor/input/0047.jpg', 'indoor/input/0077.jpg', 'indoor/input/0080.jpg'],
                 ['indoor/input/0033.jpg', 'indoor/input/0047.jpg', 'indoor/input/0077.jpg', 'indoor/input/0019.jpg'],
                 ]

    for node4 in all_node4:
        # node4 = node4[2:]
        image_list = [os.path.join(name, img) for img in node4]
        images_tensor = load_images_as_tensor_from_list(image_list)
        images_tensor = images_tensor.to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None]) # Add batch dimension
        # print(res['logits'].shape)
        pred = res['logits']   # [B, N]
        pred = torch.sigmoid(pred).float().cpu().numpy() # [B, N] (0, 1)
        pred = pred.squeeze(0)
        print(node4)
        print(pred)