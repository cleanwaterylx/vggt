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
from utils.process_database import create_image_pair_list, remove_doppelgangers
import time
import argparse
import subprocess
from scipy.special import softmax

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


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    parser.add_argument('--colmap_exe_command', default='colmap', type=str, help='COLMAP executable command.')
    parser.add_argument('--matching_type', default='vocab_tree_matcher', type=str, help="Feature matching type: ['vocab_tree_matcher', 'exhaustive_matcher']")
    # parser.add_argument('--skip_feature_matching', action='store_true', help="Skip COLMAP feature matching stage.")
    # parser.add_argument('--database_path', type=str, default=None, help="Path to the COLMAP database.")
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output results.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Doppelgangers threshold.')
    parser.add_argument('--pretrained', type=str, default='checkpoints/dopp-crop-focalloss_lr1e-3_warmup20/checkpoint-best.pth', help="Path to the pretrained model checkpoint.")
    parser.add_argument('--skip_mapper', action='store_true', help="Skip COLMAP mapping stage.")
    
    args = parser.parse_args()
    return args


def colmap_runner(args):

    commands = [
        [
            args.colmap_exe_command, "feature_extractor",
            "--image_path", args.input_image_path,
            "--database_path", args.database_path,
            '--ImageReader.camera_model', 'PINHOLE'
        ],
        [
            args.colmap_exe_command, 'exhaustive_matcher',
            '--database_path', args.database_path
        ]   
    ]

    # if args.matching_type == 'vocab_tree_matcher':
    #     commands[1].extend(["--VocabTreeMatching.vocab_tree_path", vocab_tree_path])

    for command in commands:
        subprocess.run(command, check=True)
        

def doppelgangers_classifier(args):
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

    pairs = np.load(f"{args.output_path}/pairs_list.npy")
    prob_list = []

    for pair in tqdm(pairs, desc="Disambiguating pairs"):
        img1, img2, *_ = pair
        img_paths = [os.path.join(args.input_image_path, img) for img in [img1, img2]]
        images_tensor = load_images_as_tensor_from_list(img_paths)
        images_tensor = images_tensor.to(device)


        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None]) # Add batch dimension
        # print(res['logits'].shape)
        pred = res['logits']   # [B, N]
        pred = torch.sigmoid(pred).float().cpu().numpy() # [B, N] (0, 1)
        pred = pred.squeeze(0)
        score = pred[-1]
        prob_list.append(score)
        print(f"Pair: {img1} - {img2}, Doppelganger Score: {score:.4f}")

    np.save(f"{args.output_path}/pair_probability_list_dust3r.npy", {'prob': np.array(prob_list).reshape(-1, 1)})


def main():
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    args.database_path = os.path.join(args.output_path, 'database.db')
    colmap_runner(args)

    pair_path = create_image_pair_list(args.database_path, args.output_path)
    doppelgangers_classifier(args)
    update_database_path = remove_doppelgangers(args.database_path, f"{args.output_path}/pair_probability_list_dust3r.npy", pair_path, args.threshold)
    
    if not args.skip_mapper:
        doppelgangers_result_path = os.path.join(args.output_path, 'sparse_doppelgangers_%.3f_pheonix'%args.threshold)    
        os.makedirs(doppelgangers_result_path, exist_ok=True)       
        subprocess.run([args.colmap_exe_command, 'mapper',
                '--database_path', update_database_path,
                '--image_path', args.input_image_path,
                '--output_path', doppelgangers_result_path,
                ])
    


if __name__ == '__main__':
    start = time.time()  # 开始时间
    main()
    end = time.time()    # 结束时间
    elapsed_minutes = (end - start) / 60
    print(f"Execution time: {elapsed_minutes:.2f} minutes")
