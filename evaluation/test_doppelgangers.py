import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import math
from tqdm import tqdm

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


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    data_root='/home/disk8/dopp_data/'
    dopp_pair = np.load('training/pair_data/test_train_pairs.npy', allow_pickle=True)
    # shuffle the pairs
    np.random.seed(42) 
    np.random.shuffle(dopp_pair)
    print(f'Number of test DoppPairs: {len(dopp_pair)}')

    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    model = VGGT(enable_point=False, enable_track=False)
    # weight = torch.load('ckpt/checkpoint.pt')['model'].keys()
    # vggt_weight = {k: v for k, v in torch.load('ckpt/checkpoint.pt')['model'].items() if k in model.state_dict()}
    model.load_state_dict(torch.load('ckpt/checkpoint_fourimg_add13_layer05111723_epoch3_focalloss_dopp_adam_dopp_dataset.pt', map_location=device)['model'])
    model.to(device)
    model.eval()
    print(f"Model loaded")
    
    gts = []
    preds = []
    logits = []
    all_true = 0
    not_all_true = 0
    for pair in tqdm(dopp_pair):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label = pair
        image_0_name = os.path.join(data_root, image_0_relative_path)
        image_1_name = os.path.join(data_root, image_1_relative_path)
        selected_imgs = [image_0_name, image_1_name]
        images_tensor = load_images_as_tensor_from_list(selected_imgs)
        images_tensor = images_tensor.to(device)
        labels = [1, pos_neg_pair_label]
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None]) # Add batch dimension
        # print(res['logits'].shape)
        pred = res['logits']   # [B, N]
        pred = torch.sigmoid(pred).float().cpu().numpy() # [B, N] (0, 1)
        pred = pred.squeeze(0)  # [N]
        # print(image_0_name, image_1_name)
        # print(selected_imgs)
        # print(pred)
        # print(labels)
        # input()

        # if pred[-1] >= 0.5 and labels[-1] == 0:
        #     print(f"False Positive: {image_0_name}, {image_1_name}, Pred: {pred[-1]:.4f}, Label: {labels[-1]}")
        # print(f"Pred: {pred[-1]:.4f}, Label: {labels[-1]}")

        images_tensor = images_tensor.flip(0)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None]) # Add batch dimension
        # print(res['logits'].shape)
        pred1 = res['logits']   # [B, N]
        pred1 = torch.sigmoid(pred1).float().cpu().numpy() # [B, N] (0, 1)
        pred1 = pred1.squeeze(0)

        if pred1[-1] < pred[-1]:
            pred = pred1       
        
        logits.append(pred)
        # gts.append(pos_neg_pair_label)
        # preds.append(pred[-1])
        gts.append(labels)
        preds.append(pred)
        
            
    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    # tp = np.sum((np.array(preds) == 1) & (np.array(gts) == 1))
    # fp = np.sum((np.array(preds) == 1) & (np.array(gts) == 0))
    # tn = np.sum((np.array(preds) == 0) & (np.array(gts) == 0))
    # fn = np.sum((np.array(preds) == 0) & (np.array(gts) == 1))
    
    # print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    # np.save('test_logits_visym.npy', logits)
    # np.save('test_gts_visym.npy', gts)
    np.save('result/vggt_fourimg_add13_layer05111723_epoch3_focalloss_dopp_adam_dopp_dataset_11_test_dopp_test_train_min.npy', {'logits': logits, 'gts': gts, 'preds': preds})
    
        
        
        
        
        
        
        
        
    
    
    