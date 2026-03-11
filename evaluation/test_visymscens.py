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
    data_root='/home/disk8/dopp_data/visymscenes'
    dopp_pair = np.load('training/pair_data/test_pairs_visym_with_intrinsics.npy', allow_pickle=True)
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
    model.load_state_dict(torch.load('ckpt/checkpoint_fourimg_add13_layer012-1_epoch2.pt', map_location=device)['model'])
    model.to(device)
    model.eval()
    print(f"Model loaded")
    
    gts = []
    preds = []
    inverse = 0
    logits = []
    all_true = 0
    not_all_true = 0
    for pair in tqdm(dopp_pair):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics = pair
        pos_neg_pair_label = int(pos_neg_pair_label)
        scene1 = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
        scene2 = os.path.join(*image_1_relative_path.split('/')[:3])
        base_path1 = os.path.join(data_root, scene1)
        base_path2 = os.path.join(data_root, scene2)
        imgs_1 = sorted([file for file in os.listdir(base_path1) if file.endswith('.jpg')])
        imgs_2 = sorted([file for file in os.listdir(base_path2) if file.endswith('.jpg')])
        # print(scene)
        # print(len(imgs))
        # print(imgs[0], imgs[1])

        image_0_name = image_0_relative_path.split('/')[-1]
        image_1_name = image_1_relative_path.split('/')[-1]
        idx_1 = imgs_1.index(image_0_name)
        idx_2 = imgs_2.index(image_1_name)
        # print(image_0_name)
        # print('Image 0 index in the sequence:', idx)

        #  set image_0 as anchor ,idx random +- 5
        # todo random step or sample 
        step = 3
        idxs_1 = list(range(max(0, idx_1 - step), min(len(imgs_1), idx_1 + step + 1)))
        idxs_1.remove(idx_1)
        idxs_1 = [idx_1] + idxs_1
        idxs_2 = list(range(max(0, idx_2 - step), min(len(imgs_2), idx_2 + step + 1)))
        idxs_2.remove(idx_2)
        idxs_2 = idxs_2 + [idx_2]
        # print(img_per_seq)
        # print(len(idxs_1), len(idxs_2))
        
        step = 2
        if step == 0:
            idxs_1 = [idx_1]
            idxs_2 = [idx_2]
        elif step == 1:
            idxs_1 = [idx_1]
            idxs_2 = list(range(max(0, idx_2), min(len(imgs_2), idx_2 + 2)))
        else:
            idxs_1 = list(range(max(0, idx_1-1), min(len(imgs_1), idx_1 + 1)))
            idxs_2 = list(range(max(0, idx_2-1), min(len(imgs_2), idx_2 + 1)))
        
        selected_imgs = [os.path.join(base_path1, imgs_1[j]) for j in idxs_1]
        labels = [1] * len(selected_imgs)
        selected_imgs += [os.path.join(base_path2, imgs_2[j]) for j in idxs_2]
        labels += [1 if pos_neg_pair_label else 0] * len(idxs_2)
        
        if len(selected_imgs) < 4:
            # print(f"Warning: Only {len(selected_imgs)} images selected for pair ({image_0_relative_path}, {image_1_relative_path}). Expected 4.")
            continue
        
        # combined = list(zip(selected_imgs, labels))

        # np.random.default_rng(42).shuffle(combined)
        
        # selected_imgs, labels = map(list, zip(*combined))
        
        images_tensor = load_images_as_tensor_from_list(selected_imgs)
        images_tensor = images_tensor.to(device)
        
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
        
        if labels[0] == 1 and labels[1] == 1 and labels[2] == 0 and labels[3] == 0:
            if pred[0] >= 0.5 and pred[1] >= 0.5 and pred[2] < 0.5 and pred[3] < 0.5:
                all_true += 1
            else:
                not_all_true += 1
                print(labels, pred)
                
        
        logits.append(pred)
        # gts.append(pos_neg_pair_label)
        # preds.append(pred[-1])
        gts.append(labels)
        preds.append(pred)
        # if pred[0] >= 0.5 and pred[-1] >= 0.5:  # Both anchor and positive samples are predicted as positive
        #     preds.append(1)
        # else:
        #     preds.append(0)
        
    print(f"Total pairs: {len(gts)}, all true: {all_true}, Not all true: {not_all_true}, Accuracy: {all_true / (all_true + not_all_true):.4f}")
            
    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    # tp = np.sum((np.array(preds) == 1) & (np.array(gts) == 1))
    # fp = np.sum((np.array(preds) == 1) & (np.array(gts) == 0))
    # tn = np.sum((np.array(preds) == 0) & (np.array(gts) == 0))
    # fn = np.sum((np.array(preds) == 0) & (np.array(gts) == 1))
    
    # print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    # np.save('test_logits_visym.npy', logits)
    # np.save('test_gts_visym.npy', gts)
    np.save('vggt_fourimg_add13_layer012-1_epoch2_22.npy', {'logits': logits, 'gts': gts, 'preds': preds})
    
        
        
        
        
        
        
        
        
    
    
    