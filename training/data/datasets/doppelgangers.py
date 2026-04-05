import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import copy

from data.dataset_util import *
from data.base_dataset import BaseDataset


class DoppelgangersDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DOPPELGANGERS_DIR: str = None,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.data_root = DOPPELGANGERS_DIR
        self._rng = np.random.default_rng(42)

        self.meta_to_image = {
            'pair_data/train_pairs_megadepth.npy': 'train_megadepth',
            'pair_data/train_pairs_flip.npy': 'train_set_flip',
            'pair_data/train_pairs_noflip.npy': 'train_set_noflip',
            'pair_data/test_pairs.npy': 'test_set',
        }

        train_meta = [
            'pair_data/train_pairs_megadepth.npy',
            'pair_data/train_pairs_flip.npy',
            'pair_data/train_pairs_noflip.npy'
        ]

        test_meta = ['pair_data/test_pairs.npy']
        if split == "train":
            self.dopp_pair = self._load_pairs(train_meta)
        else:
            self.dopp_pair = self._load_pairs(test_meta)
        
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: VisymScenes Data dataset length: {len(self)}")

    def _load_pairs(self, metadata_list):
        all_pairs = []

        for metadata in metadata_list:
            meta_path = metadata

            pairs = np.load(meta_path, allow_pickle=True)
            img_dir = self.meta_to_image[metadata]

            for pair in pairs:
                if 'gif' in pair[0] or 'gif' in pair[1]:
                    continue
                
                im1 = os.path.join(
                    'doppelgangers',
                    'images',
                    img_dir,
                    pair[0]
                )

                im2 = os.path.join(
                    'doppelgangers',
                    'images',
                    img_dir,
                    pair[1]
                )

                all_pairs.append([im1, im2, int(pair[2])])
        self._rng.shuffle(all_pairs)
        return all_pairs

    def __len__(self):
        return len(self.dopp_pair)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ):

        im1_rel, im2_rel, label = self.dopp_pair[seq_index]

        target_image_shape = self.get_target_shape(aspect_ratio)

        im1_path = os.path.join(self.data_root, im1_rel)
        im2_path = os.path.join(self.data_root, im2_rel)

        images = []
        original_sizes = []
        labels = []

        for im_path in [im1_path, im2_path]:
            image = read_image_cv2(im_path)
            original_size = np.array(image.shape[:2])
            depth_map = np.ones((image.shape[0], image.shape[1]),dtype=np.float32)

            # dummy intrinsics
            intrinsic = np.array([
                [1000, 0, image.shape[1] // 2], 
                [0, 1000, image.shape[0] // 2], 
                [0, 0, 1]], dtype=np.float32)

            # print("im_path =", im_path)
            # print("type(image) =", type(image))
            # print("image.size =", image.size if hasattr(image, "size") else None)
            # print("original_size =", original_size, original_size.dtype, original_size.shape)
            # print("target_image_shape =", target_image_shape)

            image, depth_map,_, _ = self.process_one_image_no_extri(
                image,
                depth_map,
                intrinsic,
                original_size,
                target_image_shape,
            )

            images.append(image)
            original_sizes.append(original_size)

        labels = [1, label]

        batch = {
            "seq_name": f"doppel_{seq_index}",
            "frame_num": 2,
            "images": images,
            "original_sizes": original_sizes,
            "labels": labels,
        }

        return batch