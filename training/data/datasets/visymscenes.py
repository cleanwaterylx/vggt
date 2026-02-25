# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


class VisymScenesDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        VISYMSCENES_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the VisymScenesDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VISYMSCENES_DIR (str): Directory path to VisymScenes data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If VISYMSCENES_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.train_dopp_pair_path = 'pair_data/train_pairs_visym_with_intrinsics.npy' 
        self.test_dopp_pair_path = 'pair_data/test_pairs_visym_with_intrinsics.npy'
        self.data_root = VISYMSCENES_DIR
        self._rng = np.random.default_rng(42)

        if VISYMSCENES_DIR is None :
            raise ValueError("Both VISYMSCENES_DIR must be specified.")


        if split == "train":
            self.dopp_pair = np.load(self.train_dopp_pair_path, allow_pickle=True)
        elif split == "test":
            self.dopp_pair = np.load(self.test_dopp_pair_path, allow_pickle=True)
        else:
            raise ValueError(f"Invalid split: {split}")


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"VISYMSCENES_DIR is {VISYMSCENES_DIR}")

        self.VISYMSCENES_DIR = VISYMSCENES_DIR

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: VisymScenes Data dataset length: {len(self)}")
        
    def __len__(self):
        return len(self.dopp_pair)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
            
        
        pair = self.dopp_pair[seq_index]
        
        if seq_name is None:
            seq_name = pair[0] + "_" + pair[1]  # create a sequence name based on the two image paths

        target_image_shape = self.get_target_shape(aspect_ratio)
        
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics = pair
        pos_neg_pair_label = int(pos_neg_pair_label)
        scene1 = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
        scene2 = os.path.join(*image_1_relative_path.split('/')[:3])
        base_path1 = os.path.join(self.data_root, scene1)
        base_path2 = os.path.join(self.data_root, scene2)
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
        step = (img_per_seq - 4) // 4
        idxs_1 = list(range(max(0, idx_1 - step), min(len(imgs_1), idx_1 + step + 1)))
        idxs_2 = list(range(max(0, idx_2 - step), min(len(imgs_2), idx_2 + step + 1)))
        # print(img_per_seq)
        # print(len(idxs_1), len(idxs_2))
        

        images = []
        original_sizes = []
        labels = []
        
        for i, idx_1 in enumerate(idxs_1):
            img_path = os.path.join(base_path1, imgs_1[idx_1])
            
            image = read_image_cv2(img_path)
            intrinsic = intrinsics[0] 
            original_size = np.array(image.shape[:2])
            depth_map = np.ones((image.shape[0], image.shape[1]),dtype=np.float32)
            image, depth_map,_, _ = self.process_one_image_no_extri(
                image,
                depth_map,
                intrinsic,
                original_size,
                target_image_shape,
            )
            
            images.append(image)
            original_sizes.append(original_size)
            labels.append(1)
        
        for i, idx_2 in enumerate(idxs_2):
            img_path = os.path.join(base_path2, imgs_2[idx_2])
            
            image = read_image_cv2(img_path)
            intrinsic = intrinsics[1] 
            original_size = np.array(image.shape[:2])
            depth_map = np.ones((image.shape[0], image.shape[1]),dtype=np.float32)
            image, depth_map,_, _ = self.process_one_image_no_extri(
                image,
                depth_map,
                intrinsic,
                original_size,
                target_image_shape,
            )
            images.append(image)
            original_sizes.append(original_size)
            labels.append(1 if pos_neg_pair_label else 0)
        
        while len(images) < img_per_seq:
            images.append(copy.deepcopy(images[-1]))
            original_sizes.append(original_sizes[-1])
            labels.append(labels[-1])

        set_name = "visymscenes"
        
        combined = list(zip(images, original_sizes, labels))

        self._rng.shuffle(combined)
        
        images, original_sizes, labels = map(list, zip(*combined))
        
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "original_sizes": original_sizes,
            "labels": labels,
        }
        return batch
