import os
import json
import glob
import pdb
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import shutil
import logging
import logging
import h5py as h5
import torch.nn.functional as F

class PlayroomDataset(Dataset):
    def __init__(self, training, args, frame_idx=0, dataset_dir = './datasets/Playroom'):

        self.training = training
        self.frame_idx = frame_idx
        self.args = args

        # meta.json is only required for TDW datasets
        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_*', '*[0-8]'))
        else:
            self.file_list = sorted(glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-3]', '*9')))

        if args.precompute_flow: # precompute flows for training and validation dataset
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_*', '*')) # glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]')) #+ \


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        frame_idx = self.frame_idx if os.path.exists(self.get_image_path(file_name, self.frame_idx)) else 0
        img1 = read_image(self.get_image_path(file_name, frame_idx))


        flag = os.path.exists(self.get_image_path(file_name, self.frame_idx+1))
        img2 = read_image(self.get_image_path(file_name, frame_idx+1)) if flag else img1
        segment_colors = read_image(self.get_image_path(file_name.replace('/images/', '/segments/'), frame_idx))
        gt_segment = self.process_segmentation_color(segment_colors, file_name)

        ret = {'img1': img1, 'img2': img2, 'gt_segment': gt_segment}

        if not self.args.compute_flow and not self.args.precompute_flow:
            flow_path = os.path.join(file_name.replace('/images/', '/flows/'), f'frame{frame_idx}.npy')
            flow = np.load(flow_path)
            magnitude = torch.tensor((flow ** 2).sum(0, keepdims=True) ** 0.5)
            segment_target = (magnitude > self.args.flow_threshold)
            ret['segment_target'] = segment_target
        elif self.args.precompute_flow:
            ret['file_name'] = self.get_image_path(file_name, frame_idx)

        return ret

    @staticmethod
    def get_image_path(file_name, frame_idx):
        return os.path.join(file_name, f'frame{frame_idx}' + '.png')

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        zone_id = int(self.meta[meta_key]['zone'])
        raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        return segment_map

class SpelkeBench(Dataset):
    def __init__(self, args, frame_idx=0, dataset_path = './datasets/SpelkeBench/550_openx_entity_dataset.h5'):
        self.frame_idx = frame_idx
        self.args = args
        self.dataset = h5.File(dataset_path, 'r')
        self.file_list = list(self.dataset.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        rgb = self.dataset[file_name]['rgb'][:]
        img1 = torch.tensor(self.dataset[file_name]['rgb'][:]).permute(2, 0, 1).float()  # [C, H, W]
        img1 = F.interpolate(img1.unsqueeze(0),size=(512, 512),mode='area').squeeze(0)
        img2 = img1.clone()

        centroids = torch.tensor(self.dataset[file_name]['centroid'][:])  # [N, 2] (x, y)
        centroids = centroids * 2 # scale up for 512x512

        gt_segment = self.dataset[file_name]['segment'][:]
        ret = {'img1': img1, 'rgb': rgb, 'gt_segment': gt_segment, 'centroids': centroids, 'file_name': file_name}

        # use GT segment; unfair but see how it does...
        ret['segment_target'] = gt_segment

        return ret


def fetch_dataloader(args, training=True, drop_last=True):
    """ Create the data loader for the corresponding trainign set """
    if args.dataset == 'playroom':
        dataset = PlayroomDataset(training=training, args=args)
    elif args.dataset == 'spelkebench':
        dataset = SpelkeBench(args=args)
    else:
        raise ValueError(f'Expect dataset in [playroom], but got {args.dataset} instead')

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=training,
                            num_workers=args.num_workers,
                            drop_last=drop_last)

    logging.info(f"Load dataset [{args.dataset}-{'train' if training else 'val'}] with {len(dataset)} image pairs")
    return dataloader


if __name__ == "__main__":
    pass
