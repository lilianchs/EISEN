from __future__ import print_function, division
import pdb
import sys
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
import h5py
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import torch
import torch.nn as nn
from torch.nn import functional as F
from core.datasets import fetch_dataloader
from core.eisen import EISEN
from core.optimizer import fetch_optimizer
from core.raft import EvalRAFT
from segmentation.utils import plot_segments_pointseg, get_baseline_metrics_pointseg
import core.utils.sync_batchnorm as sync_batchnorm
num_gpus = torch.cuda.device_count()

def evaluate(args):
    val_loader = fetch_dataloader(args, training=False, drop_last=False)
    model = nn.DataParallel(EISEN())
    model = sync_batchnorm.convert_model(model)
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold) if args.compute_flow else None

    assert args.ckpt is not None
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)
    logging.info(f'Restore checkpoint from {args.ckpt}')

    model.cuda()
    logging.info(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    assert args.batch_size / num_gpus == 1, \
            f"Effective Batch size for evaluation should be 1, but got batch size {args.batch_size} and {num_gpus} gpus"

    evaluate_helper(args, val_loader, model)

    return


def evaluate_helper(args, dataloader, model):
    model.eval()

    with h5py.File(args.h5_save_file, "w") as h5f:
        for step, data_dict_og in enumerate(dataloader):
            data_dict = {k: v.cuda() for k, v in data_dict_og.items() if not k == 'file_name'}
            data_dict['file_name'] = data_dict_og['file_name']

            bs = data_dict['img1'].shape[0]
            if bs < args.batch_size:  # add padding if bs is smaller than batch size (required since drop_last is set to False)
                pad_size = args.batch_size - bs
                for key in data_dict.keys():
                    padding = torch.cat([torch.zeros_like(data_dict[key][0:1])] * pad_size, dim=0)
                    data_dict[key] = torch.cat([data_dict[key], padding], dim=0)

            segment_target = data_dict['segment_target']

            _, loss, metric, segment, segment_pred = model(data_dict, segment_target.detach(),
                                             get_segments=True)

            for batch_idx in range(bs):  # Only process original batch size, not padded
                file_name = data_dict['file_name'][batch_idx]
                print(f"Processing {file_name}...")

                # Get data for this batch item
                rgb = data_dict['rgb'][batch_idx]  # Shape: (C, 512, 512)
                segments = segment_pred[batch_idx]  # Shape: (N, 512, 512)
                centroids = data_dict['centroids'][batch_idx]  # Shape: (N, 2)

                # Resize from 512x512 to 256x256
                segments_resized = F.interpolate(segments.unsqueeze(1), size=(256, 256), mode='area').squeeze(1)
                centroids_resized = centroids / 2.0  # Divide by 2 for resize

                # Convert to numpy and transpose RGB (C,H,W) -> (H,W,C)
                rgb_np = rgb.cpu().numpy()  # (256, 256, C)
                # rgb_np = rgb_np[:, :, [2, 1, 0]]
                segments_np = segments_resized.cpu().numpy()  # (N, 256, 256)
                centroids_np = centroids_resized.cpu().numpy()  # (N, 2)
                # Create group for this image
                img_grp = h5f.create_group(f"{file_name}")
                img_grp.create_dataset("rgb", data=rgb_np, compression="gzip")

                gt_segment = data_dict['gt_segment'].cpu().numpy()[0]
                img_grp.create_dataset("segment", data=gt_segment, compression="gzip")

                img_grp.create_dataset("centroid", data=centroids_np, compression="gzip")

                # Save each predicted segment as seg0/pt0, seg1/pt0, etc.
                for si in range(segments_np.shape[0]):
                    seg_grp = img_grp.create_group(f"seg{si}")
                    pt_grp = seg_grp.create_group(f"pt0")  # Always pt0 as specified
                    pt_grp.create_dataset("segment", data=segments_np[si], compression="gzip")
                    pt_grp.create_dataset("centroid", data=centroids_np[si], compression="gzip")

                print(f'{file_name} saved in {args.h5_save_file}')

    return

def get_model_args(args):
    params = dict()
    params['affinity_res'] = args.affinity_size
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--name', default='eisen', help="name your experiment")
    parser.add_argument('--dataset', default="playroom", help="determines which dataset to use for training")

    # dataloader
    parser.add_argument('--num_workers', type=int, default=16)

    # training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--affinity_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--flow_threshold', type=float, default=0.5, help='binary threshold on raft dlows')
    parser.add_argument('--ckpt', type=str, help='path to restored checkpoint')

    # evaluation
    parser.add_argument('--h5_save_file', default='/ccn2/u/lilianch/external_repos/EISEN/vis/segments.h5')
    parser.add_argument('--vis_dir', default='/ccn2/u/lilianch/external_repos/EISEN/vis/test/')
    parser.add_argument('--val_freq', type=int, default=5000, help='validation and checkpoint frequency')
    parser.add_argument('--visualize', action='store_true', help='visualize segments')
    parser.add_argument('--vis_only', action='store_true', help='visualize segments only')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--metrics_only', action='store_true')

    # logging
    parser.add_argument('--print_freq', type=int, default=100, help='frequency for printing loss')
    parser.add_argument('--wandb', action='store_true', help='enable wandb login')

    # flow
    parser.add_argument('--compute_flow', action='store_true', help='compute flow during training (slower option)')

    args = parser.parse_args()
    torch.manual_seed(1)
    np.random.seed(1)

    if args.vis_only:
        plot_segments_pointseg(args.h5_save_file, args.vis_dir)

    if args.metrics_only:
        get_baseline_metrics_pointseg(args.h5_save_file, args.vis_dir)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if not os.path.isdir(f'checkpoints/{args.name}'):
        os.mkdir(f'checkpoints/{args.name}')

    evaluate(args)

    if args.visualize:
        plot_segments_pointseg(args.h5_save_file, args.vis_dir)

    if args.metrics:
        get_baseline_metrics_pointseg(args.h5_save_file, args.vis_dir)


