# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
from urllib.request import urlretrieve
try:
  import open3d as o3d
except ImportError:
  raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import MinkowskiEngine as ME

from models.res16unet import Res16UNet34D
from lib.datasets.scannet import VALID_CLASS_IDS_200, SCANNET_COLOR_MAP_200

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='MinkUNet34D-train-conv1-5.pth')
parser.add_argument('--file_name', type=str, default='1.ply')
parser.add_argument('--bn_momentum', type=float, default=0.05)
parser.add_argument('--voxel_size', type=float, default=0.02)
parser.add_argument('--conv1_kernel_size', type=int, default=5)

def load_file(file_name, voxel_size):
  pcd = o3d.io.read_point_cloud(file_name)
  coords = np.array(pcd.points)
  feats = np.array(pcd.colors)

  quantized_coords = np.floor(coords / voxel_size)
  inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)

  return quantized_coords[inds], feats[inds], pcd


def generate_input_sparse_tensor(file_name, voxel_size=0.02):
  # Create a batch, this process is done in a data loader during training in parallel.
  batch = [load_file(file_name, voxel_size)]
  coordinates_, featrues_, pcds = list(zip(*batch))
  coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

  # Normalize features and create a sparse tensor
  return coordinates, (features - 0.5).float()


if __name__ == '__main__':
  config = parser.parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Define a model and load the weights
  model = Res16UNet34D(3, 200, config).to(device)
  model_dict = torch.load(config.weights)
  model.load_state_dict(model_dict['state_dict'])
  model.eval()

  # Measure time
  with torch.no_grad():
    coordinates, features = generate_input_sparse_tensor(
        config.file_name, voxel_size=config.voxel_size)

    # Feed-forward pass and get the prediction
    sinput = ME.SparseTensor(features, coords=coordinates).to(device)
    soutput = model(sinput)

  # Feed-forward pass and get the prediction
  _, pred = soutput.F.max(1)
  pred = pred.cpu().numpy()

  # Map color
  colors = np.array([SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l]] for l in pred])

  # Create a point cloud file
  pred_pcd = o3d.geometry.PointCloud()
  coordinates = soutput.C.numpy()[:, 1:]  # first column is the batch index
  pred_pcd.points = o3d.utility.Vector3dVector(coordinates * config.voxel_size)
  pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)

  # Move the original point cloud
  pcd = o3d.io.read_point_cloud(config.file_name)
  pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) + np.array([0, 5, 0]))

  # Visualize the input point cloud and the prediction
  o3d.visualization.draw_geometries([pcd, pred_pcd])
