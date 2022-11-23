# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import warnings

import open3d as o3d
import numpy as np
import torch
import os
import os.path as osp
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device

from lib.datasets.scannet import VALID_CLASS_IDS_200
import MinkowskiEngine as ME

import wandb

# Needs to be updated when doing evaluation
TESTING_FILES_TXT = '/local/home/yunzou/scannet_data/scannet_200_processed/train/scannetv2_test.txt'
TESTING_SCANS_ROOT = '/local/home/yunzou/scannet_data/scannet_200_processed/test'
PRED_ROOT = 'testing_set_prediction'

# Load point cloud file, from https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/indoor.py
def load_file(file_name):
  pcd = o3d.io.read_point_cloud(file_name)
  coords = np.array(pcd.points)
  colors = np.array(pcd.colors)
  return coords, colors, pcd

# Normalize color function, from https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/indoor.py
def normalize_color(color: torch.Tensor, is_color_in_range_0_255: bool = False) -> torch.Tensor:
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.
    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()


def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               has_gt=False,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None,
               data_type='validation'):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    acc = hist.diagonal() / hist.sum(1) * 100
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
        "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
            loss=losses, top1=scores, mIOU=np.nanmean(ious),
            mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
    if class_names is not None:
      debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
    debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'
    
    wandb_log_dict = {}
    if data_type != 'testing':

      for i in range(len(ious)):
        iou = ious[i]
        class_name = str(i)

        if class_names:
          if i < len(class_names):
            class_name = class_names[i]
            print('Class name mapping', i, class_name)
            wandb_log_dict['%s/IoU_%s' % (data_type, class_name)] = iou
      
  logging.info(debug_str)
  return wandb_log_dict


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, average=None)


def test(curr_train_iter, model, data_loader, config, transform_data_fn=None, has_gt=True, data_type='validation'):

  if config.trained_model_path:
    checkpoint_fn = config.trained_model_path + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      model.load_state_dict(state['state_dict'])
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()



  # testing mode => output the prediction, original point cloud is needed
  if data_type == 'testing':
    pred_res_list = []
    # write the prediction result of test set
    scene_names = []
    with open(TESTING_FILES_TXT, 'r') as f:
      scene_names = f.readlines()
    scene_names = [s.split()[0] for s in scene_names]

    for scene_name in scene_names:
      scene_path = os.path.join(TESTING_SCANS_ROOT, '%s.ply' % scene_name)
      coords, colors, pcd = load_file(scene_path)

      with torch.no_grad():

          # Currently we are using 0.05 voxel size
          voxel_size = 0.05
          # Feed-forward pass and get the prediction
          in_field = ME.TensorField(
              features=normalize_color(torch.from_numpy(colors)),
              coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
              quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
              minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
              device=device,
          )
          # Convert to a sparse tensor
          sinput = in_field.sparse()
          # Output sparse tensor
          soutput = model(sinput)
          # get the prediction on the input tensor field
          out_field = soutput.slice(in_field)
          logits = out_field.F

      _, pred = logits.max(1)
      pred = [VALID_CLASS_IDS_200[idx] for idx in pred.tolist()]
      pred_res_list.append(pred)

    cwd = os.getcwd()
    test_set_prediction_root = os.path.join(cwd, 'test_set_prediction')
    if not os.path.isdir(test_set_prediction_root):
      os.mkdir(test_set_prediction_root)

    for i in range(len(pred_res_list)):
      pred_res = pred_res_list[i]
      scene_name = scene_names[i]
      result_txt = '%s.txt' % scene_name
      with open(os.path.join(test_set_prediction_root, result_txt), 'w') as f:
        for res in pred_res:
          f.write('%d\n' % res)
    return


  # validation mode => do not output the prediction, no original point cloud is needed
  else:
    with torch.no_grad():
      for iteration in range(max_iter):
        data_timer.tic()
        if config.return_transformation:
          coords, input, target, pointcloud, transformation = data_iter.next()
        else:
          coords, input, target = data_iter.next()
        data_time = data_timer.toc(False)

        # Preprocess input
        iter_timer.tic()

        if config.normalize_color:
          input[:, :3] = input[:, :3] / 255. - 0.5
        sinput = ME.SparseTensor(input, coords, device=device) #.to(device)

        # Feed forward
        inputs = (sinput,)
        soutput = model(*inputs)
        output = soutput.F
        
        pred = get_prediction(dataset, output, target).int()
        iter_time = iter_timer.toc(False)

        if has_gt:
          target_np = target.numpy()
          num_sample = target_np.shape[0]
          target = target.to(device)

          cross_ent = criterion(output, target.long())
          losses.update(float(cross_ent), num_sample)
          scores.update(precision_at_one(pred, target), num_sample)
          hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
          ious = per_class_iu(hist) * 100

          prob = torch.nn.functional.softmax(output, dim=1)
          ap = average_precision(prob.cpu().detach().numpy(), target_np)
          aps = np.vstack((aps, ap))
          # Due to heavy bias in class, there exists class with no test label at all
          with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ap_class = np.nanmean(aps, 0) * 100.

        # if iteration == max_iter - 1 and iteration > 0:
        #   reordered_ious = dataset.reorder_result(ious)
        #   reordered_ap_class = dataset.reorder_result(ap_class)
        #   class_names = dataset.get_classnames()
        #   print_info(
        #       iteration,
        #       max_iter_unique,
        #       data_time,
        #       iter_time,
        #       has_gt,
        #       losses,
        #       scores,
        #       reordered_ious,
        #       hist,
        #       reordered_ap_class,
        #       class_names=class_names,
        #       data_type=data_type)
        if iteration % config.empty_cache_freq == 0:
          # Clear cache
          torch.cuda.empty_cache()

    global_time = global_timer.toc(False)

    reordered_ious = dataset.reorder_result(ious)
    reordered_ap_class = dataset.reorder_result(ap_class)
    class_names = dataset.get_classnames()
    wandb_log_dict = print_info(
          iteration,
          max_iter_unique,
          data_time,
          iter_time,
          has_gt,
          losses,
          scores,
          reordered_ious,
          hist,
          reordered_ap_class,
          class_names=class_names,
          data_type=data_type
        )
    
    logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
      data_iter.cleanup()

    return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100, wandb_log_dict
