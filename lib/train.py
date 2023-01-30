# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.

import logging
import os.path as osp
from turtle import forward

from sklearn.model_selection import train_test_split

import sys

import torch
from torch import nn
import torch.nn.functional as F

import wandb
from lib.loss import class_difficulty_reweight_loss, instance_count_reweight_loss, focal_loss, dynamic_reweight_by_training_iou, DomainCalibratedLoss

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
    Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

from MinkowskiEngine import SparseTensor
from collections import Counter
from lib.datasets.scannet import CLASS_LABELS, CLASS_LABELS_200, INSTANCE_COUNTER_20_TRAIN, INSTANCE_COUNTER_200_TRAIN, STEP_LEARN_STARTED_DICT_200, STEP_VAL_STARTED_DICT_200, STEP_ALMOST_LEARNED_DICT_200
from lib.dataloader import CooccGraphSampler

def validate(model, data_loader, curr_iter, config, transform_data_fn, class_counter, data_type='validation'):
  v_loss, v_score, v_mAP, v_mIoU, wandb_log_dict = test(curr_iter, model, data_loader, config, transform_data_fn, data_type=data_type)

  wandb_log_dict['%s/mIoU' % data_type] = v_mIoU
  wandb_log_dict['%s/loss' % data_type] = v_loss
  wandb_log_dict['%s/precision_at_1' % data_type] = v_score
  wandb_log_dict['%s/step' % data_type] = curr_iter
  
  wandb.log(wandb_log_dict)

  ious = []

  class_labels = CLASS_LABELS if config.dataset[-3:] != '200' else CLASS_LABELS_200


  # This version only adap
  for c in class_labels:
    ious.append(wandb_log_dict['%s/IoU_%s' % (data_type, c)])

  return v_mIoU, ious


def train(model, data_loader, val_data_loader, val_train_data_loader, config, transform_data_fn=None):
  device = get_torch_device(config.is_cuda)
  # Set up the train flag for batch normalization
  model.train()

  # Configuration
  data_timer, iter_timer, dataloader_timer = Timer(), Timer(), Timer()
  # coords_timer, normalize_timer, sinput_timer, forward_timer, loss_timer, backward_timer = Timer(), Timer(), Timer(), Timer(), Timer(), Timer()
  data_time_avg, iter_time_avg, dataloader_time_avg = AverageMeter(), AverageMeter(), AverageMeter()
  # coords_time_avg, normalize_time_avg, sinput_time_avg, forward_time_avg, loss_time_avg, backward_time_avg = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

  losses, scores = AverageMeter(), AverageMeter()

  optimizer = initialize_optimizer(model.parameters(), config)
  scheduler = initialize_scheduler(optimizer, config)
  
  instance_counter = INSTANCE_COUNTER_200_TRAIN if config.dataset[-3:] == '200' else INSTANCE_COUNTER_20_TRAIN
  class_labels = CLASS_LABELS if config.dataset[-3:] != '200' else CLASS_LABELS_200


  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
  if config.reweight == 'instance':
    criterion = instance_count_reweight_loss(device=device, config=config, class_labels=class_labels, instance_counter=instance_counter)
  elif config.reweight == 'step_learn_started':
    criterion = class_difficulty_reweight_loss(device=device, config=config, class_labels=class_labels, class_difficulty=STEP_LEARN_STARTED_DICT_200)
  elif config.reweight == 'step_val_started':
    criterion = class_difficulty_reweight_loss(device=device, config=config, class_labels=class_labels, class_difficulty=STEP_VAL_STARTED_DICT_200)
  elif config.reweight == 'step_almost_learned':
    criterion = class_difficulty_reweight_loss(device=device, config=config, class_labels=class_labels, class_difficulty=STEP_ALMOST_LEARNED_DICT_200)
  elif config.reweight == 'focal_loss':
    criterion = focal_loss(device, class_difficulty=STEP_LEARN_STARTED_DICT_200, class_labels=class_labels, gamma=config.focal_loss_gamma)
  elif config.reweight == 'dynamic_iou_reweight':
    criterion = dynamic_reweight_by_training_iou(device, config.ignore_label, None)
  elif config.reweight == 'domain_calibrated_loss':
    criterion = DomainCalibratedLoss(device, config, class_labels=class_labels)
  
  class_counter = Counter()

  # To be removed when domain class counter is built
  domains = [
    'Misc.', 'Bathroom', 'Office', 'Conference Room', 'Hallway', 'Apartment', 'Living room / Lounge', 'Classroom',
    'Stairs', 'Bedroom / Hotel', 'Bookstore / Library', 'Kitchen', 'Lobby', 'Copy/Mail Room', 'ComputerCluster',
    'Dining Room', 'Game room', 'Laundry Room', 'Gym', 'Closet', 'Storage/Basement/Garage'
  ]
  domain_class_counter = {}
  for domain in domains:
    domain_class_counter = Counter()

  cooccGraphSampler = None

  if config.sampler == 'CooccGraphSampler':
    cooccGraphSampler = data_loader.sampler

  # Train the network
  logging.info('===> Start training')
  best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

  # logging.info('===> Testing before training')
  # val_miou, val_ious = validate(model, val_data_loader, curr_iter, config, transform_data_fn, class_counter, 'validation')
  # train_miou, train_ious = validate(model, val_train_data_loader, curr_iter, config, transform_data_fn, class_counter, 'training')

  if config.resume:
    checkpoint_fn = config.resume + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      curr_iter = state['iteration'] + 1
      epoch = state['epoch']
      model.load_state_dict(state['state_dict'])
      if config.resume_optimizer:
        scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
        optimizer.load_state_dict(state['optimizer'])
      if 'best_val' in state:
        best_val_miou = state['best_val']
        best_val_iter = state['best_val_iter']
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

  data_iter = data_loader.__iter__()
  while is_training:
    for iteration in range(len(data_loader) // config.iter_size):
      optimizer.zero_grad()
      data_time, dataloader_time, batch_loss = 0, 0, 0
      # coords_time, normalize_time, sinput_time, forward_time, loss_time, backward_time = 0, 0, 0, 0, 0, 0
      iter_timer.tic()

      for sub_iter in range(config.iter_size):
        # Get training data
        data_timer.tic()

        dataloader_timer.tic()

        scene_type = None

        if config.sampler == 'CooccGraphSampler':
          coords, input, target, scene_instance_counter = data_iter.next()

          target = cooccGraphSampler.ignoreHelper(scene_instance_counter, target, config.ignore_label)
        
        elif config.sampler == 'SceneTypeSampler':
          coords, input, target, scene_type = data_iter.next()
        elif config.return_transformation:
          coords, input, target, pointcloud, transformation = data_iter.next()
        else:
          coords, input, target = data_iter.next()
        
        dataloader_time += dataloader_timer.toc(False)

        # For some networks, making the network invariant to even, odd coords is important

        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

        # Preprocess input
        if config.normalize_color:
          input[:, :3] = input[:, :3] / 255. - 0.5

        sinput = SparseTensor(input, coords, device=device) #.to(device)

        data_time += data_timer.toc(False)

        # model.initialize_coords(*init_args)
        soutput = model(sinput)

        # The output of the network is not sorted
        target = target.long().to(device)

        if config.sampler == 'SceneTypeSampler' and config.reweight == 'domain_calibrated_loss':
          scene_type = torch.cat(scene_type, axis=0).long().to(device)

          loss = criterion(soutput.F, target.long(), scene_type)
          loss /= config.iter_size
          batch_loss += loss.item()
          loss.backward()
        
        else:
          loss = criterion(soutput.F, target.long())
          # Compute and accumulate gradient
          loss /= config.iter_size
          batch_loss += loss.item()
          loss.backward()

      # Update number of steps
      optimizer.step()
      scheduler.step()

      data_time_avg.update(data_time)
      iter_time_avg.update(iter_timer.toc(False))
      dataloader_time_avg.update(dataloader_time)

      pred = get_prediction(data_loader.dataset, soutput.F, target)
      score = precision_at_one(pred, target)
      losses.update(batch_loss, target.size(0))
      scores.update(score, target.size(0))

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
        debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
            epoch, curr_iter,
            len(data_loader) // config.iter_size, losses.avg, lrs)
        debug_str += "Score {:.3f}\tData time: {:.4f}, Iter time: {:.4f}, Dataloader time: {:.4f}".format(
            scores.avg, data_time_avg.avg, iter_time_avg.avg, dataloader_time_avg.avg)
        
        logging.info(debug_str)
        # Reset timers
        data_time_avg.reset()
        iter_time_avg.reset()
        # Write logs

        wandb_log_dict_train = {}

        wandb_log_dict_train['training/loss'] = losses.avg
        wandb_log_dict_train['training/precision_at_1'] = scores.avg
        wandb_log_dict_train['training/learning_rate'] = scheduler.get_lr()[0]
        wandb_log_dict_train['training/step'] = curr_iter

        wandb.log(wandb_log_dict_train)

        losses.reset()
        scores.reset()

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

      # Validation
      if curr_iter % config.val_freq == 0:
        val_miou, val_ious = validate(model, val_data_loader, curr_iter, config, transform_data_fn, class_counter, 'validation')
        if val_miou > best_val_miou:
          best_val_miou = val_miou
          best_val_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                     "best_val")
        logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

      if curr_iter % config.val_train_freq == 0:
        train_miou, train_ious = validate(model, val_train_data_loader, curr_iter, config, transform_data_fn, class_counter, 'training')
        logging.info("Current train miou: {:.3f} at iter {}".format(train_miou, curr_iter))


        if config.reweight =='dynamic_iou_reweight':
          criterion = dynamic_reweight_by_training_iou(device, config.ignore_label, train_ious)

        # Recover back
        model.train()

      # End of iteration
      curr_iter += 1

    epoch += 1

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()
  
  # Save the final model
  checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
  val_miou, val_ious = validate(model, val_data_loader, curr_iter, config, transform_data_fn, class_counter, 'validation')
  if val_miou > best_val_miou:
    best_val_miou = val_miou
    best_val_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
  logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))
