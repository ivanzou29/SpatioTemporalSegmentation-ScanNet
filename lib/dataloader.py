# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch
import random
import numpy as np
import pickle5 as pickle
from torch.utils.data.sampler import Sampler
from lib.scannet200_splits import COMMON_CATS_SCANNET_200, CLASS_LABELS_200


class InfSampler(Sampler):
  """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

  def __init__(self, data_source, shuffle=False):
    self.data_source = data_source
    self.shuffle = shuffle
    self.reset_permutation()

  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)

  next = __next__  # Python 2 compatibility


class CommonClassesSampler(Sampler):

  """
      Samples scenes based on the appearances of instances of common classes

      Arguments:
          data_source (Dataset): dataset to sample from
          instance_counter_by_scene_path (path): path to dictionary storing the instance counter for each scene
          train_scene_list_path (path): path to the list of training scenes
          common_class_list (list): list of common classes predefined by ScanNet
          augmentation_factor (int): if augmentation factor is 2, the corresponding scenes containing common classes are replicated 2 more times
  """
  def __init__(
    self, 
    data_source, 
    instance_counter_by_scene_path='lib/scannet200_instance_counter/instance_counter_train_by_scene.pickle', 
    train_scene_list_path='lib/scene_list_train.pickle',
    common_class_list=COMMON_CATS_SCANNET_200,
    augmentation_factor=2,
    shuffle=False
  ):
    self.data_source = data_source

    self.instance_counter_by_scene = {}
    self.train_scene_list = []
    with open(instance_counter_by_scene_path, 'rb') as handler:
      self.instance_counter_by_scene = pickle.load(handler)
    
    with open(train_scene_list_path, 'rb') as handler:
      self.train_scene_list = pickle.load(handler)
    
    self.shuffle = shuffle
    self.common_class_set = set(common_class_list)
    self.augmentation_factor = augmentation_factor
    self.l = 0
    self.reset_permutation()
  
  def reset_permutation(self):
    perm_list = []
    for i in range(len(self.train_scene_list)):
      scene_name = self.train_scene_list[i]
      
      contains_common_class = False
      for c in self.instance_counter_by_scene[scene_name]:
        if c in self.common_class_set:
          contains_common_class = True
          break
      perm_list.append(i)
      if contains_common_class:
        for j in range(self.augmentation_factor):
          perm_list.append(i)
    
    self._perm = perm_list
    if self.shuffle:
      random.shuffle(self._perm)
    
    self.l = len(self._perm)

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return self.l

  next = __next__  # Python 2 compatibility


class CooccGraphSampler(Sampler):

  """
      Samples scenes randomly like a normal sampler
      However, class labels are sometimes ignored based on the global and scene-level coocc graph
      The idea is to hide some class labels so that more efforts will be focused on minor edges

      Arguments:
          data_source (Dataset): dataset to sample from
          train_scene_list_path (path): path to the list of training scenes
          scene_weight_dict_by_coocc_graph_path (path): path to scene weight dictionary by cooccurrence graph
          pairs_to_ignore_path (path): path to pairs of classes of different instances that we wish to add randomness and partially ignore
          instance_counter_by_scene_path (path): path to dictionary storing the instance counter for each scene

  """
  def __init__(
    self, 
    data_source, 
    train_scene_list_path='lib/scene_list_train.pickle',
    scene_weight_dict_by_coocc_graph_path='lib/scene_aug_dict_by_coocc_graph.pickle',
    pairs_to_ignore_path='lib/scannet200_instance_counter/coocc_graph_pairs_to_ignore.pickle',
    instance_counter_by_scene_path='lib/scannet200_instance_counter/instance_counter_train_by_scene.pickle',
    shuffle=False
  ):
    self.data_source = data_source

    self.train_scene_list = []
    self.scene_weight_dict_by_coocc_graph = {}
    self.pairs_to_ignore = []
    self.instance_counter_by_scene = {}
    
    with open(train_scene_list_path, 'rb') as handler:
      self.train_scene_list = pickle.load(handler)
    
    with open(scene_weight_dict_by_coocc_graph_path, 'rb') as handler:
      self.scene_weight_dict_by_coocc_graph = pickle.load(handler)
    
    with open(pairs_to_ignore_path, 'rb') as handler:
      self.pairs_to_ignore = pickle.load(handler)

    with open(instance_counter_by_scene_path, 'rb') as handler:
      self.instance_counter_by_scene = pickle.load(handler)
    
    self.pairs_to_ignore = set(self.pairs_to_ignore)

    self.class_labels = CLASS_LABELS_200
    self.class_label_to_index = {}
    for i in range(len(self.class_labels)):
      self.class_label_to_index[self.class_labels[i]] = i
    
    self.shuffle = shuffle
    self.l = 0
    self.reset_permutation()

    print('Initialization successful!')
  
  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)
  

  def ignoreHelper(self, scene_instance_counter_batch, labels_batch, ignore_label):

    bs = len(scene_instance_counter_batch)

    for batch_id in range(bs):
      scene_instance_counter = scene_instance_counter_batch[batch_id]
      labels = labels_batch[batch_id]

      class_list = list(scene_instance_counter.keys())
      n = len(class_list)

      for i in range(n - 1):
        for j in range(i + 1, n):
          if (class_list[i], class_list[j]) in self.pairs_to_ignore or (class_list[j], class_list[i]) in self.pairs_to_ignore:
            rand = random.randint(1, 3)
            
            if rand == 1:
              # ignore class_list[i]
              index_to_ignore = self.class_label_to_index[class_list[i]]
              labels[labels == index_to_ignore] = ignore_label
            elif rand == 2:
              # ignore class_list[j]
              index_to_ignore = self.class_label_to_index[class_list[j]]
              labels[labels == index_to_ignore] = ignore_label
            else:
              # ignore both
              index_to_ignore_list = [
                self.class_label_to_index[class_list[i]],
                self.class_label_to_index[class_list[j]]
              ]
              
              for index_to_ignore in index_to_ignore_list:
                labels[labels == index_to_ignore] = ignore_label

      labels_batch[batch_id] = labels

    return labels_batch

  next = __next__  # Python 2 compatibility
