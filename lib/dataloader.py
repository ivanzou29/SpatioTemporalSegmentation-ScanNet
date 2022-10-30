# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch
import random
import pickle5 as pickle
from torch.utils.data.sampler import Sampler
from lib.scannet200_splits import COMMON_CATS_SCANNET_200


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

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)

  next = __next__  # Python 2 compatibility
