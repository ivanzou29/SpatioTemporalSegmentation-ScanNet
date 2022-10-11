import torch
from torch import nn


from lib.datasets.scannet import CLASS_LABELS, CLASS_LABELS_200, INSTANCE_COUNTER_200_TRAIN

def instance_count_reweight_loss(device, config, class_labels=CLASS_LABELS_200, instance_counter=INSTANCE_COUNTER_200_TRAIN):
    weightTensor = torch.Tensor([(50 / instance_counter[c]) for c in class_labels]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, weight=weightTensor)
    return criterion

def cooccurrence_graph_reweight_loss(device, config, cooccurrence_graph):
    return
