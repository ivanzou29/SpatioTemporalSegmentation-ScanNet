import torch
from torch import nn


from lib.datasets.scannet import CLASS_LABELS, CLASS_LABELS_200, INSTANCE_COUNTER_200_TRAIN, INSTANCE_COUNTER_20_TRAIN


def instance_count_reweight_loss(device, config, class_labels=CLASS_LABELS_200, instance_counter=INSTANCE_COUNTER_200_TRAIN, multiplier_constant=50):
    '''
        -reweight loss function that uses a weighted cross-entropy loss based on instance count of each class
        -class weight is inversely proportional to the instance count
        -multiplier constant is set defaultly as 50 to make the reweighted cross-entropy loss value similar to the previous one
    '''
    weightTensor = torch.Tensor([(multiplier_constant / instance_counter[c]) for c in class_labels]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, weight=weightTensor)
    return criterion

def class_difficulty_reweight_loss(device, config, class_difficulty, maximum_weight):
    '''
        -reweight loss function that uses the class learning difficulty defined by the time when each class begins to be learned
        -difficulty is proportional to the training steps when the training IoU of a specific class becomes > 0
        -for classes that never get learned during the initial experiment, a maximum weight is set as a threshold
    '''
    return

def cooccurrence_graph_reweight_loss(device, config, cooccurrence_graph):
    return
