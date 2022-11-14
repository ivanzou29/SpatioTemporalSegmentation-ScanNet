import torch
from torch import nn


from lib.datasets.scannet import CLASS_LABELS, CLASS_LABELS_200, INSTANCE_COUNTER_200_TRAIN, INSTANCE_COUNTER_20_TRAIN, STEP_LEARN_STARTED_DICT_200, STEP_VAL_STARTED_DICT_200, STEP_ALMOST_LEARNED_DICT_200


def instance_count_reweight_loss(device, config, class_labels=CLASS_LABELS_200, instance_counter=INSTANCE_COUNTER_200_TRAIN, multiplier_constant=50):
    '''
        -reweight loss function that uses a weighted cross-entropy loss based on instance count of each class
        -class weight is inversely proportional to the instance count
        -multiplier constant is set by default as 50 to make the reweighted cross-entropy loss value similar to the previous one
    '''
    weight_tensor = torch.Tensor([(multiplier_constant / instance_counter[c]) if c in instance_counter else 0 for c in class_labels]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, weight=weight_tensor)
    return criterion

def class_difficulty_reweight_loss(device, config, class_labels=CLASS_LABELS_200, class_difficulty=STEP_LEARN_STARTED_DICT_200, divisor_constant=20000, threshold=0.5):
    '''
        -reweight loss function that uses the class learning difficulty defined by the time when each class begins to be learned
        -difficulty is proportional to the training steps when the training IoU of a specific class becomes > 0
        -for classes that never get learned during the initial experiment, a maximum weight threshold has been included in the difficulty dictionary
        -divisor constant is set by default 20000 to make reweighted cross-entropy loss value similar to the previous one
    '''

    weight_tensor = torch.Tensor([max(threshold, (class_difficulty[c] / divisor_constant)) for c in class_labels]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, weight=weight_tensor)

    return criterion

def cooccurrence_graph_reweight_loss(device, config, cooccurrence_graph):
    return


def focal_loss(device, class_difficulty=STEP_LEARN_STARTED_DICT_200, class_labels=CLASS_LABELS_200, gamma=2, threshold=0.5, divisor_constant=20000):
    alpha = torch.Tensor([max(threshold, (class_difficulty[c] / divisor_constant)) for c in class_labels]).to(device)
    return torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=alpha,
        gamma=gamma,
        reduction='mean',
        ignore_index=255
    )
