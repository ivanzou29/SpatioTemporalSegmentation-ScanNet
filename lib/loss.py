import torch
import pickle5 as pickle
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


def focal_loss(device, class_difficulty=STEP_LEARN_STARTED_DICT_200, class_labels=CLASS_LABELS_200, gamma=0.5, threshold=0.5, divisor_constant=20000):
    alpha = torch.Tensor([max(threshold, (class_difficulty[c] / divisor_constant)) for c in class_labels]).to(device)
    return torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=alpha,
        gamma=gamma,
        reduction='mean',
        ignore_index=255
    )


def dynamic_reweight_by_training_iou(device, ignore_label, ious=None):
    # Reimburse the classes that have not been properly learned with larger weights
    # As 100 is the upper limit of IoU, we can set the weights proportional to the difference 100 - Class IoU
    # Normalizing the weights so that the sum of weights is the same as the number of classes

    if not ious:
        return nn.CrossEntropyLoss(ignore_index=ignore_label)
    else:
        num_classes = len(ious)

        weights = [(100 - (ious[i] if ious[i] else 0)) for i in range(num_classes)]
        weights_sum = sum(weights)
        weight_tensor = torch.Tensor([num_classes * (weights[i] / weights_sum) for i in range(num_classes)]).to(device)

        return nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight_tensor)


class DomainCalibratedLoss(nn.Module):
    def __init__(self, device, config, class_labels=CLASS_LABELS_200, dcc_pickle_path='lib/domain_class_counter.pickle'):
        """
            Constructor method to initialize an object of DomainCalibratedLoss

            Args:
                device (str): usually CUDA if supported, otherwise the computation can be slow
                config (dict): the training configuration used, for us to query the ignore_label
                class_labels (tuple): the tuple containing all the semantic classes to be evaluated in the task
                dcc_pickle_path (str): the path to the pickle file that contains the domain class counter to be used in calculating the loss
        """

        super(DomainCalibratedLoss, self).__init__()

        self.device = device
        self.ignore_label = config.ignore_label
        self.class_labels = class_labels
        self.dcc_pickle_path = dcc_pickle_path

        # self.domain_class_counter is the domain class counter
        self.domain_class_counter = None
        with open(dcc_pickle_path, 'rb') as handler:
            self.domain_class_counter = pickle.load(handler)
    
        if not self.domain_class_counter:
            raise ModuleNotFoundError
        
        self.all_domains = list(self.domain_class_counter.keys())

        # Transform domain_class_counter into a dictionary of torch.Tensors for easier access later
        for d in self.all_domains:
            self.domain_class_counter[d] = torch.Tensor([(self.domain_class_counter[d][c] if c in self.domain_class_counter[d] else 0) for c in range(len(self.class_labels))]).to(self.device)

    def forward(self, inputs, targets, domains):
        targets = targets.view(-1)
        domain_calibrated_loss = 0

        valid_idx = torch.where(targets != self.ignore_label)

        targets = targets[valid_idx]
        inputs = inputs[valid_idx]

        for i in range(len(inputs)):
            tar = targets[i].item()
            pred = inputs[i]
            domain_calibrated_loss += (-torch.log(
                self.domain_class_counter[domains[i]][tar] * torch.exp(pred[tar]) / 
                torch.dot(self.domain_class_counter[domains[i]], torch.exp(pred))
            ))

        domain_calibrated_loss /= len(inputs)
        return domain_calibrated_loss
