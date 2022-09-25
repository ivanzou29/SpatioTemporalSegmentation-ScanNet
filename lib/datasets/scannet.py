# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

from lib.dataset import SparseVoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu

TEST_FULL_PLY_PATH = 'test/%s_vh_clean_2.ply'
FULL_EVAL_PATH = 'outputs/fulleval'

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

## ScanNet200 Benchmark constants ###
VALID_CLASS_IDS_200 = (
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
  72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
  155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
  488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191
)

CLASS_LABELS_200 = (
  'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
  'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
  'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
  'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
  'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
  'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
  'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
  'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
  'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
  'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
  'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress'
)

SCANNET_COLOR_MAP_200 = {
  0: (0., 0., 0.),
  1: (174., 199., 232.),
  2: (188., 189., 34.),
  3: (152., 223., 138.),
  4: (255., 152., 150.),
  5: (214., 39., 40.),
  6: (91., 135., 229.),
  7: (31., 119., 180.),
  8: (229., 91., 104.),
  9: (247., 182., 210.),
  10: (91., 229., 110.),
  11: (255., 187., 120.),
  13: (141., 91., 229.),
  14: (112., 128., 144.),
  15: (196., 156., 148.),
  16: (197., 176., 213.),
  17: (44., 160., 44.),
  18: (148., 103., 189.),
  19: (229., 91., 223.),
  21: (219., 219., 141.),
  22: (192., 229., 91.),
  23: (88., 218., 137.),
  24: (58., 98., 137.),
  26: (177., 82., 239.),
  27: (255., 127., 14.),
  28: (237., 204., 37.),
  29: (41., 206., 32.),
  31: (62., 143., 148.),
  32: (34., 14., 130.),
  33: (143., 45., 115.),
  34: (137., 63., 14.),
  35: (23., 190., 207.),
  36: (16., 212., 139.),
  38: (90., 119., 201.),
  39: (125., 30., 141.),
  40: (150., 53., 56.),
  41: (186., 197., 62.),
  42: (227., 119., 194.),
  44: (38., 100., 128.),
  45: (120., 31., 243.),
  46: (154., 59., 103.),
  47: (169., 137., 78.),
  48: (143., 245., 111.),
  49: (37., 230., 205.),
  50: (14., 16., 155.),
  51: (196., 51., 182.),
  52: (237., 80., 38.),
  54: (138., 175., 62.),
  55: (158., 218., 229.),
  56: (38., 96., 167.),
  57: (190., 77., 246.),
  58: (208., 49., 84.),
  59: (208., 193., 72.),
  62: (55., 220., 57.),
  63: (10., 125., 140.),
  64: (76., 38., 202.),
  65: (191., 28., 135.),
  66: (211., 120., 42.),
  67: (118., 174., 76.),
  68: (17., 242., 171.),
  69: (20., 65., 247.),
  70: (208., 61., 222.),
  71: (162., 62., 60.),
  72: (210., 235., 62.),
  73: (45., 152., 72.),
  74: (35., 107., 149.),
  75: (160., 89., 237.),
  76: (227., 56., 125.),
  77: (169., 143., 81.),
  78: (42., 143., 20.),
  79: (25., 160., 151.),
  80: (82., 75., 227.),
  82: (253., 59., 222.),
  84: (240., 130., 89.),
  86: (123., 172., 47.),
  87: (71., 194., 133.),
  88: (24., 94., 205.),
  89: (134., 16., 179.),
  90: (159., 32., 52.),
  93: (213., 208., 88.),
  95: (64., 158., 70.),
  96: (18., 163., 194.),
  97: (65., 29., 153.),
  98: (177., 10., 109.),
  99: (152., 83., 7.),
  100: (83., 175., 30.),
  101: (18., 199., 153.),
  102: (61., 81., 208.),
  103: (213., 85., 216.),
  104: (170., 53., 42.),
  105: (161., 192., 38.),
  106: (23., 241., 91.),
  107: (12., 103., 170.),
  110: (151., 41., 245.),
  112: (133., 51., 80.),
  115: (184., 162., 91.),
  116: (50., 138., 38.),
  118: (31., 237., 236.),
  120: (39., 19., 208.),
  121: (223., 27., 180.),
  122: (254., 141., 85.),
  125: (97., 144., 39.),
  128: (106., 231., 176.),
  130: (12., 61., 162.),
  131: (124., 66., 140.),
  132: (137., 66., 73.),
  134: (250., 253., 26.),
  136: (55., 191., 73.),
  138: (60., 126., 146.),
  139: (153., 108., 234.),
  140: (184., 58., 125.),
  141: (135., 84., 14.),
  145: (139., 248., 91.),
  148: (53., 200., 172.),
  154: (63., 69., 134.),
  155: (190., 75., 186.),
  156: (127., 63., 52.),
  157: (141., 182., 25.),
  159: (56., 144., 89.),
  161: (64., 160., 250.),
  163: (182., 86., 245.),
  165: (139., 18., 53.),
  166: (134., 120., 54.),
  168: (49., 165., 42.),
  169: (51., 128., 133.),
  170: (44., 21., 163.),
  177: (232., 93., 193.),
  180: (176., 102., 54.),
  185: (116., 217., 17.),
  188: (54., 209., 150.),
  191: (60., 99., 204.),
  193: (129., 43., 144.),
  195: (252., 100., 106.),
  202: (187., 196., 73.),
  208: (13., 158., 40.),
  213: (52., 122., 152.),
  214: (128., 76., 202.),
  221: (187., 50., 115.),
  229: (180., 141., 71.),
  230: (77., 208., 35.),
  232: (72., 183., 168.),
  233: (97., 99., 203.),
  242: (172., 22., 158.),
  250: (155., 64., 40.),
  261: (118., 159., 30.),
  264: (69., 252., 148.),
  276: (45., 103., 173.),
  283: (111., 38., 149.),
  286: (184., 9., 49.),
  300: (188., 174., 67.),
  304: (53., 206., 53.),
  312: (97., 235., 252.),
  323: (66., 32., 182.),
  325: (236., 114., 195.),
  331: (241., 154., 83.),
  342: (133., 240., 52.),
  356: (16., 205., 144.),
  370: (75., 101., 198.),
  392: (237., 95., 251.),
  395: (191., 52., 49.),
  399: (227., 254., 54.),
  408: (49., 206., 87.),
  417: (48., 113., 150.),
  488: (125., 73., 182.),
  540: (229., 32., 114.),
  562: (158., 119., 28.),
  570: (60., 205., 27.),
  572: (18., 215., 201.),
  581: (79., 76., 153.),
  609: (134., 13., 116.),
  748: (192., 97., 63.),
  776: (108., 163., 18.),
  1156: (95., 220., 156.),
  1163: (98., 141., 208.),
  1164: (144., 19., 193.),
  1165: (166., 36., 57.),
  1166: (212., 202., 34.),
  1167: (23., 206., 34.),
  1168: (91., 211., 236.),
  1169: (79., 55., 137.),
  1170: (182., 19., 117.),
  1171: (134., 76., 14.),
  1172: (87., 185., 28.),
  1173: (82., 224., 187.),
  1174: (92., 110., 214.),
  1175: (168., 80., 171.),
  1176: (197., 63., 51.),
  1178: (175., 199., 77.),
  1179: (62., 180., 98.),
  1180: (8., 91., 150.),
  1181: (77., 15., 130.),
  1182: (154., 65., 96.),
  1183: (197., 152., 11.),
  1184: (59., 155., 45.),
  1185: (12., 147., 145.),
  1186: (54., 35., 219.),
  1187: (210., 73., 181.),
  1188: (221., 124., 77.),
  1189: (149., 214., 66.),
  1190: (72., 185., 134.),
  1191: (42., 94., 198.),
}

INSTANCE_COUNTER = {
    'floor': 1555,
    'wall': 8281,
    'desk': 680,
    'door': 1485,
    'cabinet': 731,
    'picture': 862,
    'chair': 4665,
    'table': 1175,
    'window': 1212,
    'toilet': 256,
    'sink': 488,
    'bed': 370,
    'curtain': 347,
    'shower curtain': 144,
    'bathtub': 144,
    'bookshelf': 360,
    'refrigerator': 154,
    'counter': 104,
    'sofa': 1
}

INSTANCE_COUNTER_200 = {
    'office chair': 596,
    'floor': 1555,
    'copier': 70,
    'monitor': 765,
    'wall': 8281,
    'printer': 106,
    'desk': 680,
    'trash can': 1090,
    'bulletin board': 53,
    'door': 1485,
    'whiteboard': 327,
    'computer tower': 203,
    'keyboard': 246,
    'mattress': 12,
    'stair rail': 42,
    'object': 1313,
    'cabinet': 731,
    'end table': 147,
    'ceiling': 806,
    'doorframe': 768,
    'picture': 862,
    'mirror': 349,
    'shelf': 641,
    'ceiling light': 59,
    'plant': 331,
    'chair': 4665,
    'stool': 221,
    'table': 1175,
    'laundry detergent': 21,
    'laundry basket': 37,
    'laundry hamper': 65,
    'fan': 75,
    'ledge': 51,
    'window': 1212,
    'clothes': 248,
    'toilet': 256,
    'light': 93,
    'sink': 488,
    'shower': 48,
    'bathroom vanity': 126,
    'bag': 253,
    'towel': 570,
    'bottle': 226,
    'toilet paper': 291,
    'nightstand': 224,
    'bed': 370,
    'dresser': 213,
    'power outlet': 19,
    'radiator': 322,
    'book': 318,
    'windowsill': 45,
    'file cabinet': 217,
    'couch': 502,
    'tv': 219,
    'stand': 23,
    'laptop': 111,
    'pillar': 21,
    'paper towel dispenser': 129,
    'soap dispenser': 99,
    'coffee table': 258,
    'curtain': 347,
    'shower curtain': 144,
    'bathtub': 144,
    'suitcase': 118,
    'backpack': 479,
    'light switch': 61,
    'mat': 52,
    'shower curtain rod': 42,
    'trash bin': 52,
    'headphones': 12,
    'cup': 157,
    'box': 775,
    'paper bag': 39,
    'bookshelf': 360,
    'ball': 39,
    'refrigerator': 154,
    'kitchen counter': 140,
    'blanket': 72,
    'range hood': 59,
    'stove': 95,
    'calendar': 13,
    'recycling bin': 225,
    'telephone': 164,
    'lamp': 419,
    'pillow': 937,
    'closet door': 35,
    'storage bin': 63,
    'board': 100,
    'closet': 45,
    'tissue box': 73,
    'blinds': 35,
    'rail': 53,
    'stairs': 35,
    'sign': 44,
    'tray': 32,
    'coffee maker': 61,
    'water pitcher': 21,
    'poster': 13,
    'microwave': 141,
    'counter': 104,
    'basket': 48,
    'hair dryer': 21,
    'paper': 46,
    'jacket': 146,
    'clock': 58,
    'shoe': 17,
    'decoration': 60,
    'mailbox': 7,
    'armchair': 281,
    'mouse': 49,
    'fireplace': 13,
    'kitchen cabinet': 310,
    'dishwasher': 43,
    'scale': 23,
    'bathroom cabinet': 39,
    'plunger': 30,
    'dining table': 19,
    'furniture': 13,
    'column': 26,
    'bathroom stall door': 62,
    'bathroom stall': 71,
    'stuffed animal': 12,
    'bar': 66,
    'toilet paper dispenser': 29,
    'person': 46,
    'guitar': 28,
    'hat': 22,
    'paper cutter': 30,
    'tv stand': 61,
    'sofa chair': 129,
    'vacuum cleaner': 25,
    'piano': 24,
    'container': 43,
    'soap dish': 65,
    'shower door': 19,
    'alarm clock': 16,
    'closet wall': 90,
    'bathroom counter': 24,
    'bucket': 45,
    'bin': 28,
    'music stand': 14,
    'ottoman': 111,
    'dumbbell': 48,
    'projector screen': 15,
    'speaker': 43,
    'luggage': 12,
    'seat': 49,
    'handicap bar': 27,
    'projector': 12,
    'water cooler': 24,
    'ironing board': 16,
    'mini fridge': 87,
    'bench': 66,
    'toilet seat cover dispenser': 28,
    'divider': 20,
    'blackboard': 60,
    'dish rack': 35,
    'plate': 24,
    'shower head': 15,
    'machine': 14,
    'fire extinguisher': 27,
    'wardrobe': 1,
    'vent': 13,
    'tube': 41,
    'shower wall': 22,
    'ladder': 27,
    'dustpan': 12,
    'toilet paper holder': 13,
    'pipe': 25,
    'bowl': 24,
    'purse': 34,
    'case of water bottles': 15,
    'water bottle': 15,
    'washing machine': 23,
    'clothes dryer': 17,
    'folded chair': 14,
    'fire alarm': 14,
    'shower floor': 19,
    'coat rack': 14,
    'power strip': 13,
    'cart': 37,
    'closet rod': 24,
    'storage container': 39,
    'rack': 21,
    'storage organizer': 14,
    'paper towel roll': 39,
    'toaster': 17,
    'cushion': 6,
    'keyboard piano': 15,
    'broom': 22,
    'structure': 18,
    'potted plant': 12,
    'guitar case': 21,
    'oven': 23,
    'toaster oven': 14,
    'coffee kettle': 18,
    'crate': 12,
    'bicycle': 33,
    'candle': 12,
    'cd case': 24
}

class ScannetSparseVoxelizationDataset(SparseVoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
  # Original SCN uses
  # ELASTIC_DISTORT_PARAMS = ((2, 4), (8, 8))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2

  NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
  IS_FULL_POINTCLOUD_EVAL = True

  CLASS_NAMES = CLASS_LABELS

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.TrainVal: 'trainval_uncropped.txt',
      DatasetPhase.Test: 'scannetv2_test.txt'
  }

  def __init__(self,
               config,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = config.scannet_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])
  
  def get_classnames(self):
    return self.CLASS_NAMES


class ScannetSparseVoxelization2cmDataset(ScannetSparseVoxelizationDataset):
  VOXEL_SIZE = 0.02

class ScannetSparseVoxelizationDataset200(SparseVoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
  # Original SCN uses
  # ELASTIC_DISTORT_PARAMS = ((2, 4), (8, 8))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2

  NUM_LABELS = 1193  # Will be converted to 200 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(1193)) - set(VALID_CLASS_IDS_200))
  IS_FULL_POINTCLOUD_EVAL = True

  CLASS_NAMES = CLASS_LABELS_200

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.TrainVal: 'trainval_uncropped.txt',
      DatasetPhase.Test: 'scannetv2_test.txt'
  }

  def __init__(self,
               config,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = config.scannet_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])
  
  def get_classnames(self):
    return self.CLASS_NAMES

class ScannetSparseVoxelization2cmDataset200(ScannetSparseVoxelizationDataset200):
  VOXEL_SIZE = 0.02
