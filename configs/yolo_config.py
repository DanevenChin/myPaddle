# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@license : (C) Copyright,
@time    : 2020/6/15 19:10
@file    : yolo_config.py
@desc    : yolov3配置文件
"""

# 数据集路径
TRAIN_ANNOTATION_DIR = '/home/qindanfeng/work/YOLOv3/datasets/VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations'
VAL_ANNOTATION_DIR = '/home/qindanfeng/work/YOLOv3/datasets/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
TEST_ANNOTATION_DIR = '/home/qindanfeng/work/YOLOv3/datasets/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'

TRAIN_IMAGE_DIR = '/home/qindanfeng/work/YOLOv3/datasets/VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'
VAL_IMAGE_DIR = '/home/qindanfeng/work/YOLOv3/datasets/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
TEST_IMAGE_DIR = '/home/qindanfeng/work/YOLOv3/datasets/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations'

# 类别
CLASSES = {"ALL": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
           "NUM": 20}

# 数据集类型，COCO或者VOC
DATASET_TYPE = "VOC"

# 锚框
ANCHORS = [10, 13, 16, 30, 33, 23,
           30, 61, 62, 45, 59, 119,
           116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8],
                [3, 4, 5],
                [0, 1, 2]]

IGNORE_THRESH = .7