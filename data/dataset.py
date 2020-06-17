# -*- coding: utf-8 -*-
"""
# @author  : 秦丹峰
# @contact : daneven.jim@gmail.com
# @time    : 20-06-12 21:24
# @file    : dataset.py
# @desc    : 数据集设置
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from data.data_augment import image_augment
import configs.yolo_config as cfg


def get_label_names():
    """
    将类别转成id, one-hot
    :return: insect_category2id: 返回字典，关于类别映射id的字典
    """

    insect_category2id = {}
    for i, item in enumerate(cfg.CLASSES["ALL"]):
        insect_category2id[item] = i

    return insect_category2id


def get_voc_annotations(cname2cid, mode='train'):
    """
    读取xml文件，获取图片相关信息
    :param   cname2cid: 一个字典，类别的id
    :param   mode     : train or valid
    :return: records  : 返回列表，其中包含每张图片的信息
    """
    if mode == 'train':
        annotation_dir = cfg.TRAIN_ANNOTATION_DIR
        image_dir = cfg.TRAIN_IMAGE_DIR
    elif mode == 'valid':
        annotation_dir = cfg.VAL_ANNOTATION_DIR
        image_dir = cfg.VAL_IMAGE_DIR

    filenames = os.listdir(annotation_dir)
    records   = []
    ct = 0
    for fname in filenames:
        fid   = fname.split('.')[0]
        fpath = os.path.join(annotation_dir, fname)
        tree  = ET.parse(fpath)
        filename = tree.find('filename').text
        assert filename.split('.')[0] == fid, "[ERROR] The xml file name does not correspond to the picture name!"
        img_file = os.path.join(image_dir, filename)

        if tree.find('id') is None:  # 如果xml中没有id，则用索引
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])

        objs      = tree.findall('object')
        im_w      = float(tree.find('size').find('width').text)
        im_h      = float(tree.find('size').find('height').text)
        gt_bbox   = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class  = np.zeros((len(objs),), dtype=np.int32)
        is_crowd  = np.zeros((len(objs),), dtype=np.int32)
        difficult = np.zeros((len(objs),), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text  # 类别
            gt_class[i] = cname2cid[cname]  # one-hot
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)

            # 检查边框是否数组越界
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)

            # 这里使用xywh格式来表示目标物体真实框,即 [xmin,ymin,xmax,ymax] -> [x_center,y_center,w,h]
            gt_bbox[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1 + 1., y2 - y1 + 1.]

            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file'  : img_file,  # 图片路径
            'im_id'    : im_id,     # 图片id, numpy
            'h'        : im_h,      # 图片高, float
            'w'        : im_w,      # 图片宽, float
            'is_crowd' : is_crowd,  # 是否拥挤, numpy
            'gt_class' : gt_class,  # 类别, numpy
            'gt_bbox'  : gt_bbox,   # 边框, numpy
            'gt_poly'  : [],        # poly,不知道是个啥
            'difficult': difficult  # 是否为难样本, numpy
        }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


def get_bbox(gt_bbox, gt_class):
    """
    对于一般的检测任务来说，一张图片上往往会有多个目标物体
    设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    :param   gt_bbox : 输入numpy格式,[bbox_num,4],边框格式为[x_center, y_center,w,h]
    :param   gt_class: 输入numpy格式,[class_num,]
    :return: gt_bbox : 返回numpy格式,[50,4],50个边框，边框格式为[x_center, y_center,w,h]
             gt_class: 返回numpy格式,[50,],50个类别
    """

    MAX_NUM   = 50
    gt_bbox2  = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i]   = gt_class[i]
        if i >= MAX_NUM:
            print("Warning 标注文件中的边框超过50个,多余的将被忽略!")
            break
    return gt_bbox2, gt_class2


def get_img_data_from_file(record):
    """
    根据标注信息读取图片,并将坐标归一化
    :param   record   : 输入字典,包含标注信息,详细见下
    :return: img      : 图片数据,
             gt_boxes : 返回numpy, 归一化的真实框, [50,4]
             gt_labels: 返回numpy, 类别, [50,],
             (h, w)   : 图片高宽
    record = {
            'im_file'  : img_file,
            'im_id'    : im_id,
            'h'        : im_h,
            'w'        : im_w,
            'is_crowd' : is_crowd,
            'gt_class' : gt_class,
            'gt_bbox'  : gt_bbox,
            'gt_poly'  : [],
            'difficult': difficult
            }
    """
    # 图片信息
    im_file   = record['im_file']
    h         = record['h']
    w         = record['w']
    is_crowd  = record['is_crowd']
    gt_class  = record['gt_class']
    gt_bbox   = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb

    # 检查读取的图片的长宽是否与标注信息中一致
    assert img.shape[0] == int(h), \
        "image height of {} inconsistent in record({}) and img file({})".format(
            im_file, h, img.shape[0])

    assert img.shape[1] == int(w), \
        "image width of {} inconsistent in record({}) and img file({})".format(
            im_file, w, img.shape[1])

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)  # 将边框,类别分别转换成[50,4],[50,]的格式

    # 将真实框归一化,[x_center/w, y_center/h, w/w, h/h]
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)


def get_img_data(record, size=640):
    """
    获取图片，并对图片做预处理
    :param record: 字典,图片的相关信息,包含标注信息
    :param size: 网络的输入尺寸
    :return: img: 预处理后的图片(已经resize),
            gt_boxes: numpy, 归一化的标注框,
            gt_labels: numpy, 真实类别,
            scales: 图片的原始宽高
    """
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)  # 获得一张图片以及对应的标签和宽高尺寸
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)  # 做数据增强
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std  = np.array(std).reshape((1, 1, -1))
    img  = (img / 255.0 - mean) / std  # 图片归一化
    img  = img.astype('float32').transpose((2, 0, 1))  # opencv的 HWC -> CHW
    return img, gt_boxes, gt_labels, scales
