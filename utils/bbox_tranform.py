# -*- coding: utf-8 -*-
"""
# @author  : 秦丹峰
# @contact : daneven.jim@gmail.com
# @time    : 20-06-12 21:24
# @file    : bbox_tranform.py
# @desc    : 边框坐标的转换
"""

import numpy as np


def xyxy2xywl(bbox):
    '''
    坐标转换
    :param bbox: 输入numpy, [xmin,ymin,xmax,ymax]
    :return: box: 返回numpy, [x_center,y_center,w,h]
    '''
    box = np.zeros_like(bbox)
    box[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    box[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    box[..., 2] = bbox[..., 2] - bbox[..., 0]
    box[..., 3] = bbox[..., 3] - bbox[..., 1]
    return box


def xywh2xyxy(bbox):
    '''
    坐标转换
    :param bbox: 输入numpy, [x_center,y_center,w,h]
    :return: bbox: 返回numpy, [xmin,ymin,xmax,ymax]
    '''
    box = np.zeros_like(bbox)
    box[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    box[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    box[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    box[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return box


def iou_xyxy_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


# 定义Sigmoid函数
def sigmoid(x):
    return 1. / (1.0 + np.exp(-x))


# 将网络特征图输出的[tx, ty, th, tw]转化成预测框的坐标[x1, y1, x2, y2]
def get_yolo_box_xxyy(pred, anchors, num_classes, downsample):
    """
    pred是网络输出特征图转化成的numpy.ndarray
    anchors 是一个list。表示锚框的大小，
            例如 anchors = [116, 90, 156, 198, 373, 326]，表示有三个锚框，
            第一个锚框大小[w, h]是[116, 90]，第二个锚框大小是[156, 198]，第三个锚框大小是[373, 326]
    """
    batchsize = pred.shape[0]
    num_rows = pred.shape[-2]
    num_cols = pred.shape[-1]

    input_h = num_rows * downsample
    input_w = num_cols * downsample

    num_anchors = len(anchors) // 2

    # pred的形状是[N, C, H, W]，其中C = NUM_ANCHORS * (5 + NUM_CLASSES)
    # 对pred进行reshape
    pred = pred.reshape([-1, num_anchors, 5 + num_classes, num_rows, num_cols])
    pred_location = pred[:, :, 0:4, :, :]
    pred_location = np.transpose(pred_location, (0, 3, 4, 1, 2))
    anchors_this = []
    for ind in range(num_anchors):
        anchors_this.append([anchors[ind * 2], anchors[ind * 2 + 1]])
    anchors_this = np.array(anchors_this).astype('float32')

    # 最终输出数据保存在pred_box中，其形状是[N, H, W, NUM_ANCHORS, 4]，
    # 其中最后一个维度4代表位置的4个坐标
    pred_box = np.zeros(pred_location.shape)
    for n in range(batchsize):
        for i in range(num_rows):
            for j in range(num_cols):
                for k in range(num_anchors):
                    pred_box[n, i, j, k, 0] = j
                    pred_box[n, i, j, k, 1] = i
                    pred_box[n, i, j, k, 2] = anchors_this[k][0]
                    pred_box[n, i, j, k, 3] = anchors_this[k][1]

    # 这里使用相对坐标，pred_box的输出元素数值在0.~1.0之间
    pred_box[:, :, :, :, 0] = (sigmoid(pred_location[:, :, :, :, 0]) + pred_box[:, :, :, :, 0]) / num_cols
    pred_box[:, :, :, :, 1] = (sigmoid(pred_location[:, :, :, :, 1]) + pred_box[:, :, :, :, 1]) / num_rows
    pred_box[:, :, :, :, 2] = np.exp(pred_location[:, :, :, :, 2]) * pred_box[:, :, :, :, 2] / input_w
    pred_box[:, :, :, :, 3] = np.exp(pred_location[:, :, :, :, 3]) * pred_box[:, :, :, :, 3] / input_h

    # 将坐标从xywh转化成xyxy
    pred_box[:, :, :, :, 0] = pred_box[:, :, :, :, 0] - pred_box[:, :, :, :, 2] / 2.
    pred_box[:, :, :, :, 1] = pred_box[:, :, :, :, 1] - pred_box[:, :, :, :, 3] / 2.
    pred_box[:, :, :, :, 2] = pred_box[:, :, :, :, 0] + pred_box[:, :, :, :, 2]
    pred_box[:, :, :, :, 3] = pred_box[:, :, :, :, 1] + pred_box[:, :, :, :, 3]

    pred_box = np.clip(pred_box, 0., 1.0)

    return pred_box


if __name__ == '__main__':
    bbox = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    bbox = np.array(bbox)
    print(xyxy2xywl(bbox))
