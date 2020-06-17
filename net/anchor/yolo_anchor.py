# -*- coding: utf-8 -*-
"""
# @author  : 秦丹峰
# @contact : daneven.jim@gmail.com
# @file    : yolo_anchor.py
# @time    : 20-06-14 15:39
# @desc    : 根据标注信息生成对应的锚框
"""

import numpy as np
from utils.bbox_tranform import xywh2xyxy, iou_xyxy_numpy


# 标注预测框的objectness
def get_objectness_label(img,
                         gt_boxes,
                         gt_labels,
                         iou_threshold = 0.7,
                         anchors = [116, 90, 156, 198, 373, 326],
                         num_classes=7,
                         downsample=32):
    """
    根据真实的标注框转换成相对于的锚框
    :param img           : 输入的图像数据，NCHW
    :param gt_boxes      : numpy，真实框，维度是[N, 50, 4]，其中50是真实框数目的上限，当图片中真实框不足50个时，不足部分的坐标全为0
                           真实框坐标格式是[x_center, y_center,w,h]，这里使用相对值
    :param gt_labels     : numpy，真实框所属类别，维度是[N, 50]
    :param iou_threshold : 当预测框与真实框的iou大于iou_threshold时不将其看作是负样本
    :param anchors       : 列表，锚框可选的尺寸，这里是其中一个尺度的锚框，共3个尺度
    :param num_classes   : 类别数目
    :param downsample    : 下采样，特征图相对于输入网络的图片尺寸变化的比例
    :return:
    """

    img_shape   = img.shape
    batchsize   = img_shape[0]
    num_anchors = len(anchors) // 2  # 每个栅格的锚框数
    input_h     = img_shape[2]  # 每个batch的图片都是相同的宽高
    input_w     = img_shape[3]

    # 将输入图片划分成num_rows x num_cols个小方块区域，每个小方块的边长是 downsample
    # 计算一共有多少行小方块、多少列小方块
    num_rows = input_h // downsample  # 特征图的高
    num_cols = input_w // downsample  # 特征图的宽

    label_objectness     = np.zeros([batchsize, num_anchors, num_rows, num_cols])
    label_classification = np.zeros([batchsize, num_anchors, num_classes, num_rows, num_cols])
    label_location       = np.zeros([batchsize, num_anchors, 4, num_rows, num_cols])

    scale_location = np.ones([batchsize, num_anchors, num_rows, num_cols])

    # 对batchsize进行循环，依次处理每张图片
    for n in range(batchsize):
        for n_gt in range(len(gt_boxes[n])):  # 对图片上的真实框进行循环，依次找出跟真实框形状最匹配的锚框
            gt          = gt_boxes[n][n_gt]   # 边框
            gt_cls      = gt_labels[n][n_gt]  # 类别
            gt_center_x = gt[0]
            gt_center_y = gt[1]
            gt_width    = gt[2]
            gt_height   = gt[3]
            if (gt_width < 1e-3) or (gt_height < 1e-3):
                continue
            i = int(gt_center_y * num_rows)  # 在特征图的相对位置，y_center
            j = int(gt_center_x * num_cols)  # 在特征图的相对位置，x_center
            ious = []

            for ka in range(num_anchors):  # 从3个锚框中找出最合适的一个
                bbox1    = [0., 0., float(gt_width), float(gt_height)]
                anchor_w = anchors[ka * 2]
                anchor_h = anchors[ka * 2 + 1]
                bbox2    = [0., 0., anchor_w/float(input_w), anchor_h/float(input_h)]

                # 计算iou
                bbox1 = xywh2xyxy(bbox1)
                bbox2 = xywh2xyxy(bbox2)
                iou = iou_xyxy_numpy(bbox1, bbox2)

                ious.append(iou)

            # 选择iou最大那个的锚框
            ious = np.array(ious)
            inds = np.argsort(ious)
            k = inds[-1]
            label_objectness[n, k, i, j] = 1
            c = gt_cls
            label_classification[n, k, c, i, j] = 1.

            # for those prediction bbox with objectness =1, set label of location
            """
            bx = cx + sigmoid(tx)  ---->   dx = sigmoid(tx) = bx - cx
            by = cy + sigmoid(ty)  ---->   dy = sigmoid(ty) = by - cy
            bw = pw * exp(tw)      ---->   dw = log(bw / pw)
            bh = ph * exp(th)      ---->   dh = log(bh / ph)
            其中，[bx,by,bw,bh]为真实标注框且bx和by为边框的中点，注意：边框已经转化成特征图的相对坐标
            """
            dx_label = gt_center_x * num_cols - j  # 0
            dy_label = gt_center_y * num_rows - i  # 0
            dw_label = np.log(gt_width * input_w / anchors[k*2])
            dh_label = np.log(gt_height * input_h / anchors[k*2 + 1])
            label_location[n, k, 0, i, j] = dx_label
            label_location[n, k, 1, i, j] = dy_label
            label_location[n, k, 2, i, j] = dw_label
            label_location[n, k, 3, i, j] = dh_label

            # scale_location用来调节不同尺寸的锚框对损失函数的贡献，作为加权系数和位置损失函数相乘
            scale_location[n, k, i, j] = 2.0 - gt_width * gt_height

    # 目前根据每张图片上所有出现过的gt box，都标注出了objectness为正的预测框，剩下的预测框则默认objectness为0
    # 对于objectness为1的预测框，标出了他们所包含的物体类别，以及位置回归的目标
    return label_objectness.astype('float32'), \
           label_location.astype('float32'), \
           label_classification.astype('float32'), \
           scale_location.astype('float32')


# 预测框筛选
def get_iou_above_thresh_inds(pred_box, gt_boxes, iou_threshold):
    """
    挑选出跟真实框IoU大于阈值的预测框
    :param pred_box: numpy, NHWA4, 其中A为anchor的数量
    :param gt_boxes:
    :param iou_threshold: iou阈值
    :return:
    """
    batchsize = pred_box.shape[0]
    num_rows = pred_box.shape[1]
    num_cols = pred_box.shape[2]
    num_anchors = pred_box.shape[3]
    ret_inds = np.zeros([batchsize, num_rows, num_cols, num_anchors])
    for i in range(batchsize):
        pred_box_i = pred_box[i]
        gt_boxes_i = gt_boxes[i]
        for k in range(len(gt_boxes_i)):  # gt in gt_boxes_i:
            gt = gt_boxes_i[k]
            gtx_min = gt[0] - gt[2] / 2.
            gty_min = gt[1] - gt[3] / 2.
            gtx_max = gt[0] + gt[2] / 2.
            gty_max = gt[1] + gt[3] / 2.
            if (gtx_max - gtx_min < 1e-3) or (gty_max - gty_min < 1e-3):
                continue
            x1 = np.maximum(pred_box_i[:, :, :, 0], gtx_min)
            y1 = np.maximum(pred_box_i[:, :, :, 1], gty_min)
            x2 = np.minimum(pred_box_i[:, :, :, 2], gtx_max)
            y2 = np.minimum(pred_box_i[:, :, :, 3], gty_max)
            intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)
            s1 = (gty_max - gty_min) * (gtx_max - gtx_min)
            s2 = (pred_box_i[:, :, :, 2] - pred_box_i[:, :, :, 0]) * (pred_box_i[:, :, :, 3] - pred_box_i[:, :, :, 1])
            union = s2 + s1 - intersection
            iou = intersection / union
            above_inds = np.where(iou > iou_threshold)
            ret_inds[i][above_inds] = 1
    ret_inds = np.transpose(ret_inds, (0, 3, 1, 2))
    return ret_inds.astype('bool')


def label_objectness_ignore(label_objectness, iou_above_thresh_indices):
    # 注意：这里不能简单的使用 label_objectness[iou_above_thresh_indices] = -1，
    #      这样可能会造成label_objectness为1的那些点被设置为-1了
    #      只有将那些被标注为0，且与真实框IoU超过阈值的预测框才被标注为-1
    negative_indices = (label_objectness < 0.5)
    ignore_indices = negative_indices * iou_above_thresh_indices
    label_objectness[ignore_indices] = -1
    return label_objectness