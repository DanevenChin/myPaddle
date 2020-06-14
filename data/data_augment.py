# -*- coding: utf-8 -*-
"""
# @author  : 秦丹峰
# @contact : daneven.jim@gmail.com
# @time    : 20-06-12 21:24
# @file    : data_augment.py
# @desc    : 数据增强
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
from utils.bbox_tranform import xywh2xyxy, xyxy2xywl


def multi_box_iou_xywh(box1, box2):
    """
    In this case, box1 or box2 can contain multi boxes.
    Only two cases can be processed in this method:
       1, box1 and box2 have the same shape, box1.shape == box2.shape
       2, either box1 or box2 contains only one box, len(box1) == 1 or len(box2) == 1
    If the shape of box1 and box2 does not match, and both of them contain multi boxes, it will be wrong.
    """
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    # [x_center,y_center,w,h] -> [xmin,ymin,xmax,ymax]
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0., a_max=None)  # 小于0则置为0
    inter_h = np.clip(inter_h, a_min=0., a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)

def box_crop(boxes, labels, crop, img_shape):
    '''
    获得裁剪框
    :param boxes: numpy,标注框,[x_center,y_center,w,h]
    :param labels: numpy,类别
    :param crop: 元组，[xmin,ymin,w,h]
    :param img_shape: 元组，(w,h)，原图的宽高
    :return: boxes: 裁剪框, [x_center,y_center,w,h]
            labels: 裁剪框对应的类别,
            mask.sum(): 裁剪框总数
    '''
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    # 将归一化的坐标转换成原图上的坐标
    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, \
                               (boxes[:, 0] + boxes[:, 2] / 2) * im_w  # [x_center,y_center,w,h] -> [xmin,ymin,xmax,ymax]
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, \
                               (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0  # 标注框中点
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(
        axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')

    # [xmin,ymin,xmax,ymax] -> [x_center,y_center,w,h]
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (
        boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (
        boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()

# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    '''
    随机改变亮暗、对比度和颜色等
    :param img: 输入图片
    :return: img: 增强后的图片
    '''
    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)
    return img

# 随机填充
def random_expand(img,
                  gtboxes,
                  max_ratio=4.,
                  fill=None,
                  keep_ratio=True,
                  thresh=0.5):
    '''
    随机填充
    :param img: 输入图片, BGR格式
    :param gtboxes: numpy, 归一化的真实框, [x_center,y_center,w,h]
    :param max_ratio: 原图的最大比率
    :param fill: 输入列表, 改变填充的颜色
    :param keep_ratio: bool, True代表长宽比率一致
    :param thresh: 随机的概率
    :return: unit8的图片, 归一化的真实框
    '''
    if random.random() > thresh:
        return img, gtboxes

    if max_ratio < 1.0:
        return img, gtboxes

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)  # 填充的高
    ow = int(w * ratio_x)  # 填充的宽
    off_x = random.randint(0, ow - w)  # 原图的填充xmin位置
    off_y = random.randint(0, oh - h)  # 原图的填充ymin位置

    out_img = np.zeros((oh, ow, c))
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0  # 设置填充的颜色

    # 坐标转换，仍然是归一化的坐标
    out_img[off_y:off_y + h, off_x:off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes

# 随机裁剪
def random_crop(img,
                boxes,
                labels,
                scales=[0.3, 1.0],
                max_ratio=2.0,
                constraints=None,
                max_trial=50):
    '''
    随机裁剪
    :param img:
    :param boxes: numpy，标注框，[x_center,y_center,w,h]
    :param labels: numpy, 标注框对应的类别
    :param scales: 裁剪框的尺度
    :param max_ratio: 纵横比最大值
    :param constraints: 列表，与标注框的iou限制
    :param max_trial: 选择合适裁剪框的重复次数
    :return: img: 裁剪后的图片,
             boxes: numpy, 裁剪边框, [x_center,y_center,w,h]
             labels: numpy, 裁剪类别
    '''
    if len(boxes) == 0:
        return img, boxes

    if not constraints:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0),
                       (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])  # 随机选择一个尺度
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))  # 随机选择长宽比
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[(crop_x + crop_w / 2.0) / w,
                                  (crop_y + crop_h / 2.0) / h,
                                  crop_w / float(w), crop_h / float(h)]])  # [xmin,ymin,w,h] -> [x_center,y_center,w,h],并归一化

            iou = multi_box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():  # 与标注框的iou在[min_iou, max_iou]之间
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))  # 随机选择一个裁剪框
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:  # 如果裁剪框为0，则继续循环
            continue
        print(crop)
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    img = np.asarray(img)
    return img, boxes, labels

# 随机缩放
def random_interp(img, size, interp=None):
    '''
    随机缩放，使用opencv进行resize
    :param img: 输入原图
    :param size: 缩放的尺寸
    :param interp: 缩放模式
    :return: 缩放后的图片
    '''
    interp_method = [
        cv2.INTER_NEAREST,   # 最近邻插值
        cv2.INTER_LINEAR,    # 双线性插值（默认设置）
        cv2.INTER_AREA,      # 使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
        cv2.INTER_CUBIC,     # 4x4像素邻域的双三次插值
        cv2.INTER_LANCZOS4,  # 8x8像素邻域的Lanczos插值
    ]
    if not interp or interp not in interp_method:  # 如果为空或者列表中没有该方法，则从列表中随机选择一种
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    return img

# 随机翻转
def random_flip(img, gtboxes, thresh=0.5):
    '''
    随机翻转
    :param img: 输入图片
    :param gtboxes: numpy，归一化的真实框
    :param thresh: 随机阈值
    :return: img: 翻转后的图片,
             gtboxes: numpy, 翻转后的归一化坐标
    '''
    if random.random() > thresh:
        print("1")
        img = img[:, ::-1, :]
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img.astype('uint8'), gtboxes

# 随机打乱真实框排列顺序
def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate(
        [gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]

# 图像增广方法汇总
def image_augment(img, gtboxes, gtlabels, size, means=None):
    '''
    在线数据增强入口
    :param img: 原图
    :param gtboxes: 真实标注框, 注意是归一化的标注框, [x_center,y_center,w,h]
    :param gtlabels: 真实类别
    :param size: 随机缩放中缩放尺寸的设置
    :param means: 列表, 随机填充中填充颜色的设置
    :return: img: float32, 增强后的图片
            gtboxes: float32, 增强图片对应的真实框
            gtlabels: int32, 增强图片对应的真实类别
    '''
    # 随机改变亮暗、对比度和颜色等
    img = random_distort(img)
    # 随机填充
    img, gtboxes = random_expand(img, gtboxes, fill=means)
    # 随机裁剪
    img, gtboxes, gtlabels, = random_crop(img, gtboxes, gtlabels)
    # 随机缩放
    img = random_interp(img, size)
    # 随机翻转
    img, gtboxes = random_flip(img, gtboxes)
    # 随机打乱真实框排列顺序
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')


if __name__ == '__main__':
    from data.xml_process import read_xml
    xml_path = r'/home\qindanfeng\work\deep_learning\train_ssd_mobilenet\roadsign_data\PascalVOC\Annotations/0001_real.xml'
    xml_path = xml_path.replace('\\', '/')
    (num_bbox, cood, image_full_name, full_path, width_value, height_value) = read_xml(xml_path)
    img = cv2.imread(full_path)
    # print("ori_img_shape:", img.shape, type(img))
    bbox = np.array(cood)
    bbox = bbox.reshape(-1,5)
    cls =bbox[:,4]
    bbox = bbox[:,:4]
    print(bbox)
    boxes = bbox.tolist()
    # 坐标转成xywh,并归一化，类别编码
    bbox = bbox.astype('float')
    bbox = xyxy2xywl(bbox)
    img_shape = img.shape
    bbox[:, 0] = bbox[:, 0] / img_shape[1]
    bbox[:, 1] = bbox[:, 1] / img_shape[0]
    bbox[:, 2] = bbox[:, 2] / img_shape[1]
    bbox[:, 3] = bbox[:, 3] / img_shape[0]
    cls = cls.tolist()
    label = []
    for c in cls:
        if c == 'phone':
            label.append(0)
        if c == 'drink':
            label.append(1)
    label = np.array(label)
    print(label)
    # print(boxes)

    #--- 随机改变亮暗、对比度和颜色等 ---
    # distort_img = random_distort(img)
    # cv2.imwrite("img/distort.jpg", distort_img)

    #---------- 随机填充 ----------#
    # gtboxes = bbox.astype('float')
    # print(gtboxes)
    # gtboxes = xyxy2xywl(gtboxes)
    # print(gtboxes)
    # img_shape = img.shape
    # print(img_shape)
    # gtboxes[:, 0] = gtboxes[:, 0] / img_shape[1]
    # gtboxes[:, 1] = gtboxes[:, 1] / img_shape[0]
    # gtboxes[:, 2] = gtboxes[:, 2] / img_shape[1]
    # gtboxes[:, 3] = gtboxes[:, 3] / img_shape[0]
    # print(gtboxes)
    # out_img, gtboxes = random_expand(img,
    #               gtboxes,
    #               max_ratio=4.,
    #               fill=[0.1,0.2,0.3],
    #               keep_ratio=True,
    #               thresh=0.5)
    # print(gtboxes)
    # print("random_expand:", out_img.shape)
    #
    # gtboxes = xywh2xyxy(gtboxes)
    # out_box = gtboxes.tolist()
    # print(out_box)
    # h,w,_ = out_img.shape
    # for box in out_box:
    #     cv2.rectangle(out_img, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0, 0, 255), 3)
    # print(type(img))
    # cv2.imwrite("img/expand.jpg", out_img)

    #---------- 随机裁剪 ----------#
    # crop_img, crop_box, crop_cls = random_crop(img,
    #             bbox,
    #             label,
    #             scales=[0.3, 1.0],
    #             max_ratio=2.0,
    #             constraints=None,
    #             max_trial=50)
    # bbox = xywh2xyxy(bbox)
    # bbox = bbox.tolist()
    # for box in bbox:
    #     xmin = box[0]*img_shape[1]
    #     ymin = box[1]*img_shape[0]
    #     xmax = box[2]*img_shape[1]
    #     ymax = box[3]*img_shape[0]
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
    # cv2.imwrite("img/ori.jpg", img)
    #
    # print(crop_box, crop_cls)
    # crop_shape = crop_img.shape
    # print(crop_shape)
    # crop_box = xywh2xyxy(crop_box)
    # crop_box = crop_box.tolist()
    # for box in crop_box:
    #     xmin = box[0]*crop_shape[1]
    #     ymin = box[1]*crop_shape[0]
    #     xmax = box[2]*crop_shape[1]
    #     ymax = box[3]*crop_shape[0]
    #     cv2.rectangle(crop_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
    # cv2.imwrite("img/crop.jpg", crop_img)

    #------------ 随机翻转 ------------#
    # flip_img, flip_box = random_flip(img, bbox, thresh=0)
    # flip_shape = flip_img.shape
    # print(flip_shape)
    # flip_box = xywh2xyxy(flip_box)
    # flip_box = flip_box.tolist()
    # print(flip_box)
    # for box in flip_box:
    #     print(box)
    #     xmin = int(box[0]*flip_shape[1])
    #     ymin = int(box[1]*flip_shape[0])
    #     xmax = int(box[2]*flip_shape[1])
    #     ymax = int(box[3]*flip_shape[0])
    #     print(xmin, ymin, xmax, ymax)
    #     cv2.rectangle(flip_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    # cv2.imwrite("img/flip.jpg", flip_img)

    #----- 随机打乱真实框排列顺序 -----#
    # print(bbox, label)
    # bbox, label = shuffle_gtbox(bbox, label)
    # print(bbox, label)

    #---------- 数据增强 ----------#
    img, gtboxes, gtlabels = image_augment(img, bbox, label, img.shape[1], means=[0,0,255])
    augment_shape = img.shape
    print(augment_shape)
    flip_box = xywh2xyxy(gtboxes)
    flip_box = flip_box.tolist()
    print(flip_box)
    for box in flip_box:
        print(box)
        xmin = int(box[0]*augment_shape[1])
        ymin = int(box[1]*augment_shape[0])
        xmax = int(box[2]*augment_shape[1])
        ymax = int(box[3]*augment_shape[0])
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    cv2.imwrite("img/augment.jpg", img)