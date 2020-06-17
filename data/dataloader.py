# -*- coding: utf-8 -*-
"""
# @author  : 秦丹峰
# @contact : daneven.jim@gmail.com
# @time    : 20-06-114 14:30
# @file    : dataloader.py
# @desc    : 数据加载
"""

from data.dataset import *
import functools
import paddle


# 随机尺度
def get_img_size(mode):
    """
    图片多尺度
    :param   mode    : 模式选择：train,valid or test
    :return: img_size: 返回图片尺寸
    """
    if (mode == 'train') or (mode == 'valid'):
        inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ii = np.random.choice(inds)
        img_size = 320 + ii * 32
    else:
        img_size = 608

    return img_size


# 数据转换
def make_array(batch_data):
    """
    将一个batch的数据转成numpy, 详情见下。                 |img| [img1,...,imgN]
                [(img1,box1,label1,scale1),             |box| [box1,...,boxN]
    |batch|      ...,                        --->     |label| [label1,...,labelN]
                 (imgN,boxN,labelN,scaleN)]           |scale| [scale1,...,scaleN]
    :param batch_data: 列表, 一个batch的数据
    :return:
    """
    img_array       = np.array([item[0] for item in batch_data], dtype='float32')
    gt_box_array    = np.array([item[1] for item in batch_data], dtype='float32')
    gt_labels_array = np.array([item[2] for item in batch_data], dtype='int32')
    img_scale       = np.array([item[3] for item in batch_data], dtype='int32')
    return img_array, gt_box_array, gt_labels_array, img_scale


# 数据加载
def data_loader(batch_size=10, mode='train'):
    """
    批量读取数据，同一批次内图像的尺寸大小必须是一样的，
    不同批次之间的大小是随机的，
    由上面定义的get_img_size函数产生
    :param batch_size:
    :param mode: 模式指定，训练、验证还是测试
    :return: 一个batch的数据
    """
    cname2cid = get_label_names()  # 获取总类别
    records = get_voc_annotations(cname2cid, mode=mode)  # 获取所有数据的相关信息

    def reader():
        if mode == 'train':  # 如果是训练，则将数据打乱
            np.random.shuffle(records)
        batch_data = []
        img_size = get_img_size(mode)  # 获取图片的随机尺寸
        for record in records:
            # print(record)
            # 读取单张图片，并返回图片数据、标注信息以及原始图片尺寸
            img, gt_bbox, gt_labels, im_shape = get_img_data(record,
                                                             size=img_size)
            batch_data.append((img, gt_bbox, gt_labels, im_shape))
            if len(batch_data) == batch_size:
                yield make_array(batch_data)  # 如果有一个batch的数据则返回
                batch_data = []
                img_size = get_img_size(mode)
        if len(batch_data) > 0:
            yield make_array(batch_data)

    return reader


# 多线程数据加载
def multithread_loader(batch_size=10, mode='train'):
    """
    使用paddle.reader.xmap_readers实现多线程读取数据
    :param batch_size:
    :param mode: 模式指定，训练、验证或测试
    :return: img_array, gt_box_array, gt_labels_array, img_scale
    """
    cname2cid = get_label_names()
    records   = get_voc_annotations(cname2cid, mode=mode)

    def reader():
        if mode == 'train':
            np.random.shuffle(records)
        img_size = get_img_size(mode)
        batch_data = []
        for record in records:
            batch_data.append((record, img_size))
            if len(batch_data) == batch_size:
                yield batch_data
                batch_data = []
                img_size = get_img_size(mode)
        if len(batch_data) > 0:
            yield batch_data

    def get_data(samples):
        batch_data = []
        for sample in samples:
            record   = sample[0]
            img_size = sample[1]
            img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=img_size)
            batch_data.append((img, gt_bbox, gt_labels, im_shape))
        return make_array(batch_data)

    mapper = functools.partial(get_data, )

    # reader是保存一个batch待处理的数据，mapper是将数据进行处理，
    # xmap_readers做的工作就是用mapper多线程处理reader的数据
    return paddle.reader.xmap_readers(mapper, reader, process_num=8, buffer_size=10)


# ---------------------- 测试数据读取 ----------------------#
# 测试数据转换
def make_test_array(batch_data):
    img_name_array  = np.array([item[0] for item in batch_data])
    img_data_array  = np.array([item[1] for item in batch_data], dtype='float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')
    return img_name_array, img_data_array, img_scale_array


# 测试数据加载
def test_data_loader(batch_size=10, test_image_size=608):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    image_names = os.listdir(cfg.TEST_IMAGE_DIR)

    def reader():
        batch_data = []
        img_size = test_image_size
        for image_name in image_names:
            file_path = os.path.join(cfg.TEST_IMAGE_DIR, image_name)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]
            img = cv2.resize(img, (img_size, img_size))

            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            mean = np.array(mean).reshape((1, 1, -1))
            std  = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) / std
            out_img = out_img.astype('float32').transpose((2, 0, 1))  # HWC -> CHW
            img = out_img
            im_shape = [H, W]

            batch_data.append((image_name.split('.')[0], img, im_shape))
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield make_test_array(batch_data)

    return reader
