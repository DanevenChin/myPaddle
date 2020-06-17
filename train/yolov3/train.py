# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@license : (C) Copyright,
@time    : 2020/6/14 16:03
@file    : train.py
@desc    : 模型训练
"""

import sys
import os
import time
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

sys.path.append("../..")
from net.yolov3 import YOLOv3
from data.dataloader import multithread_loader
import configs.yolo_config as cfg


def get_lr(base_lr=0.0001, lr_decay=0.1):
    bd = [10000, 20000]
    lr = []
    for i in range(len(bd)+1):
        lr.append(base_lr*lr_decay**i)
    # print(bd, lr)
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    return learning_rate


if __name__ == '__main__':
    batch_size = 10
    with fluid.dygraph.guard():
        model = YOLOv3(num_classes=cfg.CLASSES["NUM"], is_train=True)  # 创建模型
        learning_rate = get_lr()
        opt = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(0.0005),
            parameter_list=model.parameters())  # 创建优化器

        train_loader = multithread_loader(batch_size=batch_size, mode='train')  # 创建训练数据读取器
        valid_loader = multithread_loader(batch_size=batch_size, mode='valid')  # 创建验证数据读取器

        MAX_EPOCH = 200
        for epoch in range(MAX_EPOCH):
            model.train()
            for i, data in enumerate(train_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)  # 前向传播，输出[P0, P1, P2]
                loss = model.get_loss(outputs, gt_boxes, gt_labels,
                                      gtscore      =gt_scores,
                                      anchors      =cfg.ANCHORS,
                                      anchor_masks =cfg.ANCHOR_MASKS,
                                      ignore_thresh=cfg.IGNORE_THRESH,
                                      use_label_smooth=False)  # 计算损失函数

                loss.backward()  # 反向传播计算梯度
                opt.minimize(loss)  # 更新参数
                model.clear_gradients()
                if i % 1 == 0:
                    timestring = time.strftime("%H:%M:%S", time.localtime(time.time()))
                    print('{} [TRAIN] epoch:{}/{} | iter {} | output loss: {}'.format(
                        timestring, epoch, MAX_EPOCH, i, loss.numpy()))

            # 模型保存
            if (epoch % 1 == 0) or (epoch == MAX_EPOCH - 1):
                if not os.path.exists("save_model/yolov3"):
                    os.mkdir("save_model/yolov3")
                fluid.save_dygraph(model.state_dict(), '{}/{}'.format("save_model/yolov3", epoch))

            # 每个epoch结束之后在验证集上进行测试
            model.eval()
            for i, data in enumerate(valid_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors=cfg.ANCHORS,
                                      anchor_masks=cfg.ANCHOR_MASKS,
                                      ignore_thresh=cfg.IGNORE_THRESH,
                                      use_label_smooth=False)
                if i % 1 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))