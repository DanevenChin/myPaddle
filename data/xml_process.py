#encoding:utf-8
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET
import os
import cv2
import time
import random
import numpy as np


def create_xml(save_xml_path,
               num_bbox,
               bbox,
               image_full_name,
               full_path,
               width_value,
               height_value):
    """
    创建图片对应的一个xml文件
    :param save_xml_path: xml保存的文件夹路径
    :param num_bbox: 边框数量
    :param bbox: 边框，格式为[xmin1,ymin1,xmax1,ymax1,class1,...,xminN,yminN,xmaxN,ymaxN,classN]
    :param image_full_name: xml对应的图片名
    :param full_path: xml对应的图片完整路径
    :param width_value: xml对应的图片宽度
    :param height_value: xml对应的图片高度
    :return:
    """

    doc = Document()  #创建DOM文档对象
    annotation = doc.createElement('annotation') #创建根元素
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('JPGImages')
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(image_full_name)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    path = doc.createElement('path')
    path_text = doc.createTextNode(full_path)
    path.appendChild(path_text)
    annotation.appendChild(path)

    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    database.appendChild(database_text)
    source.appendChild(database)
    annotation.appendChild(source)

    size = doc.createElement('size')
    # 宽
    width = doc.createElement('width')
    width_text = doc.createTextNode(str(width_value))
    width.appendChild(width_text)
    # 高
    height = doc.createElement('height')
    height_text = doc.createTextNode(str(height_value))
    height.appendChild(height_text)
    # 通道数
    depth = doc.createElement('depth')
    depth_text = doc.createTextNode('3')
    depth.appendChild(depth_text)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    annotation.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)


    num_bbox = int(num_bbox)
    for i in range(num_bbox):
        object = doc.createElement('object')
        # class name
        name = doc.createElement('name')  # 因为我们同一张图片都是同一类别
        name_text = doc.createTextNode(str(bbox[5*i+4]))
        name.appendChild(name_text)
        # pose
        pose = doc.createElement('pose')
        pose_text = doc.createTextNode('Unspecified')
        pose.appendChild(pose_text)
        # truncated
        truncated = doc.createElement('truncated')
        truncated_text = doc.createTextNode('0')
        truncated.appendChild(truncated_text)
        # difficult
        difficult = doc.createElement('difficult')
        difficult_text = doc.createTextNode('0')
        difficult.appendChild(difficult_text)
        # occluded
        occluded = doc.createElement('occluded')
        occluded_text = doc.createTextNode('0')
        occluded.appendChild(occluded_text)
        # bbox
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode(str(bbox[5*i+0]))
        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode(str(bbox[5*i+1]))
        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode(str(bbox[5*i+2]))
        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode(str(bbox[5*i+3]))
        xmin.appendChild(xmin_text)
        ymin.appendChild(ymin_text)
        xmax.appendChild(xmax_text)
        ymax.appendChild(ymax_text)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)

        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(difficult)
        object.appendChild(occluded)
        object.appendChild(bndbox)
        annotation.appendChild(object)

    # 输出文件
    f = open('{}/{}.xml'.format(save_xml_path.replace('/home', 'z:'), image_full_name.split('.')[0]),'w')
    doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
    f.close()

def read_xml(xml_path):
    """
    读取xml文件, 提取出需要用的信息
    :param xml_path: 需要读取的xml完整路径
    :return: num_bbox: 边框数量
            cood: 边框，格式为[xmin1,ymin1,xmax1,ymax1,class1,...,xminN,yminN,xmaxN,ymaxN,classN]
            image_full_name: xml对应的图片名
            full_path: xml对应的图片完整路径
            width_value: xml对应的图片宽度
            height_value: xml对应的图片高度
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_full_name = root.find("filename").text
    full_path = tree.find("path").text
    size = tree.find("size")
    width_value = size.find("width").text
    height_value = size.find("height").text
    cood = []
    for obj in tree.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        cood.append(xmin)
        cood.append(ymin)
        cood.append(xmax)
        cood.append(ymax)
        cood.append(name)
    num_bbox = len(cood)/5

    return (num_bbox, cood, image_full_name, full_path, width_value, height_value)

def check_xml(xml_path, check_dir=True):
    """
    可视化xml，检查坐标是否有误
    :param xml_path: 存放xml的文件夹
    :return:
    """
    if check_dir:
        xml_list = os.listdir(xml_path)
        for xml in xml_list:
            info = read_xml(xml_path + '/' + xml)
            img_path = xml_path.replace('Annotations', 'JPEGImages/') + info[2]
            img = cv2.imread(img_path)
            if xml.split('.')[0] != info[2].split('.')[0]:
                print("!" * 100)
            print(img.shape)
            for i in range(int(info[0])):
                xmin = info[1][5 * i]
                ymin = info[1][5 * i + 1]
                xmax = info[1][5 * i + 2]
                ymax = info[1][5 * i + 3]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255))
            cv2.imshow('check_xml', img)
            cv2.waitKey(1)
            time.sleep(0.8)
    else:
        info = read_xml(xml_path)
        print(info)
        img_path = xml_path.replace('Annotations', 'JPEGImages/').replace('.xml', '.jpg')
        print(img_path)
        img = cv2.imread(img_path)
        _, xml = os.path.split(xml_path)
        if xml.split('.')[0] != info[2].split('.')[0]:
            print("!" * 50, "xml 与 img 不一致", "!" * 50)
        print(img.shape)
        for i in range(int(info[0])):
            xmin = info[1][5 * i]
            ymin = info[1][5 * i + 1]
            xmax = info[1][5 * i + 2]
            ymax = info[1][5 * i + 3]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255))
        cv2.imshow('check_xml', img)
        cv2.waitKey(1)
        time.sleep(0.8)

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

def add_noobj_xml(xml_path, save_noobj_xml_path, noobj_num=3):
    xml_list = os.listdir(xml_path)
    for xml in sorted(xml_list):
        if os.path.exists(save_noobj_xml_path+'/'+xml):
            print("[INFO] pass:", xml)
            continue
        print('-'*50)
        print("[INFO] xml:", xml)
        (num_bbox, cood, image_full_name, full_path, width_value, height_value) = read_xml(xml_path + '/' + xml)
        print("[INFO] read info:", num_bbox, cood, image_full_name, full_path, width_value, height_value)
        noobj_num = 3
        new_cood = []
        for i in range(int(num_bbox)):

            x1 = cood[5 * i]
            y1 = cood[5 * i + 1]
            x2 = cood[5 * i + 2]
            y2 = cood[5 * i + 3]
            cls = cood[5 * i + 4]

            new_cood.append(x1)
            new_cood.append(y1)
            new_cood.append(x2)
            new_cood.append(y2)
            new_cood.append(cls)

            w = x2 - x1
            h = y2 - y1
            x = x1 + w / 2
            y = y1 + h / 2

            for j in range(noobj_num):
                iou = 1
                while iou > 0.3:
                    random_x = random.randint(int(0.25*float(width_value)), int(0.75*float(width_value)))
                    random_y = random.randint(int(0.25*float(height_value)), int(0.75*float(height_value)))

                    random_w_ratio = random.uniform(0.7, 1.3)
                    random_h_ratio = random.uniform(0.7, 1.3)
                    random_w = random_w_ratio * w
                    random_h = random_h_ratio * h

                    random_xmin = random_x - random_w / 2
                    random_xmax = random_x + random_w / 2
                    random_ymin = random_y - random_y / 2
                    random_ymax = random_y + random_y / 2

                    if random_xmin < 0 or random_ymin < 0 or random_xmax > int(width_value) or random_ymax > int(
                            height_value):
                        continue

                    iou = iou_xyxy_numpy([x1, y1, x2, y2], [random_xmin, random_ymin, random_xmax, random_ymax])

                new_cood.append(int(random_xmin))
                new_cood.append(int(random_ymin))
                new_cood.append(int(random_xmax))
                new_cood.append(int(random_ymax))
                new_cood.append("noobj")

        create_xml(save_noobj_xml_path, num_bbox + num_bbox * noobj_num, new_cood, image_full_name, full_path,
                   width_value, height_value)
        print("[INFO] xml saved!")

def change_xml(xml_path):
    xml_list = os.listdir(xml_path)
    for xml in xml_list:
        (num_bbox, cood, image_full_name, full_path, width_value, height_value) = read_xml(xml_path + '/' + xml)
        print(xml, full_path, image_full_name)
        full_path = full_path.replace(image_full_name, xml.replace('.xml', '.jpg'))
        image_full_name = xml.replace('.xml', '.jpg')
        print(full_path, image_full_name)
        create_xml(xml_path, num_bbox, cood, image_full_name, full_path, width_value, height_value)

if __name__ == '__main__':
    xml_path = r'z:\qindanfeng\work\deep_learning\train_ssd_mobilenet\roadsign_data\PascalVOC\Annotations'
    save_noobj_xml_path = r'z:\qindanfeng\work\deep_learning\train_ssd_mobilenet\roadsign_data\PascalVOC\Annotations_noobj'
    # check_xml(xml_path+'/phone_02055.xml', check_dir=False)
    add_noobj_xml(xml_path, save_noobj_xml_path)