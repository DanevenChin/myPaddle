import numpy as np

def xyxy2xywl(bbox):
    '''
    坐标转换
    :param bbox: 输入numpy, [xmin,ymin,xmax,ymax]
    :return: box: 返回numpy, [x_center,y_center,w,h]
    '''
    box = np.zeros_like(bbox)
    box[...,0] = (bbox[...,0] + bbox[...,2]) / 2
    box[...,1] = (bbox[...,1] + bbox[...,3]) / 2
    box[...,2] = bbox[...,2] - bbox[...,0]
    box[...,3] = bbox[...,3] - bbox[...,1]
    return box

def xywh2xyxy(bbox):
    '''
    坐标转换
    :param bbox: 输入numpy, [x_center,y_center,w,h]
    :return: bbox: 返回numpy, [xmin,ymin,xmax,ymax]
    '''
    box = np.zeros_like(bbox)
    box[...,0] = bbox[...,0] - bbox[...,2]/2
    box[...,2] = bbox[...,0] + bbox[...,2]/2
    box[...,1] = bbox[...,1] - bbox[...,3]/2
    box[...,3] = bbox[...,1] + bbox[...,3]/2
    return box

if __name__ == '__main__':
    bbox = [[1,2,3,4],
            [5,6,7,8]]
    bbox = np.array(bbox)
    print(xyxy2xywl(bbox))