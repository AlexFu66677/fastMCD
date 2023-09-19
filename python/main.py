import numpy as np
import cv2
import MCDWrapper
import os
import xml.etree.ElementTree as ET
import json
import re
# 计算两个矩形框的IOU
def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)
    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# 读取数据集的真实标签
def read_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        labels.append([xmin, ymin, xmax - xmin, ymax - ymin])
    return labels

# 读取模型的预测结果
def read_predictions(json_file):
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    return predictions

# 计算准确率、召回率、F1分数和正负样本数
def calculate_metrics(labels, predictions,tp ,fp , fn ):
    for bbox in labels:
        iou_max = 0
        for bbox_pred in predictions:
            iou = calculate_iou(bbox, bbox_pred)
            if iou > iou_max:
                iou_max = iou
        if iou_max >= 0.5:
            tp += 1
        else:
            fn += 1

    for bbox in predictions:
        iou_max = 0
        for bbox_gt in labels:
            iou = calculate_iou(bbox, bbox_gt)
            if iou > iou_max:
                iou_max = iou
        if iou_max < 0.5:
            fp += 1
    return tp , fp , fn


def calculate_metrics25(labels, predictions,tp ,fp , fn ):
    for bbox in labels:
        iou_max = 0
        for bbox_pred in predictions:
            iou = calculate_iou(bbox, bbox_pred)
            if iou > iou_max:
                iou_max = iou
        if iou_max >= 0.1:
            tp += 1
        else:
            fn += 1

    for bbox in predictions:
        iou_max = 0
        for bbox_gt in labels:
            iou = calculate_iou(bbox, bbox_gt)
            if iou > iou_max:
                iou_max = iou
        if iou_max < 0.1:
            fp += 1
    return tp , fp , fn

# 示例用法


np.set_printoptions(precision=2, suppress=True)
cap = cv2.VideoCapture('/home/fjl/code/moving/fusebbox_SGM/python/data/car2.mp4')
folder_path_res=("/home/fjl/code/moving/fusebbox_SGM/python/data/car2/mseandtracker_next_mse_200")
folder_path_mask="/home/fjl/code/moving/fusebbox/python/data/car4/fd_mask_16"
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)
mcd = MCDWrapper.MCDWrapper()
isFirst = True
i = 0

# folder_path = "/home/fjl/下载/PESMOD/Pexels-Welton/annotations"
# folder_path = "/home/fjl/下载/PESMOD/Pexels-Miksanskiy/annotations"
folder_path = "/home/fjl/下载/PESMOD/Pexels-Elliot-road/annotations"
# folder_path = "/home/fjl/下载/PESMOD/Pexels-Marian/annotations"
# folder_path = "/home/fjl/下载/PESMOD/Pexels-Wolfgang/annotations"
# folder_path = "/home/fjl/下载/PESMOD/Pexels-Zaborski/annotations"
# folder_path = "/home/fjl/下载/PESMOD/Pexels-Shuraev-trekking/annotations"
# folder_path = "/home/fjl/下载/PESMOD/Pexels-Grisha-snow/annotations"


file_names = os.listdir(folder_path)
file_names.sort(key=lambda x: int(re.findall('\d+', x)[0]))
tp50 = 0
fp50 = 0
fn50 = 0
tp25 = 0
fp25 = 0
fn25 = 0
num_bbox=0
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
      gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    except:
       if frame is None:
           recall50 = tp50 / (tp50 + fn50)
           precision50 = (num_bbox-fp50) / num_bbox
           f1_score50 = 2 * (precision50 * recall50) / (precision50 + recall50)
           print('Precision50:', precision50)
           print('Recall50:', recall50)
           print('F1 Score50:', f1_score50)
           print('TP50:', tp50)
           print('FP50:', fp50)
           print('FN50:', fn50)

           recall25 = tp25 / (tp25 + fn25)
           precision25 = (num_bbox-fp25) / num_bbox
           f1_score25 = 2 * (precision25 * recall25) / (precision25 + recall25)
           print('Precision25:', precision25)
           print('Recall25:', recall25)
           print('F1 Score25:', f1_score25)
           print('TP25:', tp25)
           print('FP25:', fp25)
           print('FN25:', fn25)
           print(num_bbox)
           break
    mask = np.zeros(gray.shape, np.uint8)
    tracker_res=[]
    bgs_res=[]
    mse_res=[]
    if (isFirst):
        mcd.init(gray)
        isFirst = False
    else:
        bgs_res,tracker_res,mse_res,mask = mcd.run(gray)

    # for num in bgs_res:
    #     (x, y, w, h) = num
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for num in tracker_res:
    #     (x, y, w, h) = num
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for num in mse_res:
        (x, y, w, h) = num
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        num_bbox+=1
    frame[mask > 0, 2] = 255
    labels = read_labels(os.path.join(folder_path, file_names[i]))
    predictions = mse_res
    tp50, fp50, fn50 = calculate_metrics(labels, predictions, tp50, fp50, fn50)
    tp25, fp25, fn25 = calculate_metrics25(labels, predictions, tp25, fp25, fn25)


    i=i+1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if not os.path.exists(folder_path_mask):
        os.makedirs(folder_path_mask)
    if not os.path.exists(folder_path_res):
        os.makedirs(folder_path_res)
    # cv2.imwrite(os.path.join(folder_path_mask, str(i) + '.jpg'), mask)
    cv2.imwrite(os.path.join(folder_path_res, str(i) + '.jpg'), frame)