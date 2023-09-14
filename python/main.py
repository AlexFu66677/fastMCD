import numpy as np
import cv2
import MCDWrapper
import os

np.set_printoptions(precision=2, suppress=True)
cap = cv2.VideoCapture('/home/fjl/code/moving/fusebbox/python/data/car0.mp4')
folder_path_res="/home/fjl/code/moving/fusebbox/python/data/car0/bbox_12"
folder_path_mask="/home/fjl/code/moving/fusebbox/python/data/car4/fd_mask_16"
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)
mcd = MCDWrapper.MCDWrapper()
isFirst = True
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    tracker_res=[]
    res=[]
    if (isFirst):
        mcd.init(gray)
        isFirst = False
    else:
        res, tracker_res = mcd.run(gray)
    for num in res:
        (x, y, w, h) = num
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    i=i+1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if not os.path.exists(folder_path_mask):
        os.makedirs(folder_path_mask)
    if not os.path.exists(folder_path_res):
        os.makedirs(folder_path_res)
    # cv2.imwrite(os.path.join(folder_path_mask, str(i) + '.jpg'), mask)
    cv2.imwrite(os.path.join(folder_path_res, str(i) + '.jpg'), frame)
