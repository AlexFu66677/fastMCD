import numpy as np
import cv2
import itertools
feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# 判断点属于哪个区域
def get_region(point, image_width, image_height):
    x, y = point
    region_width = image_width // 4
    region_height = image_height // 4
    row = y // region_height
    col = x // region_width
    return row * 4 + col
    # return row

# 将属于同一个区域的点放在同一个集合中
def group_points(points0,points1):
    points0_groups = [[] for _ in range(16)]
    points1_groups = [[] for _ in range(16)]
    for i in range(len(points0)):
        region = int(get_region(points0[i], 1920, 1080))
        points0_groups[region].append(points0[i])
        points1_groups[region].append(points1[i])
    return points0_groups,points1_groups

class KLTWrapper:
    def __init__(self):
        self.win_size = 10
        self.status = 0
        self.count = 0
        self.flags = 0

        self.image = None
        self.imgPrevGray = None
        self.H = None

        self.GRID_SIZE_W = 32
        self.GRID_SIZE_H = 24
        self.MAX_COUNT = 0
        self.points0 = None
        self.points1 = None


    def init(self, imgGray):

        (nj, ni) = imgGray.shape

        self.MAX_COUNT = (float(ni) / self.GRID_SIZE_W + 1.0) * (float(nj) / self.GRID_SIZE_H + 1.0)
        self.lk_params = dict(winSize=(self.win_size, self.win_size),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_MAX_ITER| cv2.TERM_CRITERIA_EPS, 20, 0.03))
        self. H = np.array([np.identity(3) for _ in range(16)])

    def InitFeatures(self, imgGray):

        self.quality = 0.01
        self.min_distance = 10

        (nj, ni) = imgGray.shape

        self.count = ni / self.GRID_SIZE_W * nj / self.GRID_SIZE_H

        lenI = ni / self.GRID_SIZE_W - 1
        lenJ = nj / self.GRID_SIZE_H - 1
        J = np.arange(lenI*lenJ) / lenJ * self.GRID_SIZE_W + self.GRID_SIZE_W / 2
        I = np.arange(lenJ*lenI) % lenJ * self.GRID_SIZE_H + self.GRID_SIZE_H / 2

        self.points1 = np.expand_dims(np.array(list(zip(J, I))), 1).astype(np.float32)
        self.points0, self.points1 = self.points1, self.points0

    def RunTrack(self, image, imgPrev):

        if self.count > 0:
            self.points1, _st, _err = cv2.calcOpticalFlowPyrLK(imgPrev, image, self.points0, None, **self.lk_params)
            good0 = self.points0[_st == 1]
            good1 = self.points1[_st == 1]
            self.count = len(good0)

        if self.count > 10:
            self.makeHomoGraphy(good0, good1)
        else:
            self.H = np.array([np.identity(3) for _ in range(16)])
        self.InitFeatures(image)

    def makeHomoGraphy(self, p0, p1):
        point0_groups ,point1_groups= group_points(p0,p1)
        a = [np.array([]) for _ in range(16)]
        b = [np.array([]) for _ in range(16)]
        for i in range(16):
          #更改前后向
          b[i]=np.asarray(point0_groups[i])
          a[i]=np.asarray(point1_groups[i])
        if len(a[0])<4:
            H0=np.identity(3)
        else:
            H0, status = cv2.findHomography(a[0], b[0], cv2.RANSAC, 1.0)
            if H0 is None:
               H0=np.identity(3)

        if len(a[1])<4:
            H1 = np.identity(3)
        else:
            H1, status = cv2.findHomography(a[1], b[1], cv2.RANSAC, 1.0)
            if H1 is None:
               H1 = np.identity(3)

        if len(a[2])<4:
            H2 = np.identity(3)
        else:
            H2, status = cv2.findHomography(a[2], b[2], cv2.RANSAC, 1.0)
            if H2 is None:
               H2= np.identity(3)

        if len(a[3])<4:
            H3 = np.identity(3)
        else:
            H3, status = cv2.findHomography(a[3], b[3], cv2.RANSAC, 1.0)
            if H3 is None:
               H3 = np.identity(3)

        if len(a[4])<4:
            H4 = np.identity(3)
        else:
            H4, status = cv2.findHomography(a[4], b[4], cv2.RANSAC, 1.0)
            if H4 is None:
               H4 = np.identity(3)

        if len(a[5])<4:
            H5 = np.identity(3)
        else:
            H5, status = cv2.findHomography(a[5], b[5], cv2.RANSAC, 1.0)
            if H5 is None:
               H5 = np.identity(3)

        if len(a[6])<4:
            H6 = np.identity(3)
        else:
            H6, status = cv2.findHomography(a[6], b[6], cv2.RANSAC, 1.0)
            if H6 is None:
               H6 = np.identity(3)

        if len(a[7])<4:
            H7 = np.identity(3)
        else:
            H7, status = cv2.findHomography(a[7], b[7], cv2.RANSAC, 1.0)
            if H7 is None:
               H7 = np.identity(3)

        if len(a[8])<4:
            H8 = np.identity(3)
        else:
            H8, status = cv2.findHomography(a[8], b[8], cv2.RANSAC, 1.0)
            if H8 is None:
               H8 = np.identity(3)

        if len(a[9])<4:
            H9 = np.identity(3)
        else:
            H9, status = cv2.findHomography(a[9], b[9], cv2.RANSAC, 1.0)
            if H9 is None:
               H9 = np.identity(3)

        if len(a[10])<4:
            H10 = np.identity(3)
        else:
            H10, status = cv2.findHomography(a[10], b[10], cv2.RANSAC, 1.0)
            if H10 is None:
               H10 = np.identity(3)

        if len(a[11])<4:
            H11 = np.identity(3)
        else:
            H11, status = cv2.findHomography(a[11], b[11], cv2.RANSAC, 1.0)
            if H11 is None:
               H11 = np.identity(3)

        if len(a[12])<4:
            H12 = np.identity(3)
        else:
            H12, status = cv2.findHomography(a[12], b[12], cv2.RANSAC, 1.0)
            if H12 is None:
               H12 = np.identity(3)

        if len(a[13])<4:
            H13 = np.identity(3)
        else:
            H13, status = cv2.findHomography(a[13], b[13], cv2.RANSAC, 1.0)
            if H13 is None:
               H13 = np.identity(3)

        if len(a[14])<4:
            H14 = np.identity(3)
        else:
            H14, status = cv2.findHomography(a[14], b[14], cv2.RANSAC, 1.0)
            if H14 is None:
               H14 = np.identity(3)

        if len(a[15])<4:
            H15= np.identity(3)
        else:
            H15, status = cv2.findHomography(a[15], b[15], cv2.RANSAC, 1.0)
            if H15 is None:
               H15 = np.identity(3)
        # H2, status = cv2.findHomography(a[2], b[2], cv2.RANSAC, 1.0)
        # if H2 is None:
        #     H2=np.identity(3)
        # H3, status = cv2.findHomography(a[3], b[3], cv2.RANSAC, 1.0)
        # if H3 is None:
        #     H3=np.identity(3)
        # H4, status = cv2.findHomography(a[4], b[4], cv2.RANSAC, 1.0)
        # if H4 is None:
        #     H4=np.identity(3)
        # H5, status = cv2.findHomography(a[5], b[5], cv2.RANSAC, 1.0)
        # if H5 is None:
        #     H5=np.identity(3)
        # H6, status = cv2.findHomography(a[6], b[6], cv2.RANSAC, 1.0)
        # if H6 is None:
        #     H6=np.identity(3)
        # H7, status = cv2.findHomography(a[7], b[7], cv2.RANSAC, 1.0)
        # if H7 is None:
        #     H7=np.identity(3)
        # H8, status = cv2.findHomography(a[8], b[8], cv2.RANSAC, 1.0)
        # if H8 is None:
        #     H8=np.identity(3)
        # H9, status = cv2.findHomography(a[9], b[9], cv2.RANSAC, 1.0)
        # if H9 is None:
        #     H9=np.identity(3)
        # H10, status = cv2.findHomography(a[10], b[10], cv2.RANSAC, 1.0)
        # if H10 is None:
        #     H10=np.identity(3)
        # H11, status = cv2.findHomography(a[11], b[11], cv2.RANSAC, 1.0)
        # if H11 is None:
        #     H11=np.identity(3)
        # H12, status = cv2.findHomography(a[12], b[12], cv2.RANSAC, 1.0)
        # if H12 is None:
        #     H12=np.identity(3)
        # H13, status = cv2.findHomography(a[13], b[13], cv2.RANSAC, 1.0)
        # if H13 is None:
        #     H13=np.identity(3)
        # H14, status = cv2.findHomography(a[14], b[14], cv2.RANSAC, 1.0)
        # if H14 is None:
        #     H14=np.identity(3)
        # H15, status = cv2.findHomography(a[15], b[15], cv2.RANSAC, 1.0)
        # if H15 is None:
        #     H15=np.identity(3)

        try:
            self.H = np.array([H0, H1, H2, H3,H4,H5,H6,H7,H8,H9,H10,H11,H12,H13,H14,H15])
        except:
            print("error")
        # self.H = np.array([H0, H1, H2, H3])