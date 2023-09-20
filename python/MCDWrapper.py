import numpy as np
import cv2
import KLTWrapper
import ProbModel
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
def mse(image1, image2):
    # 计算均方误差
    # abs = (image1.astype("float") - image2.astype("float"))**2
    # abs = abs.flatten()
    # abs = abs[abs > 100]
    # err = abs.mean()
    err = (image1.astype("float") - image2.astype("float")) ** 2
    err = err.flatten()
    # err = err[err > 4]
    err = err.mean()
    # err = np.sum(err)
    # err /= float(image1.shape[0] * image1.shape[1])
    return err


def dhash(image, hash_size=4):
    image = Image.fromarray(image)
    return imagehash.average_hash(image, hash_size=hash_size)


def hamming_distance(hash1, hash2, hash_size=4):
    distance = hash1 - hash2
    similarity = 1 - (distance / (hash_size * hash_size))
    return similarity


def get_region(point, image_width, image_height):
    x, y = point
    region_width = image_width // 4
    region_height = image_height // 4
    row = y // region_height
    col = x // region_width
    return row * 4 + col


def calculate_intersection_area(rectangle1, rectangle2):
    x1, y1, w1, h1 = rectangle1
    x2, y2, w2, h2 = rectangle2
    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = intersection_x * intersection_y
    return intersection_area


def calculate_union_area(rectangle1, rectangle2):
    x1, y1, w1, h1 = rectangle1
    x2, y2, w2, h2 = rectangle2
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - calculate_intersection_area(rectangle1, rectangle2)
    return union_area


def calculate_iou(bbox, track_box):
    intersection_area = calculate_intersection_area(bbox, track_box)
    union_area = calculate_union_area(bbox, track_box)
    iou = intersection_area / union_area
    return iou


def combine_rectangles(rectangles):
    combined_rectangles = []
    for i in range(len(rectangles)):
        is_overlapping = False
        rect = rectangles[i]
        for oth_rect in rectangles:
            if rect == oth_rect:
                continue
            combined_rect = [0, 0, 0, 0]
            combined_rect[0] = min(oth_rect[0], rect[0])
            combined_rect[1] = min(oth_rect[1], rect[1])
            x1 = max(oth_rect[0] + oth_rect[2], rect[0] + rect[2])
            y1 = max(oth_rect[1] + oth_rect[3], rect[1] + rect[3])
            if x1 - combined_rect[0] <= oth_rect[2] + rect[2] and y1 - combined_rect[1] <= oth_rect[3] + rect[3]:
                combined_rect[2] = x1 - combined_rect[0]
                combined_rect[3] = y1 - combined_rect[1]
                combined_rectangles.append(combined_rect)
                is_overlapping = True
                break
        if not is_overlapping:
            combined_rectangles.append(rect)
    combined_rectangles = np.unique(combined_rectangles, axis=0)
    return combined_rectangles


class tracker:
    def __init__(self, x, y, w, h):
        self.bgs_hitCounter = 0
        self.bgs_missCounter = 0
        self.hitCounter = 0
        self.missCounter = 0
        self.update_status = True
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = (x, y, w, h)

    def calculate_iou(self, bbox):
        intersection_area = calculate_intersection_area(bbox, self.track_box)
        union_area = calculate_union_area(bbox, self.track_box)
        iou = intersection_area / union_area
        return iou

    def update(self, bbox):
        self.box = bbox
        self.update_status = True


class MCDWrapper:
    def __init__(self):
        self.imgIpl = None
        self.imgGray = None
        self.imgGrayPrev = None
        self.frm_cnt = 0
        self.lucasKanade = KLTWrapper.KLTWrapper()
        self.model = ProbModel.ProbModel()
        self.tracked_list = None
        self.bgs_tracked_list = None
        self.fream_num = None

    def init(self, image):
        self.imgGray = image
        self.imgGrayPrev = image
        self.lucasKanade.init(self.imgGray)
        self.model.init(self.imgGray)
        self.tracked_list = []
        self.bgs_tracked_list = []
        self.fream_num = 0

    def run(self, frame):
        framediff_res = []
        background_res = []
        bgs_tracked_res = []
        fd_tracked_res = []
        self.frm_cnt += 1
        self.imgIpl = frame
        self.imgGray = frame
        if self.imgGrayPrev is None:
            self.imgGrayPrev = self.imgGray.copy()

        self.lucasKanade.RunTrack(self.imgGray, self.imgGrayPrev)
        if self.fream_num == 119:
            self.model.alexinit(frame)

        self.model.motionCompensate(self.lucasKanade.H)

        if self.fream_num == 130:
            self.model.exchange_model()
            self.fream_num = 0

        bgs_mask = self.model.update(frame)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(bgs_mask, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 40:
                continue
            background_res.append((x, y-2, w, h+2))
        for tracked in self.bgs_tracked_list:
            tracked.update_status = False
        if len(self.bgs_tracked_list) == 0:
            for contour in contours:
                if cv2.contourArea(contour) < 40:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                self.bgs_tracked_list.append(tracker(x, y-2, w, h+2))
        else:
            for box in background_res:
                num = get_region((box[0], box[1]), 1920, 1080)
                new_point = cv2.perspectiveTransform(np.asarray([box[0], box[1]]).reshape(-1, 1, 2).astype(np.float32),
                                                     self.lucasKanade.H[num]).reshape(-1)
                matched = False
                for tracked in self.bgs_tracked_list:
                    if calculate_iou((new_point[0],new_point[1],box[2],box[3]), tracked.box) > 0.4:
                # matched = False
                # for tracked in self.bgs_tracked_list:
                #     if calculate_iou(box, tracked.box) > 0.4:
                        tracked.update(box)
                        matched = True
                        tracked.hitCounter += 1
                        tracked.missCounter = 0
                if not matched:
                    self.bgs_tracked_list.append(tracker(box[0], box[1], box[2], box[3]))
            self.bgs_tracked_list = [obj for obj in self.bgs_tracked_list if obj.missCounter < 3]
            for i in range(len(self.bgs_tracked_list)):
                if not self.bgs_tracked_list[i].update_status:
                    self.bgs_tracked_list[i].missCounter += 1
                    # bgs_tracked_res.append(self.bgs_tracked_list[i].box)
                    continue
            # self.bgs_tracked_list = [obj for obj in self.bgs_tracked_list if obj.missCounter < 3]
            # for i in range(len(self.bgs_tracked_list)):
                if self.bgs_tracked_list[i].hitCounter > 2:
                    bgs_tracked_res.append(self.bgs_tracked_list[i].box)

        bgs_tracked_res = np.unique(bgs_tracked_res, axis=0)
        bgs_tracked_list_point = []
        new_bgs_tracked_res = []
        for i in bgs_tracked_res:
            num = get_region((i[0], i[1]), 1920, 1080)
            new_point = cv2.perspectiveTransform(np.asarray([i[0], i[1]]).reshape(-1, 1, 2).astype(np.float32),
                                                 self.lucasKanade.H[num]).reshape(-1)
            if new_point[0] < 0:
                new_point[0] = 0
            if new_point[0] >= 1920:
                new_point[0] = 1920 - 1
            if new_point[1] < 0:
                new_point[1] = 0
            if new_point[1] >= 1080:
                new_point[1] = 1080 - 1
            bgs_tracked_list_point.append([int(new_point[0]), int(new_point[1]), i[2], i[3]])

            old_bbox = self.imgGray[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]
            new_bbox = self.imgGrayPrev[int(new_point[1]):int(new_point[1]) + i[3],
                       int(new_point[0]):int(new_point[0]) + i[2]]
            if old_bbox.shape[0] != new_bbox.shape[0] or old_bbox.shape[1] != new_bbox.shape[1]:
                continue
            else:
                # hash1=dhash(old_bbox)
                # hash2=dhash(new_bbox)
                # rr=hamming_distance(hash1, hash2)
                # rr=ssim(old_bbox, new_bbox)
                # print(rr)
                rr = mse(old_bbox, new_bbox)
                if rr > 250:
                    new_bgs_tracked_res.append(i)

        self.imgGrayPrev = self.imgGray.copy()
        self.fream_num += 1
        # return bgs_tracked_res,fd_tracked_res
        new_bgs_tracked_res = np.unique(new_bgs_tracked_res, axis=0)
        return background_res,bgs_tracked_res, new_bgs_tracked_res,thresh
        # return bgs_tracked_res, bgs_tracked_list_point,thresh

