import numpy as np
import cv2
import KLTWrapper
import ProbModel
from PIL import Image


def mse(image1, image2):
    # 计算均方误差
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


# def dhash(image, hash_size=8):
#     return imagehash.average_hash(Image.open(image), hash_size=hash_size)


def hamming_distance(hash1, hash2, hash_size=8):
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

    def init(self, image):
        self.imgGray = image
        self.imgGrayPrev = image
        self.lucasKanade.init(self.imgGray)
        self.model.init(self.imgGray)
        self.tracked_list = []
        self.bgs_tracked_list = []

    def fdCompensate(self, H, pre):
        height, width = pre.shape
        num_batch = 1
        pos_x = np.tile(np.arange(width), [height * num_batch])
        grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
        pos_y = np.tile(np.reshape(grid_y, [-1]), [num_batch])
        points = np.array([pos_x, pos_y])
        point = points.transpose().astype(np.float32).reshape(height, width, 1, 2)
        point0 = point[:height // 4, :width // 4, :, :].reshape(-1, 1, 2)
        point1 = point[:height // 4, width // 4:width // 2, :, :].reshape(-1, 1, 2)
        point2 = point[:height // 4, width // 2:-width // 4, :, :].reshape(-1, 1, 2)
        point3 = point[:height // 4, -width // 4:, :, :].reshape(-1, 1, 2)
        point4 = point[height // 4:height // 2, :width // 4, :, :].reshape(-1, 1, 2)
        point5 = point[height // 4:height // 2, width // 4:width // 2, :, :].reshape(-1, 1, 2)
        point6 = point[height // 4:height // 2, width // 2:-width // 4, :, :].reshape(-1, 1, 2)
        point7 = point[height // 4:height // 2, -width // 4:, :, :].reshape(-1, 1, 2)
        point8 = point[height // 2:-height // 4, :width // 4, :, :].reshape(-1, 1, 2)
        point9 = point[height // 2:-height // 4, width // 4:width // 2, :, :].reshape(-1, 1, 2)
        point10 = point[height // 2:-height // 4, width // 2:-width // 4, :, :].reshape(-1, 1, 2)
        point11 = point[height // 2:-height // 4, -width // 4:, :, :].reshape(-1, 1, 2)
        point12 = point[-height // 4:, :width // 4, :, :].reshape(-1, 1, 2)
        point13 = point[-height // 4:, width // 4:width // 2, :, :].reshape(-1, 1, 2)
        point14 = point[-height // 4:, width // 2:-width // 4, :, :].reshape(-1, 1, 2)
        point15 = point[-height // 4:, -width // 4:, :, :].reshape(-1, 1, 2)
        tempMean0 = cv2.perspectiveTransform(point0, H[0]).reshape(-1, width // 4, 1, 2)
        tempMean1 = cv2.perspectiveTransform(point1, H[1]).reshape(-1, width // 4, 1, 2)
        tempMean2 = cv2.perspectiveTransform(point2, H[2]).reshape(-1, width // 4, 1, 2)
        tempMean3 = cv2.perspectiveTransform(point3, H[3]).reshape(-1, width // 4, 1, 2)
        tempMean4 = cv2.perspectiveTransform(point4, H[4]).reshape(-1, width // 4, 1, 2)
        tempMean5 = cv2.perspectiveTransform(point5, H[5]).reshape(-1, width // 4, 1, 2)
        tempMean6 = cv2.perspectiveTransform(point6, H[6]).reshape(-1, width // 4, 1, 2)
        tempMean7 = cv2.perspectiveTransform(point7, H[7]).reshape(-1, width // 4, 1, 2)
        tempMean8 = cv2.perspectiveTransform(point8, H[8]).reshape(-1, width // 4, 1, 2)
        tempMean9 = cv2.perspectiveTransform(point9, H[9]).reshape(-1, width // 4, 1, 2)
        tempMean10 = cv2.perspectiveTransform(point10, H[10]).reshape(-1, width // 4, 1, 2)
        tempMean11 = cv2.perspectiveTransform(point11, H[11]).reshape(-1, width // 4, 1, 2)
        tempMean12 = cv2.perspectiveTransform(point12, H[12]).reshape(-1, width // 4, 1, 2)
        tempMean13 = cv2.perspectiveTransform(point13, H[13]).reshape(-1, width // 4, 1, 2)
        tempMean14 = cv2.perspectiveTransform(point14, H[14]).reshape(-1, width // 4, 1, 2)
        tempMean15 = cv2.perspectiveTransform(point15, H[15]).reshape(-1, width // 4, 1, 2)
        TempMean0 = np.concatenate([tempMean0, tempMean1, tempMean2, tempMean3], axis=1).reshape(-1, 1, 2)
        TempMean1 = np.concatenate([tempMean4, tempMean5, tempMean6, tempMean7], axis=1).reshape(-1, 1, 2)
        TempMean2 = np.concatenate([tempMean8, tempMean9, tempMean10, tempMean11], axis=1).reshape(-1, 1, 2)
        TempMean3 = np.concatenate([tempMean12, tempMean13, tempMean14, tempMean15], axis=1).reshape(-1, 1, 2)
        tempMean = np.concatenate([TempMean0, TempMean1, TempMean2, TempMean3], axis=0).reshape(-1, 1, 2)
        NewX = tempMean[:, :, 0].flatten().astype(np.int32)
        NewY = tempMean[:, :, 1].flatten().astype(np.int32)
        x0 = np.clip(NewX, 0, width - 1)
        y0 = np.clip(NewY, 0, height - 1)
        bg = np.ones([height, width]) * 255
        dim = width * height
        for i in range(dim):
            bg[y0[i]][x0[i]] = pre[pos_y[i]][pos_x[i]]
        return bg

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
        self.model.motionCompensate(self.lucasKanade.H)
        bgs_mask = self.model.update(frame)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(bgs_mask, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h < 60:
                continue
            background_res.append((x, y, w, h))
        for tracked in self.bgs_tracked_list:
            tracked.update_status = False
        if len(self.bgs_tracked_list) == 0:
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                self.bgs_tracked_list.append(tracker(x, y, w, h))
        else:
            for box in background_res:
                matched = False
                for tracked in self.bgs_tracked_list:
                    if calculate_iou(box, tracked.box) > 0.4:
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
                    continue
                if self.bgs_tracked_list[i].hitCounter > 3:
                    bgs_tracked_res.append(self.bgs_tracked_list[i].box)

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
                try:
                    rr = mse(old_bbox, new_bbox)
                except:
                    print(num)
                if rr > 200:
                    new_bgs_tracked_res.append(i)


        # res = self.fdCompensate(self.lucasKanade.H, self.imgGray)
        # mask = self.imgGrayPrev - res
        # thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]
        # kernel = np.ones((5, 5), np.uint8)
        # thresh = cv2.dilate(thresh, kernel, iterations=2)
        # thresh = cv2.erode(thresh, kernel, iterations=2)
        # thresh = thresh.astype(np.uint8)
        # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     if cv2.contourArea(contour) < 50:
        #         continue
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     framediff_res.append((x, y, w, h))
        # for tracked in self.tracked_list:
        #     tracked.update_status = False
        # if len(self.tracked_list) == 0:
        #     for contour in contours:
        #         if cv2.contourArea(contour) < 100:
        #             continue
        #         (x, y, w, h) = cv2.boundingRect(contour)
        #         self.tracked_list.append(tracker(x, y, w, h))
        # else:
        #     for box in framediff_res:
        #         matched = False
        #         for tracked in self.tracked_list:
        #             if calculate_iou(box, tracked.box) > 0.5:
        #                 tracked.update(box)
        #                 matched = True
        #                 tracked.hitCounter += 1
        #                 tracked.missCounter = 0
        #         if not matched:
        #             self.tracked_list.append(tracker(box[0], box[1], box[2], box[3]))
        #
        #     for i in range(len(self.tracked_list)):
        #         if not self.tracked_list[i].update_status:
        #             self.tracked_list[i].missCounter += 1
        #             continue
        #         if self.tracked_list[i].hitCounter > 3:
        #             fd_tracked_res.append(self.tracked_list[i].box)
        #     self.tracked_list = [obj for obj in self.tracked_list if obj.missCounter < 2]

        self.imgGrayPrev = self.imgGray.copy()
        # return bgs_tracked_res,fd_tracked_res
        return new_bgs_tracked_res, bgs_tracked_list_point



