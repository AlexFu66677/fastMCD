#include	<iostream>
#include	<cstdlib>
#include	<cstring>
#include	<vector>
#include	<algorithm>
/************************************************************************/
/* Includes for the OpenCV                                              */
/************************************************************************/
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
// Inlcludes for this wrapper
#include "KLTWrapper.hpp"
#include "prob_model.hpp"

using namespace std;
using namespace cv;
class Tracker {
public:
    int bgs_hitCounter;
    int bgs_missCounter;
    int hitCounter;
    int missCounter;
    bool update_status;
    int x;
    int y;
    int w;
    int h;
    Rect box;
    Tracker(int x, int y, int w, int h) {
        bgs_hitCounter = 0;
        bgs_missCounter = 0;
        hitCounter = 0;
        missCounter = 0;
        update_status = true;
        this->x = x;
        this->y = y;
        this->w = w;
        this->h = h;
        box = Rect(x, y, w, h);
    }

    float calculate_intersection_area(Rect bbox1, Rect bbox2) {
        Rect intersection = bbox1 & bbox2;
        return intersection.area();
    }

    float calculate_union_area(Rect bbox1, Rect bbox2) {
        float iou = calculate_intersection_area(bbox1, bbox2);
        float union_area = bbox1.area() + bbox2.area() - iou;
        return union_area;
    }

    float calculate_iou(Rect bbox) {
        float intersection_area = calculate_intersection_area(bbox, box);
        float union_area = calculate_union_area(bbox, box);
        float iou = intersection_area / union_area;
        return iou;
    }

    void update(Rect bbox) {
        box = bbox;
        update_status = true;
    }
};