#ifndef _MCDWRAPPER_CPP_
#define _MCDWRAPPER_CPP_

#include "MCDWrapper.hpp"
#include "params.hpp"
#include <cstring>
#include <ctime>

MCDWrapper::MCDWrapper() {}

MCDWrapper::~MCDWrapper() { delete h; }

void MCDWrapper::Init(const UMat &imgGray) {

  frm_cnt = 0;
  Mat mat_imgGray;
  imgGray.copyTo(mat_imgGray);
  m_LucasKanade.Init(imgGray.cols, imgGray.rows);
  BGModel.init(mat_imgGray);
  imgGrayPrev = imgGray.clone();
  h = new double[16 * 9];
}
cv::Point2f MCDWrapper::compensate(cv::Point2f point, double *h) {
  float newW = 0;
  float newX = 0;
  float newY = 0;
  int modelWidth = imgGrayPrev.cols;
  int modelHeight = imgGrayPrev.rows;
  int X = point.x;
  int Y = point.y;

  int h_idxi = X / 480;
  int h_idxj = Y / 270;
  int h_id = h_idxi + h_idxj * 4;

  newW = h[h_id * 9 + 6] * X + h[h_id * 9 + 7] * Y + h[h_id * 9 + 8];
  newX = (h[h_id * 9 + 0] * X + h[h_id * 9 + 1] * Y + h[h_id * 9 + 2]) / newW;
  newY = (h[h_id * 9 + 3] * X + h[h_id * 9 + 4] * Y + h[h_id * 9 + 5]) / newW;
  return cv::Point2f(newX, newY);
}

std::vector<cv::Rect> MCDWrapper::Run(const UMat &imgGray) {
  /*************************
    计算KLT 单应矩阵H 模型补偿
  **************************/
  m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
  m_LucasKanade.GetHomography(h);
  Mat Temp;
  imgGray.copyTo(Temp);
  BGModel.motionCompensate(h, Temp);
  Mat H(3, 3, CV_32F);
  memcpy(H.data, h, sizeof(float) * 9);

  /*************************
      模型更新 前景提取
  **************************/
  Mat Temp1;
  imgGray.copyTo(Temp1);
  BGModel.update(Temp1);
  /*************************
      形态学处理并绘制初始框
  **************************/
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  vector<Rect> bgs_tracked_res;
  vector<Rect> bgs_tracked_list_point;
  vector<Rect> new_bgs_tracked_res;

  UMat thresh;
  thresh = UMat::zeros(imgGray.rows, imgGray.cols, CV_8UC1);
  Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
  dilate(BGModel.mask, thresh, kernel, Point(-1, -1), 1);
  erode(thresh, thresh, kernel, Point(-1, -1), 1);
  background_res.clear();
  findContours(thresh.clone(), contours, hierarchy, RETR_EXTERNAL,
               CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    if (contourArea(contours[i]) < 40)
      continue;
    Rect rect = boundingRect(contours[i]);
    background_res.push_back(rect);
  }
  /************************************
      tracker追踪 通过运动连续性对运动目标框进行筛选
  *************************************/
  //更新所有trackerbox状态
  for (size_t i = 0; i < bgs_tracked_list.size(); i++) {
    bgs_tracked_list[i].update_status = false;
  }
  //初始化tracker list
  if (bgs_tracked_list.empty()) {
    for (size_t i = 0; i < contours.size(); i++) {
      if (contourArea(contours[i]) < 40)
        continue;
      Rect bbox = boundingRect(contours[i]);
      Tracker tracker(bbox.x, bbox.y - 2, bbox.width, bbox.height + 2);
      bgs_tracked_list.push_back(tracker);
    }
  } else {
    //更新tracker list中匹配框的坐标,将未匹配的加入list
    for (size_t i = 0; i < background_res.size(); i++) {
      Rect box = background_res[i];
      cv::Point2f new_point = compensate(cv::Point2f(box.x, box.y), h);
      bool matched = false;
      for (size_t j = 0; j < bgs_tracked_list.size(); j++) {
        if (bgs_tracked_list[j].calculate_iou(
                Rect(new_point.x, new_point.y, box.width, box.height)) > 0.4) {
          bgs_tracked_list[j].update(box);
          matched = true;
          bgs_tracked_list[j].hitCounter++;
          bgs_tracked_list[j].missCounter = 0;
        }
      }
      if (!matched) {
        Tracker tracker(box.x, box.y, box.width, box.height);
        bgs_tracked_list.push_back(tracker);
      }
    }

    //删除tracker list中missCounter >= 3的box
    vector<Tracker>::iterator iter;
    for (iter = bgs_tracked_list.begin(); iter != bgs_tracked_list.end();) {
      if (iter->missCounter >= 3) {
        iter = bgs_tracked_list.erase(iter);
      } else {
        iter++;
      }
    }

    //更新tracker list中box的missCounter,hitCounter,将结果加入bgs_tracked_res
    for (size_t i = 0; i < bgs_tracked_list.size(); i++) {
      if (!bgs_tracked_list[i].update_status) {
        bgs_tracked_list[i].missCounter++;
        continue;
      }
      if (bgs_tracked_list[i].hitCounter > 2) {
        bgs_tracked_res.push_back(bgs_tracked_list[i].box);
      }
    }
  }

  //计算unique删除重复的box
  unique_bgs_tracked_res.clear();
  for (size_t i = 0; i < bgs_tracked_res.size(); i++) {
    bool is_unique = true;
    for (size_t j = 0; j < unique_bgs_tracked_res.size(); j++) {
      if (bgs_tracked_res[i] == unique_bgs_tracked_res[j]) {
        is_unique = false;
        break;
      }
    }
    if (is_unique) {
      unique_bgs_tracked_res.push_back(bgs_tracked_res[i]);
    }
  }
  /************************************
      计算均方误差追踪 通过相似性对运动目标框进行筛选
  *************************************/
  new_bgs_tracked_res.clear();
  for (size_t i = 0; i < unique_bgs_tracked_res.size(); i++) {
    Rect box = unique_bgs_tracked_res[i];
    cv::Point2f new_point = compensate(cv::Point2f(box.x, box.y), h);
    float newX = new_point.x;
    float newY = new_point.y;
    if (newX < 0) {
      newX = 0;
    }
    if (newX + box.width >= imgGray.cols) {
      newX = imgGray.cols - box.width - 1;
    }
    if (newY < 0) {
      newY = 0;
    }
    if (newY + box.height >= imgGray.rows) {
      newY = imgGray.rows - box.height - 1;
    }
    if (box.height >= imgGray.rows || box.width >= imgGray.cols) {
      continue;
    }
    bgs_tracked_list_point.push_back(
        Rect(int(newX), int(newY), box.width, box.height));

    UMat old_bbox_tmp = imgGray(box);
    UMat new_bbox_tmp =
        imgGrayPrev(Rect(int(newX), int(newY), box.width, box.height));
    Mat old_bbox;
    Mat new_bbox;
    old_bbox_tmp.copyTo(old_bbox);
    new_bbox_tmp.copyTo(new_bbox);

    if (old_bbox.rows != new_bbox.rows || old_bbox.cols != new_bbox.cols) {
      continue;
    } else {
      double mse = 0;
      for (int i = 0; i < old_bbox.cols; i++) {
        for (int j = 0; j < old_bbox.rows; j++) {
          uchar a = old_bbox.at<uchar>(j, i);
          uchar b = new_bbox.at<uchar>(j, i);
          double diff = a - b;
          double rr = diff * diff;
          mse += rr;
        }
      }
      mse = mse / (old_bbox.cols * old_bbox.rows);
      if (mse > 250.0) {
        new_bgs_tracked_res.push_back(box);
      }
    }
  }
  imgGrayPrev = imgGray.clone();
  return new_bgs_tracked_res;
}

#endif // _MCDWRAPPER_CPP_