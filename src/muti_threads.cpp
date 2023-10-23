#include "KLTWrapper.hpp"
#include "ctime"
#include "model.hpp"
#include "params.hpp"
#include "tracker.hpp"
#include "json/json.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace cv;

std::mutex mtx;
std::queue<array<double, 144>> frameQueue;
std::queue<UMat> cur_frames;
std::queue<UMat> show_frames;
bool isFinished = false;
ProbModel BGModel;
void readFrame() {
  cv::VideoCapture cap(
      "/home/fjl/code/moving/fusebbox_SGM/python/data/car21.mp4");
  int frame_num = 1;
  KLTWrapper m_LucasKanade;
  UMat imgGrayPrev;
  while (true) {
    array<double, 144> hhh;
    double *hh = new double[16 * 9];
    UMat tempFrame;
    cap >> tempFrame;
    if (tempFrame.empty()) {
      break;
    }
    UMat imgGray;
    cv::cvtColor(tempFrame, imgGray, CV_RGB2GRAY);
    if (frame_num == 1) {
      std::lock_guard<std::mutex> lock(mtx);
      //   show_frames.push(tempFrame);
      m_LucasKanade.Init(tempFrame.cols, tempFrame.rows);
      Mat mat_imgGray;
      imgGray.copyTo(mat_imgGray);
      BGModel.init(mat_imgGray);
    } else {
      m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
      m_LucasKanade.GetHomography(hh);
      for (int i = 0; i < 144; i++) {
        hhh[i] = hh[i];
      }
      std::lock_guard<std::mutex> lock(mtx);
      frameQueue.push(hhh);
      cur_frames.push(imgGray);
      show_frames.push(tempFrame);
    }
    imgGrayPrev = imgGray.clone();
    delete hh;
    ++frame_num;
  }

  std::lock_guard<std::mutex> lock(mtx);
  isFinished = true;
}
cv::Point2f compensate(cv::Point2f point, double *h) {
  float newW = 0;
  float newX = 0;
  float newY = 0;
  int modelWidth = 1920;
  int modelHeight = 1080;
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

int main() {
  struct timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);
  std::thread t1(readFrame);
  t1.detach();
  Json::Value data;
  vector<Rect> background_res;
  vector<Tracker> bgs_tracked_list;
  std::vector<cv::Rect> unique_bgs_tracked_res;
  double *h = new double[16 * 9];
  UMat imggray;
  UMat cur_frame;
  UMat imggrayPrev;

  int num = 1;
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (frameQueue.empty()) {
        if (isFinished) {
          break;
        } else {
          continue;
        }
      }
      for (int i = 0; i < 16 * 9; i++) {
        h[i] = frameQueue.front()[i];
      }
      imggray = cur_frames.front();
      frameQueue.pop();
      cur_frames.pop();
    }
    if (imggrayPrev.empty()) {
      imggrayPrev = imggray.clone();
    }

    Mat Temp;
    imggray.copyTo(Temp);
    BGModel.motionCompensate(h, Temp);

    vector<Vec4i> hierarchy;
    vector<Rect> bgs_tracked_res;
    vector<vector<Point>> contours;
    vector<Rect> new_bgs_tracked_res;
    vector<Rect> bgs_tracked_list_point;

    UMat thresh;
    thresh = UMat::zeros(imggray.rows, imggray.cols, CV_8UC1);
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
          if (bgs_tracked_list[j].calculate_iou(Rect(
                  new_point.x, new_point.y, box.width, box.height)) > 0.4) {
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

      //更新tracker
      // list中box的missCounter,hitCounter,将结果加入bgs_tracked_res
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
      if (newX + box.width >= imggray.cols) {
        newX = imggray.cols - box.width - 1;
      }
      if (newY < 0) {
        newY = 0;
      }
      if (newY + box.height >= imggray.rows) {
        newY = imggray.rows - box.height - 1;
      }
      if (box.height >= imggray.rows || box.width >= imggray.cols) {
        continue;
      }
      bgs_tracked_list_point.push_back(
          Rect(int(newX), int(newY), box.width, box.height));

      UMat old_bbox_tmp = imggray(box);
      UMat new_bbox_tmp =
          imggrayPrev(Rect(int(newX), int(newY), box.width, box.height));
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
    imggrayPrev = imggray.clone();
    std::lock_guard<std::mutex> lock(mtx);
    cur_frame = show_frames.front();
    show_frames.pop();
    for (size_t i = 0; i < new_bgs_tracked_res.size(); i++) {
      rectangle(cur_frame, new_bgs_tracked_res[i].tl(),
                new_bgs_tracked_res[i].br(), Scalar(0, 255, 0), 2);
    }
    // char bufbuf[1000];
    // sprintf(bufbuf, "/home/fjl/code/fast/fastMCD-new/data/res10/frm%05d.png",
    //         num);
    // num++;
    // imwrite(bufbuf, cur_frame);
    imshow("OUT", cur_frame);
    cvWaitKey(1);
  }
  delete h;
  clock_gettime(CLOCK_MONOTONIC, &finish);
  std::cout << finish.tv_sec - start.tv_sec << endl;
  return 0;
}