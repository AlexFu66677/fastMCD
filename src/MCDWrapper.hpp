
#ifndef _MCDWRAPPER_H_
#define _MCDWRAPPER_H_
#include "KLTWrapper.hpp"
#include "prob_model.hpp"
#include "tracker.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

class MCDWrapper {
public:
  int frm_cnt;

  KLTWrapper m_LucasKanade;
  UMat imgGrayPrev;
  ProbModel BGModel;
  vector<Tracker> bgs_tracked_list;
  vector<Rect> background_res;
  std::vector<cv::Rect> unique_bgs_tracked_res;
  double *h;

public:
  MCDWrapper();
  ~MCDWrapper();

  void Init(const UMat &imgGary);
  cv::Point2f compensate(cv::Point2f, double *h);
  std::vector<cv::Rect> Run(const UMat &imgGary);
};

#endif //_MCDWRAPPER_H_
