#pragma once

#include "ctime"
#include <opencv/cv.h>
#define GRID_SIZE_W (16)
#define GRID_SIZE_H (12)
using namespace cv;

typedef unsigned char BYTE;

class KLTWrapper {
private:
  Mat image;
  int win_size;
  int MAX_COUNT;
  int height;
  int width;
  std::vector<cv::Point2f> prev_pts;
  std::vector<cv::Point2f> next_pts;
  std::vector<uchar> status;
  std::vector<float> err;
  int count;
  int flags;

  double matH[16][9];

private:
  void MakeHomoGraphy(std::vector<cv::Point2f> &prev_pts,
                      std::vector<cv::Point2f> &next_pts);

public:
  KLTWrapper(void);
  ~KLTWrapper(void);

  void Init(int image_width, int image_height);
  void InitFeatures();
  void RunTrack(const UMat &imgGray,
                const UMat &prevGray); // with MakeHomography
  void GetHomography(double *h);
  int get_region(cv::Point2f point);
  std::vector<std::vector<std::vector<cv::Point2f>>>
  group_points(std::vector<cv::Point2f> good0, std::vector<cv::Point2f> good1);
};
