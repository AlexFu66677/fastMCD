
#include "MCDWrapper.hpp"
#include "ctime"
#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
#define MILLION 1000000
int main(int argc, char *argv[]) {
  cv::VideoCapture vid_capture(
      "/home/fjl/code/moving/fusebbox_SGM/python/data/car2.mp4");
  int fps = vid_capture.get(5);
  int frame_count = vid_capture.get(7);
  int frame_width = vid_capture.get(3);
  int frame_height = vid_capture.get(4);
  MCDWrapper *mcdwrapper = new MCDWrapper();
  int frame_num = 1;
  struct timespec start, finish;
  time_t start1, stop1;
  start1 = time(NULL);
  double elapsed = 0.0;
  clock_gettime(CLOCK_MONOTONIC, &start);
  while (frame_num <= frame_count) {

    UMat frame;
    bool isSuccess = vid_capture.read(frame);
    UMat raw_img_origin;
    cv::cvtColor(frame, raw_img_origin, CV_RGB2GRAY);
    std::vector<cv::Rect> res;

    if (frame_num == 1) {
      mcdwrapper->Init(raw_img_origin);

    } else {
      res = mcdwrapper->Run(raw_img_origin);
    }
    for (size_t i = 0; i < res.size(); i++) {
      rectangle(frame, res[i].tl(), res[i].br(), Scalar(0, 255, 0), 2);
    }

    // char bufbuf[1000];
    // sprintf(bufbuf, "/home/fjl/code/fast/fastMCD-new/data/res9/frm%05d.png",
    //         frame_num);
    // imwrite(bufbuf, frame);
    imshow("OUT", frame);
    cvWaitKey(1);

    ++frame_num;
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);
  std::cout << finish.tv_sec - start.tv_sec << endl;
  stop1 = time(NULL);
  std::cout << stop1 - start1 << endl;
  return 0;
}
