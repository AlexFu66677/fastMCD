// Copyright (c) 2016 Kwang Moo Yi.
// All rights reserved.

// This  software  is  strictly   for  non-commercial  use  only.  For
// commercial       use,       please        contact       me       at
// kwang.m<dot>yi<AT>gmail<dot>com.   Also,  when  used  for  academic
// purposes, please cite  the paper "Detection of  Moving Objects with
// Non-stationary Cameras in 5.8ms:  Bringing Motion Detection to Your
// Mobile Device,"  Yi et  al, CVPRW 2013  Redistribution and  use for
// non-commercial purposes  in source  and binary forms  are permitted
// provided that  the above  copyright notice  and this  paragraph are
// duplicated  in   all  such   forms  and  that   any  documentation,
// advertising  materials,   and  other  materials  related   to  such
// distribution and use acknowledge that the software was developed by
// the  Perception and  Intelligence Lab,  Seoul National  University.
// The name of the Perception  and Intelligence Lab and Seoul National
// University may not  be used to endorse or  promote products derived
// from this software without specific prior written permission.  THIS
// SOFTWARE IS PROVIDED ``AS IS''  AND WITHOUT ANY WARRANTIES.  USE AT
// YOUR OWN RISK!

#pragma once

#include <opencv/cv.h>
#define GRID_SIZE_W (64)
#define GRID_SIZE_H (48)
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

	// For Homography Matrix
	double matH[16][9];

 private:
	void SwapData(const Mat& imgGray);
	void MakeHomoGraphy(std::vector<cv::Point2f>&prev_pts,std::vector<cv::Point2f>&next_pts);

 public:
	 KLTWrapper(void);
	~KLTWrapper(void);

	void Init(const Mat& imgGray);
	void InitFeatures();
	void RunTrack(const Mat& imgGray, const Mat& prevGray);	// with MakeHomography
	void GetHomography(double (*h)[9]);
	int get_region(cv::Point2f point, int image_width, int image_height);
	std::vector<std::vector<std::vector<cv::Point2f>>> group_points(std::vector<cv::Point2f> good0,std::vector<cv::Point2f> good1);
};
