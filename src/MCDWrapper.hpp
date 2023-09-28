
#ifndef	_MCDWRAPPER_H_
#define	_MCDWRAPPER_H_

/************************************************************************/
/* Basic Includes                                                       */
/************************************************************************/
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
#include "tracker.hpp"
using namespace std;
using namespace cv;

class MCDWrapper {

/************************************************************************/
/*  Internal Variables					                                */
/************************************************************************/
 public:

	int frm_cnt;

	Mat detect_img;

	/* Note that the variable names are legacy */
	KLTWrapper m_LucasKanade;
	Mat img;
	Mat imgTemp;
	Mat imgGray;
	Mat imgGrayPrev;

	Mat imgGaussLarge;
	Mat imgGaussSmall;
	Mat imgDOG;

	Mat debugCopy;
	Mat debugDisp;

	ProbModel BGModel;
    vector<char>Res;
	vector<Tracker> bgs_tracked_list;
/************************************************************************/
/*  Methods								                                */
/************************************************************************/
 public:

	 MCDWrapper();
	~MCDWrapper();

	void Init(const Mat & in_imgIpl);
	cv::Point2f compensate(cv::Point2f,double (*h)[9]);
	std::vector<cv::Rect> Run(const Mat & in_imgIpl,int frame_num);

};

#endif				//_MCDWRAPPER_H_
