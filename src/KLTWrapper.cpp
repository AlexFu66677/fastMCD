
#include "KLTWrapper.hpp"
#include <opencv/cv.h>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

KLTWrapper::KLTWrapper(void)
{
	win_size = 10;
}

KLTWrapper::~KLTWrapper(void)
{
}

void KLTWrapper::Init(const Mat& imgGray)
{
	width = imgGray.cols;
	height = imgGray.rows;
	flags=0;
	for (int i = 0; i < 9; i++)
		matH[i] = i / 3 == i % 3 ? 1 : 0;
}

void KLTWrapper::InitFeatures()
{
	/* automatic initialization */
	double quality = 0.01;
	double min_distance = 10;
    prev_pts.clear();
	int ni = width;
	int nj = height;

	count = (int(ni / GRID_SIZE_W)) * (int(nj / GRID_SIZE_H));

	int cnt = 0;
	for (int i = 0; i < ni / GRID_SIZE_W - 1; ++i) {
		for (int j = 0; j < nj / GRID_SIZE_H - 1; ++j) {
			prev_pts.push_back(cv::Point2f(i * GRID_SIZE_W + GRID_SIZE_W / 2,j * GRID_SIZE_H + GRID_SIZE_H / 2));
		}
	}
}

void KLTWrapper::RunTrack(const Mat& imgGray, const Mat& prevGray)
{
	int nMatch[MAX_COUNT];
	std::vector<cv::Point2f> good0;
	good0.clear();
	std::vector<cv::Point2f> good1;
	good1.clear();
	if (count > 0) {
		cv::calcOpticalFlowPyrLK(prevGray, imgGray, prev_pts, next_pts, status, err, Size(10,10),3
		,cv::TermCriteria((TermCriteria::COUNT) | (TermCriteria::EPS), 20, (0.03)),0);

		for (int i = 0; i < prev_pts.size(); i++) {
			if (!status[i]||next_pts[i].x<0||next_pts[i].y<0||next_pts[i].x > width || next_pts[i].y>height) {
				continue;
			}
			good0.push_back(prev_pts[i]);
            good1.push_back(next_pts[i]);
		}
	}

	if (count >= 10) {
		MakeHomoGraphy(good0, good1);
	} else {
		for (int ii = 0; ii < 9; ++ii) {
			matH[ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
	}
    InitFeatures();

}

void KLTWrapper::GetHomography(double *pmatH)
{
	memcpy(pmatH, matH, sizeof(matH));
}

void KLTWrapper::MakeHomoGraphy(std::vector<cv::Point2f>&good0,std::vector<cv::Point2f> &good1)
{
	double h[9];
	Mat _h = Mat(3, 3, CV_64F);
	_h=findHomography(good1, good0, RANSAC, 1);

	for (int i = 0; i < 9; i++) {
		matH[i] = _h.at<double>(i);
	}
}
