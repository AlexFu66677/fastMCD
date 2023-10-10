
#include "KLTWrapper.hpp"
#include <opencv/cv.h>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

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
	for (int j = 0; j < 16; ++j) {
		for (int ii = 0; ii < 9; ++ii) {
			matH[j][ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
	}
}

int KLTWrapper::get_region(cv::Point2f point, int image_width, int image_height) {
    int x = point.x;
    int y = point.y;
    int region_width = image_width / 4;
    int region_height = image_height / 4;
    int row = y / region_height;
    int col = x / region_width;
    return row * 4 + col;
}

vector<vector<vector<Point2f>>> KLTWrapper::group_points(vector<Point2f> good0,vector<Point2f> good1){
    vector<vector<vector<Point2f>>> points_groups(2, vector<vector<Point2f>>(16));
    for(size_t i=0;i<good0.size();i++)
	{
		int region = int(get_region(good1[i], 1920, 1080));
		points_groups[0][region].push_back(good0[i]);
        points_groups[1][region].push_back(good1[i]);
    }
return points_groups;
}

void KLTWrapper::InitFeatures()
{
	double quality = 0.01;
	double min_distance = 10;
    prev_pts.clear();
	int ni = width;
	int nj = height;

	count = (int(ni / GRID_SIZE_W)-1) * (int(nj / GRID_SIZE_H)-1);

	int cnt = 0;
	for (int i = 0; i < count; ++i) {
			prev_pts.push_back(cv::Point2f(float(i)/(int(nj / GRID_SIZE_H)-1)* GRID_SIZE_W + GRID_SIZE_W / 2,i%(int(nj / GRID_SIZE_H)-1)* GRID_SIZE_H + GRID_SIZE_H / 2));
	}
}

void KLTWrapper::RunTrack(const Mat& imgGray, const Mat& prevGray)
{
	clock_t star,end;

	int nMatch[MAX_COUNT];
	std::vector<cv::Point2f> good0;
	good0.clear();
	std::vector<cv::Point2f> good1;
	good1.clear();
	if (count > 0) {
		star=clock();
		cv::UMat OCLprevGray, OCLimgGray;
		prevGray.copyTo(OCLprevGray); 
        imgGray.copyTo(OCLimgGray); 
		UMat umatNextPts, umatStatus, umatErr;
		cv::calcOpticalFlowPyrLK(OCLprevGray, OCLimgGray, prev_pts, umatNextPts, umatStatus, umatErr);
		// std::vector<cv::Point2f> Next_pts;
	    // std::vector<uchar> Status;
        // std::vector<float> Err;

		// umatNextPts.reshape(2, 1).copyTo(Next_pts);
        // umatStatus.reshape(1, 1).copyTo(Status);
        // umatErr.reshape(1, 1).copyTo(Err);

		umatNextPts.reshape(2, 1).copyTo(next_pts);
        umatStatus.reshape(1, 1).copyTo(status);
        umatErr.reshape(1, 1).copyTo(err);

		// cv::calcOpticalFlowPyrLK(prevGray, imgGray, prev_pts, next_pts, status, err, Size(5,5),3
		// ,cv::TermCriteria((TermCriteria::COUNT) | (TermCriteria::EPS), 20, (0.03)),0);
		end=clock();
	    double rt = end - star;
		std::cout<<"calcOpticalFlowPyrLK:"<<rt/CLOCKS_PER_SEC<<std::endl;
		for (int i = 0; i < prev_pts.size(); i++) {
			if (!status[i]||next_pts[i].x<0||next_pts[i].y<0||next_pts[i].x > width || next_pts[i].y>height) {
				continue;
			}
			good0.push_back(prev_pts[i]);
            good1.push_back(next_pts[i]);
			// good1.push_back(prev_pts[i]);
		}
	}

	if (count >= 10) {
		MakeHomoGraphy(good0, good1);
	} else {
		for (int j = 0; j < 16; ++j) {
		for (int ii = 0; ii < 9; ++ii) {
			matH[j][ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
		}
	}
    InitFeatures();

}

void KLTWrapper::GetHomography(double (*h)[9])
{
	memcpy(h, matH, sizeof(matH));
}

void KLTWrapper::MakeHomoGraphy(std::vector<cv::Point2f>&good0,std::vector<cv::Point2f> &good1)
{
	vector<vector<vector<Point2f>>> point_groups=group_points(good1,good0);
	double h[16][9];
	vector <Mat> _h(16, Mat(3, 3, CV_64F));

for(int j=0;j<16;j++){
	if(point_groups[0][j].size()<4)
	{
		for (int ii = 0; ii < 9; ++ii) {
			matH[j][ii]= ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
	}else{
		_h[j]=findHomography(point_groups[0][j], point_groups[1][j], RANSAC, 1);
		if(_h[j].empty()){
           for (int ii = 0; ii < 9; ++ii) {
			   matH[j][ii]= ii % 3 == ii / 3 ? 1.0f : 0.0f;
		    }
		}else{
		   for (int ii = 0; ii < 9; ++ii) {
			matH[j][ii]= _h[j].at<double>(ii);
		}
		}
	}
	}
}
