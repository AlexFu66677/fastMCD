
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "MCDWrapper.hpp"
#include "ctime"
using namespace std;

int main(int argc, char *argv[])
{
    cv::VideoCapture vid_capture("/home/fjl/code/moving/fusebbox_SGM/python/data/car2.mp4");
    int fps = vid_capture.get(5);
    int frame_count = vid_capture.get(7);
	int frame_width=vid_capture.get(3);	
    int frame_height=vid_capture.get(4);
	MCDWrapper *mcdwrapper = new MCDWrapper();
	int frame_num = 1;
	bool bRun = true;
	const char window_name[] = "OUTPUT";
	cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
	Mat mask=Mat::zeros(frame_height, frame_width, CV_8UC1);
	Mat thresh=Mat::zeros(frame_height, frame_width, CV_8UC1);
    clock_t startTime,endTime,perstar,perend;
	startTime = clock();
	while (bRun == true && frame_num <= frame_count) {
        Mat frame;
        bool isSuccess = vid_capture.read(frame);
	    Mat frame_copy_origin =frame.clone();
	    Mat raw_img_origin =frame.clone();
        std::vector<cv::Rect>res;
		if (frame_num == 1) {
			mcdwrapper->Init(raw_img_origin);

		} else {
			perstar=clock();
			res = mcdwrapper->Run(raw_img_origin,frame_num);
			perend=clock();
			double pertime = perend - perstar;
	        std::cout<<pertime/CLOCKS_PER_SEC<<endl;
		}
		// for (size_t i = 0; i < mcdwrapper->background_res.size(); i++) {
        //     rectangle(frame_copy_origin, mcdwrapper->background_res[i].tl(), mcdwrapper->background_res[i].br(), Scalar(255, 0, 0), 2);
        // }
		// for (size_t i = 0; i < mcdwrapper->unique_bgs_tracked_res.size(); i++) {
        //     rectangle(frame_copy_origin, mcdwrapper->unique_bgs_tracked_res[i].tl(), mcdwrapper->unique_bgs_tracked_res[i].br(), Scalar(0, 0, 255), 2);
        // }
		for (size_t i = 0; i < res.size(); i++) {
            rectangle(frame_copy_origin, res[i].tl(), res[i].br(), Scalar(0, 255, 0), 2);
        }
		// char bufbuf[1000];
		// sprintf(bufbuf, "/home/fjl/code/fastMCD/src/res10/frm%05d.png", frame_num);
		// imwrite(bufbuf,frame_copy_origin);
	    // imshow("OUT",frame_copy_origin);
		// cvWaitKey(10);
		++frame_num;

	}
	endTime = clock();
    double rt_motionComp = endTime - startTime;
	std::cout<<rt_motionComp/CLOCKS_PER_SEC<<endl;

	return 0;
}
