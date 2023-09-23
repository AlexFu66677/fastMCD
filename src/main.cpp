
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "MCDWrapper.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    cv::VideoCapture vid_capture("/home/fjl/code/fastMCD/data/woman.mp4");
    int fps = vid_capture.get(5);
    int frame_count = vid_capture.get(7);
	int frame_width=vid_capture.get(3);	
    int frame_height=vid_capture.get(4);
	MCDWrapper *mcdwrapper = new MCDWrapper();
	IplImage *frame = 0, *frame_copy = 0, *vil_conv = 0, *raw_img = 0, *model_copy = 0, *model_img = 0;
	IplImage *edge = 0;
	int frame_num = 1;
	bool bRun = true;
	const char window_name[] = "OUTPUT";
	cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
	Mat mask=Mat::zeros(frame_height, frame_width, CV_8UC1);

	while (bRun == true && frame_num <= frame_count) {	// the main loop
	int t=0;
	int tt=0;

        Mat frame;
        bool isSuccess = vid_capture.read(frame);
	    Mat _frame =frame.clone();
        IplImage imgTmp = _frame;
		IplImage *IplBuffer=cvCloneImage(&imgTmp);

	    Mat frame_copy_origin =frame.clone();
	    Mat raw_img_origin =frame.clone();

		if (frame_num == 1) {
			mcdwrapper->Init(raw_img_origin);

		} else {
			mask = mcdwrapper->Run(raw_img_origin,frame_num);

		}

        if (!frame_copy) {
			frame_copy = cvCreateImage(cvSize(IplBuffer->width, IplBuffer->height), IPL_DEPTH_8U, IplBuffer->nChannels);
			raw_img = cvCreateImage(cvSize(IplBuffer->width, IplBuffer->height), IPL_DEPTH_8U, IplBuffer->nChannels);
		}
		if (IplBuffer->origin == IPL_ORIGIN_TL) {
			cvCopy(IplBuffer, frame_copy, 0);
			cvCopy(IplBuffer, raw_img, 0);
		} else {
			cvFlip(IplBuffer, frame_copy, 0);
			cvFlip(IplBuffer, raw_img, 0);
		}


        int nCount = cv::countNonZero(mask);
		char bufbuf[1000];
		sprintf(bufbuf, "/home/fjl/code/fastMCD/src/res2/frm%05d.png", frame_num);
		imwrite(bufbuf,mask);
		++frame_num;

	}
	// cvReleaseImage(&raw_img);
	// cvReleaseImage(&frame_copy);
	// cvReleaseVideoWriter(&pVideoOut);

	return 0;
}
