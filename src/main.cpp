
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
    clock_t startTime,endTime,perstar,perend,t1,t2;
	startTime = clock();
	while (bRun == true && frame_num <= frame_count) {
		perstar=clock();
		t1=clock();
        UMat frame;
        bool isSuccess = vid_capture.read(frame);
	    UMat raw_img_origin; 
		cv::cvtColor(frame, raw_img_origin, CV_RGB2GRAY);
        std::vector<cv::Rect>res;
        t2=clock();

		if (frame_num == 1) {
			mcdwrapper->Init(raw_img_origin);

		} else {
			res = mcdwrapper->Run(raw_img_origin);
		}
		for (size_t i = 0; i < res.size(); i++) {
            rectangle(frame, res[i].tl(), res[i].br(), Scalar(0, 255, 0), 2);
        }
		perend=clock();
		double pertime = perend - perstar;
		double t=t2-t1;
	    std::cout<<"alltime:"<<pertime/CLOCKS_PER_SEC<<endl;
        std::cout<<"read:"<<t/CLOCKS_PER_SEC<<endl;
		// char bufbuf[1000];
		// sprintf(bufbuf, "/home/fjl/code/fast/fastMCD-new/data/res4/frm%05d.png", frame_num);
		// imwrite(bufbuf,frame);
	    // imshow("OUT",frame);
		// cvWaitKey(10);
		++frame_num;

	}
	endTime = clock();
    double rt_motionComp = endTime - startTime;
	std::cout<<rt_motionComp/CLOCKS_PER_SEC<<endl;

	return 0;
}
