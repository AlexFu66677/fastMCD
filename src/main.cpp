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
	// char test_file[100];
	// char abs_file[1000];
	// if (argc != 3) {
	// 	printf("Usage: Binary [VideoFileFullPathWithExt] [bool DumpOutputImgs (1/0)]\n");
	// 	return -1;
	// }
	// /************************************************************************/
	// /*  Get MP4 Name exluding all paths                                     */
	// /************************************************************************/
	// strncpy(abs_file, (char *)(argv[1]), 1000);
	// //first run to find '/' token
	// int last_token_pos = 0;
	// for (int i = 0; abs_file[i] != '\0'; ++i) {
	// 	if (abs_file[i] == '/' || abs_file[i] == '\\')
	// 		last_token_pos = i;
	// }
	// int tmpidx = 0;
	// for (int i = last_token_pos; abs_file[i] != '.'; ++i) {
	// 	test_file[tmpidx++] = abs_file[i];
	// }
	// test_file[tmpidx] = '\0';
	// /************************************************************************/
	// /*  Initialize Variables                                                */
	// /************************************************************************/
	// // wrapper class for mcd
	// MCDWrapper *mcdwrapper = new MCDWrapper();
	// const char window_name[] = "OUTPUT";
	// IplImage *frame = 0, *frame_copy = 0, *vil_conv = 0, *raw_img = 0, *model_copy = 0, *model_img = 0;
	// IplImage *edge = 0;
	// string infile_name;
	// infile_name.append(abs_file);
	// system("mkdir ./results");
	// string mp4file_name("./results/");
	// mp4file_name.append(test_file);
	// mp4file_name.append("_result");
	// mp4file_name.append(".mp4");
	// CvCapture *pInVideo = cvCaptureFromAVI(infile_name.data());
	// int dTotalFrameNum = (int)cvGetCaptureProperty(pInVideo, CV_CAP_PROP_FRAME_COUNT);
	// cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
	// cvSetCaptureProperty(pInVideo, CV_CAP_PROP_POS_FRAMES, 0);
	// int frame_num = 1;
	// bool bRun = true;
	while (bRun == true && frame_num <= frame_count) {	// the main loop

		// Grab Frame
        Mat frame;
        bool isSuccess = vid_capture.read(frame);
	    Mat _frame =frame.clone();
        IplImage imgTmp = _frame;
		IplImage *IplBuffer=cvCloneImage(&imgTmp);

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

		if (frame_num == 1) {

			// Init the wrapper for first frame
			mcdwrapper->Init(raw_img);

		} else {

			// Run detection
			mcdwrapper->Run();

		}

		// Display detection results as overlay
		for (int j = 0; j < frame_copy->height; ++j) {
			for (int i = 0; i < frame_copy->width; ++i) {

				float draw_orig = 0.5;

				BYTE *pMaskImg = (BYTE *) (mcdwrapper->detect_img->imageData);
				int widthstepMsk = mcdwrapper->detect_img->widthStep;

				int mask_data = pMaskImg[i + j * widthstepMsk];

				((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 2] = draw_orig * ((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 2];
				((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 1] = draw_orig * ((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 1];
				((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 0] = draw_orig * ((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 0];

				if (frame_num > 1) {
					((BYTE *) (frame_copy->imageData))[i * 3 + j * frame_copy->widthStep + 2] += mask_data > 0 ? 255 * (1.0 - draw_orig) : 0;
				}

			}
		}

		// Print some frame numbers as well
		char buf[100];
		sprintf(buf, "%d", frame_num);
		CvFont font;
		double hScale = 0.5;
		double vScale = 0.5;
		int lineWidth = 2;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, hScale, vScale, 0, lineWidth);
		cvPutText(frame_copy, buf, cvPoint(10, 20), &font, cvScalar(255, 255, 0));

		cvShowImage(window_name, frame_copy);


		switch (cvWaitKey(1)) {
		case 'q':	// press q to quit
			bRun = false;
			break;
		default:
			break;
		}
		++frame_num;

	}
	// cvReleaseImage(&raw_img);
	// cvReleaseImage(&frame_copy);
	// cvReleaseVideoWriter(&pVideoOut);

	return 0;
}
