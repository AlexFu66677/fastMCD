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

#ifndef	_MCDWRAPPER_CPP_
#define	_MCDWRAPPER_CPP_

#include <ctime>
#include <cstring>
#include "MCDWrapper.hpp"
#include "params.hpp"

#if defined _WIN32 || defined _WIN64
int gettimeofday(struct timeval *tp, int *tz)
{
	LARGE_INTEGER tickNow;
	static LARGE_INTEGER tickFrequency;
	static BOOL tickFrequencySet = FALSE;
	if (tickFrequencySet == FALSE) {
		QueryPerformanceFrequency(&tickFrequency);
		tickFrequencySet = TRUE;
	}
	QueryPerformanceCounter(&tickNow);
	tp->tv_sec = (long)(tickNow.QuadPart / tickFrequency.QuadPart);
	tp->tv_usec = (long)(((tickNow.QuadPart % tickFrequency.QuadPart) * 1000000L) / tickFrequency.QuadPart);

	return 0;
}
#else
#include <sys/time.h>
#endif

MCDWrapper::MCDWrapper()
{
}

MCDWrapper::~MCDWrapper()
{
}

void
 MCDWrapper::Init(const Mat& in_img)
{

	frm_cnt = 0;
	img = in_img;
	// Allocate
	imgTemp= img.clone();
	imgGray = Mat::zeros(img.rows, img.cols, CV_8UC1);
	detect_img = Mat::zeros(img.rows,img.cols,  CV_8UC1);
    cv::cvtColor(imgTemp, imgGray, CV_RGB2GRAY);
	m_LucasKanade.Init(imgGray);
	BGModel.init(imgGray);
	imgGrayPrev=imgGray.clone();
}
cv::Point2f MCDWrapper::compensate(cv::Point2f point,double (*h)[9])
{
		       float newW = 0;
		       float newX = 0;
		       float newY = 0;
			   int modelWidth=imgGray.cols;
			   int modelHeight=imgGray.rows;
			   int X=point.x;
			   int Y=point.y;
				// transformed coordinates with h
		       if(X<modelWidth/4&&Y<modelHeight/4)
				{
		        newW = h[0][6] * X + h[0][7] * Y + h[0][8];
				newX = (h[0][0] * X + h[0][1] * Y + h[0][2]) / newW;
				newY = (h[0][3] * X + h[0][4] * Y + h[0][5]) / newW;
				}
				else if(X<modelWidth/2&&X>=modelWidth/4&&Y<modelHeight/4)
				{
				newW = h[1][6] * X + h[1][7] * Y + h[1][8];
				newX = (h[1][0] * X + h[1][1] * Y + h[1][2]) / newW;
				newY = (h[1][3] * X + h[1][4] * Y + h[1][5]) / newW;
				}
				else if(X<modelWidth/4*3&&X>=modelWidth/2&&Y<modelHeight/4)
				{
				newW = h[2][6] * X + h[2][7] * Y + h[2][8];
				newX = (h[2][0] * X + h[2][1] * Y + h[2][2]) / newW;
				newY = (h[2][3] * X + h[2][4] * Y + h[2][5]) / newW;
				}
				else if(X<modelWidth&&X>=modelWidth/4*3&&Y<modelHeight/4)
				{
				newW = h[3][6] * X + h[3][7] * Y + h[3][8];
				newX = (h[3][0] * X + h[3][1] * Y + h[3][2]) / newW;
				newY = (h[3][3] * X + h[3][4] * Y + h[3][5]) / newW;
				}

				else if(X<modelWidth/4&&Y<modelHeight/2&&Y>=modelHeight/4)
				{
				newW = h[4][6] * X + h[4][7] * Y + h[4][8];
				newX = (h[4][0] * X + h[4][1] * Y + h[4][2]) / newW;
				newY = (h[4][3] * X + h[4][4] * Y + h[4][5]) / newW;
				}
				else if(X<modelWidth/2&&X>=modelWidth/4&&Y<modelHeight/2&&Y>=modelHeight/4)
				{
				newW = h[5][6] * X + h[5][7] * Y + h[5][8];
				newX = (h[5][0] * X + h[5][1] * Y + h[5][2]) / newW;
				newY = (h[5][3] * X + h[5][4] * Y + h[5][5]) / newW;
				}
				else if(X<modelWidth/4*3&&X>=modelWidth/2&&Y<modelHeight/2&&Y>=modelHeight/4)
				{
				newW = h[6][6] * X + h[6][7] * Y + h[6][8];
				newX = (h[6][0] * X + h[6][1] * Y + h[6][2]) / newW;
				newY = (h[6][3] * X + h[6][4] * Y + h[6][5]) / newW;
				}
				else if(X<modelWidth&&X>=modelWidth/4*3&&Y<modelHeight/2&&Y>=modelHeight/4)
				{
				newW = h[7][6] * X + h[7][7] * Y + h[7][8];
				newX = (h[7][0] * X + h[7][1] * Y + h[7][2]) / newW;
				newY = (h[7][3] * X + h[7][4] * Y + h[7][5]) / newW;
				}
				else if(X<modelWidth/4&&Y<modelHeight/4*3&&Y>=modelHeight/2)
				{
				newW = h[8][6] * X + h[8][7] * Y + h[8][8];
				newX = (h[8][0] * X + h[8][1] * Y + h[8][2]) / newW;
				newY = (h[8][3] * X + h[8][4] * Y + h[8][5]) / newW;
				}
				else if(X<modelWidth/2&&X>=modelWidth/4&&Y<modelHeight/4*3&&Y>=modelHeight/2)
				{
				newW = h[9][6] * X + h[9][7] * Y + h[9][8];
				newX = (h[9][0] * X + h[9][1] * Y + h[9][2]) / newW;
				newY = (h[9][3] * X + h[9][4] * Y + h[9][5]) / newW;
				}
				else if(X<modelWidth/4*3&&X>=modelWidth/2&&Y<modelHeight/4*3&&Y>=modelHeight/2)
				{
				newW = h[10][6] * X + h[10][7] * Y + h[10][8];
				newX = (h[10][0] * X + h[10][1] * Y + h[10][2]) / newW;
				newY = (h[10][3] * X + h[10][4] * Y + h[10][5]) / newW;
				}
				else if(X<modelWidth&&X>=modelWidth/4*3&&Y<modelHeight/4*3&&Y>=modelHeight/2)
				{
				newW = h[11][6] * X + h[11][7] * Y + h[11][8];
				newX = (h[11][0] * X + h[11][1] * Y + h[11][2]) / newW;
				newY = (h[11][3] * X + h[11][4] * Y + h[11][5]) / newW;
				}
				else if(X<modelWidth/4&&Y<modelHeight&&Y>=modelHeight/4*3)
				{
				newW = h[12][6] * X + h[12][7] * Y + h[12][8];
				newX = (h[12][0] * X + h[12][1] * Y + h[12][2]) / newW;
				newY = (h[12][3] * X + h[12][4] * Y + h[12][5]) / newW;
				}
				else if(X<modelWidth/2&&X>=modelWidth/4&&Y<modelHeight&&Y>=modelHeight/4*3)
				{
				newW = h[13][6] * X + h[13][7] * Y + h[13][8];
				newX = (h[13][0] * X + h[13][1] * Y + h[13][2]) / newW;
				newY = (h[13][3] * X + h[13][4] * Y + h[13][5]) / newW;
				}
				else if(X<modelWidth/4*3&&X>=modelWidth/2&&Y<modelHeight&&Y>=modelHeight/4*3)
				{
				newW = h[14][6] * X + h[14][7] * Y + h[14][8];
				newX = (h[14][0] * X + h[14][1] * Y + h[14][2]) / newW;
				newY = (h[14][3] * X + h[14][4] * Y + h[14][5]) / newW;
				}
				else if(X<modelWidth&&X>=modelWidth/4*3&&Y<modelHeight&&Y>=modelHeight/4*3)
				{
				newW = h[15][6] * X + h[15][7] * Y + h[15][8];
				newX = (h[15][0] * X + h[15][1] * Y + h[15][2]) / newW;
				newY = (h[15][3] * X + h[15][4] * Y + h[15][5]) / newW;
				}
				return cv::Point2f(newX,newY);
}
std::vector<cv::Rect> MCDWrapper::Run(const Mat & input_img,int frame_num)
{

	frm_cnt++;

	timeval tic, toc, tic_total, toc_total;
	float rt_motionComp;	// motion Compensation time
	float rt_modelUpdate;	// model update time
	float rt_total;		// Background Subtraction time
	cv::cvtColor(input_img, imgGray, CV_RGB2GRAY);


	//--TIME START
	gettimeofday(&tic, NULL);
	// Calculate Backward homography
	// Get H
	double h[16][9];
	m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
	m_LucasKanade.GetHomography(h);
	BGModel.motionCompensate(h,frame_num);
	Mat H(3, 3, CV_32F);
	//mat.create(3, 3, CV_32F);
    memcpy(H.data,h, sizeof(float) * 9);
    // int modelWidth=88;
    // int modelHeight=72;
	// for(int i=0;i<modelWidth*modelHeight;i++)
	// {
	// 	m_Mean[i] = BGModel.m_Mean[0][i];
	// 	m_Var[i] = BGModel.m_Var[0][i];
	// 	m_Age[i] = BGModel.m_Age[0][i];
	// 	m_Mean_Temp[i] = BGModel.m_Mean_Temp[0][i];
	// 	m_Var_Temp[i] = BGModel.m_Var_Temp[0][i];
	// 	m_Age_Temp[i] = BGModel.m_Age_Temp[0][i];
	// }
	//--TIME END
	gettimeofday(&toc, NULL);
	rt_motionComp = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//--TIME START
	gettimeofday(&tic, NULL);
	// Update BG Model and Detect
	Mat Temp= imgGray.clone();
	BGModel.update(Temp);

	Mat thresh=Mat::zeros(input_img.rows, input_img.cols, CV_8UC1);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(BGModel.mask, thresh, kernel, Point(-1, -1), 1);
    erode(thresh, thresh, kernel, Point(-1, -1), 1);
	vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
	vector<Rect> background_res;
	vector<Rect> bgs_tracked_res;
	vector<Rect>bgs_tracked_list_point;
    vector<Rect>new_bgs_tracked_res;
    findContours(thresh.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) < 40)
            continue;
        Rect rect = boundingRect(contours[i]);
		background_res.push_back(rect);
    }
	for (size_t i = 0; i < bgs_tracked_list.size(); i++) {
        bgs_tracked_list[i].update_status=false;
	}
    if (bgs_tracked_list.empty()) {
        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) < 40)
                continue;
            Rect bbox = boundingRect(contours[i]);
            Tracker tracker(bbox.x, bbox.y - 2, bbox.width, bbox.height + 2);
            bgs_tracked_list.push_back(tracker);
        }
    }
	
	else{
		for (size_t i = 0; i < background_res.size(); i++) {	
			Rect box = background_res[i];
			cv::Point2f new_point=compensate(cv::Point2f(box.x,box.y),h);

			// float newW = h[6] * box.x + h[7] * box.y + h[8];
			// float newX = (h[0] * box.x + h[1] * box.y + h[2]) / newW;
			// float newY = (h[3] * box.x + h[4] * box.y + h[5]) / newW;
			// Point2f new_point(newX, newY);
			bool matched = false;
			for (size_t j = 0; j < bgs_tracked_list.size(); j++) {
                if (bgs_tracked_list[j].calculate_iou(Rect(new_point.x, new_point.y, box.width, box.height)) > 0.4) {
                    bgs_tracked_list[j].update(box);
                    matched = true;
                    bgs_tracked_list[j].hitCounter++;
                    bgs_tracked_list[j].missCounter = 0;
                }
            }
            if (!matched) {
                Tracker tracker(box.x, box.y, box.width, box.height);
                bgs_tracked_list.push_back(tracker);
            }
		}
        // bgs_tracked_list.erase(remove_if(bgs_tracked_list.begin(), bgs_tracked_list.end(), [](const Tracker& obj) 
		//                        { return obj.missCounter >= 1; }), bgs_tracked_list.end());
    vector<Tracker>::iterator iter;
    for (iter = bgs_tracked_list.begin(); iter!= bgs_tracked_list.end();)
    {
        if (iter->missCounter >= 3)
        {
            iter = bgs_tracked_list.erase(iter);
        }
		else{
            iter ++ ; 
		}
    }
		for (size_t i = 0; i < bgs_tracked_list.size(); i++) {
            if (!bgs_tracked_list[i].update_status) {
                bgs_tracked_list[i].missCounter++;
                continue;
            }
            if (bgs_tracked_list[i].hitCounter > 2) {
                bgs_tracked_res.push_back(bgs_tracked_list[i].box);
            }
        }
	}

    std::vector<cv::Rect> unique_bgs_tracked_res;
    for (size_t i = 0; i < bgs_tracked_res.size(); i++) {
        bool is_unique = true;
        for (size_t j = 0; j < unique_bgs_tracked_res.size(); j++) {
            if (bgs_tracked_res[i] == unique_bgs_tracked_res[j]) {
                is_unique = false;
                break;
            }
        }
        if (is_unique) {
            unique_bgs_tracked_res.push_back(bgs_tracked_res[i]);
        }
    }
	new_bgs_tracked_res.clear();
	for (size_t i = 0; i < unique_bgs_tracked_res.size(); i++) {	
			Rect box = unique_bgs_tracked_res[i];
			cv::Point2f new_point=compensate(cv::Point2f(box.x,box.y),h);
			// float newW = h[6] * box.x + h[7] * box.y + h[8];
			// float newX = (h[0] * box.x + h[1] * box.y + h[2]) / newW;
			// float newY = (h[3] * box.x + h[4] * box.y + h[5]) / newW;
			float newX = new_point.x;
			float newY = new_point.y;
			if (newX < 0) {
                newX = 0;
            }
            if (newX + box.width >= input_img.cols) {
                newX = input_img.cols - box.width -1;
            }
            if (newY < 0) {
                newY = 0;
            }
            if (newY+box.height >= input_img.rows) {
                newY = input_img.rows - box.width -1;
            }
		    bgs_tracked_list_point.push_back(Rect(int(newX), int(newY), box.width, box.height));
			Mat old_bbox = input_img(box);
            Mat new_bbox = input_img(Rect(int(newX), int(newY), box.width, box.height));
			if (old_bbox.rows != new_bbox.rows || old_bbox.cols != new_bbox.cols) {
                 continue;
            }else {
                double rr = cv::norm(old_bbox, new_bbox, cv::NORM_L2SQR);
                if (rr > 250.0) {
                    new_bgs_tracked_res.push_back(box);
                }
            }
	}
	gettimeofday(&toc, NULL);
	rt_modelUpdate = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	rt_total = rt_motionComp + rt_modelUpdate;


	// Debug Output
	for (int i = 0; i < 100; ++i) {
		printf("\b");
	}
	printf("OF: %.2f(ms)\tBGM: %.2f(ms)\tTotal time: \t%.2f(ms)", MAX(0.0, rt_motionComp), MAX(0.0, rt_modelUpdate), MAX(0.0, rt_total));


	imgGrayPrev=imgGray.clone();
	cvWaitKey(10);
    // return BGModel.mask;
	return  background_res;

}

#endif				// _MCDWRAPPER_CPP_
