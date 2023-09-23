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

Mat MCDWrapper::Run(const Mat & input_img,int frame_num)
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
	double h[9];
	m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
	m_LucasKanade.GetHomography(h);
	BGModel.motionCompensate(h,frame_num);
    int modelWidth=88;
    int modelHeight=72;
	for(int i=0;i<modelWidth*modelHeight;i++)
	{
		m_Mean[i] = BGModel.m_Mean[0][i];
		m_Var[i] = BGModel.m_Var[0][i];
		m_Age[i] = BGModel.m_Age[0][i];

		m_Mean_Temp[i] = BGModel.m_Mean_Temp[0][i];
		m_Var_Temp[i] = BGModel.m_Var_Temp[0][i];
		m_Age_Temp[i] = BGModel.m_Age_Temp[0][i];
	}
	//--TIME END
	gettimeofday(&toc, NULL);
	rt_motionComp = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//--TIME START
	gettimeofday(&tic, NULL);
	// Update BG Model and Detect
	Mat Temp= imgGray.clone();
	BGModel.update(Temp);
	for(int i=0;i<modelWidth*modelHeight;i++)
	{
		m_Mean1[i] = BGModel.m_Mean[0][i];
		m_Var1[i] = BGModel.m_Var[0][i];
		m_Age1[i] = BGModel.m_Age[0][i];

		m_Mean_Temp1[i] = BGModel.m_Mean_Temp[0][i];
		m_Var_Temp1[i] = BGModel.m_Var_Temp[0][i];
		m_Age_Temp1[i] = BGModel.m_Age_Temp[0][i];
	}
	Res.clear();
	for(int i=0;i<BGModel.res.size();i++)
	{
		Res.push_back(BGModel.res[i]);
	}
	//--TIME END
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
    return BGModel.mask;

}

#endif				// _MCDWRAPPER_CPP_
