
#ifndef _PROB_MODEL_H_
#define _PROB_MODEL_H_

/************************************************************************/
/* Basic Includes                                                       */
/************************************************************************/
#include	<iostream>
#include	<cstdlib>
#include	<cstring>
#include	<vector>
#include	<algorithm>
#include    <opencv2/imgproc.hpp>
#include    <opencv/cv.h>
#include    <ctime>
/************************************************************************/
/*  Necessary includes for this Algorithm                               */
/************************************************************************/

#include "params.hpp"

class ProbModel {

 public:

	Mat m_Cur;
	Mat m_DistImg;
    Mat mask;
	float *m_Mean[NUM_MODELS];
	float *m_Var[NUM_MODELS];
	float *m_Age[NUM_MODELS];

	float *m_Mean_Temp[NUM_MODELS];
	float *m_Var_Temp[NUM_MODELS];
	float *m_Age_Temp[NUM_MODELS];

	int *m_ModelIdx;

	int modelWidth;
	int modelHeight;

	int obsWidth;
	int obsHeight;
    std::vector<char>res;
 public:
	 ProbModel() {

		for (int i = 0; i < NUM_MODELS; ++i) {
			m_Mean[i] = 0;
			m_Var[i] = 0;
			m_Age[i] = 0;
			m_Mean_Temp[i] = 0;
			m_Var_Temp[i] = 0;
			m_Age_Temp[i] = 0;
		} m_ModelIdx = 0;
	}
	~ProbModel() {
		uninit();
	}

	void uninit(void) {
		for (int i = 0; i < NUM_MODELS; ++i) {
			if (m_Mean[i] != 0) {
				delete m_Mean[i];
				m_Mean[i] = 0;
			}
			if (m_Var[i] != 0) {
				delete m_Var[i];
				m_Var[i] = 0;
			}
			if (m_Age[i] != 0) {
				delete m_Age[i];
				m_Age[i] = 0;
			}
			if (m_Mean_Temp[i] != 0) {
				delete m_Mean_Temp[i];
				m_Mean_Temp[i] = 0;
			}
			if (m_Var_Temp[i] != 0) {
				delete m_Var_Temp[i];
				m_Var_Temp[i] = 0;
			}
			if (m_Age_Temp[i] != 0) {
				delete m_Age_Temp[i];
				m_Age_Temp[i] = 0;
			}
		}
		if (m_ModelIdx != 0) {
			delete m_ModelIdx;
			m_ModelIdx = 0;
		}

	}

	void init(Mat pInputImg) {

		uninit();

		m_Cur = pInputImg;

		obsWidth = pInputImg.cols;
		obsHeight = pInputImg.rows;

		modelWidth = obsWidth / BLOCK_SIZE;
		modelHeight = obsHeight / BLOCK_SIZE;

		/////////////////////////////////////////////////////////////////////////////
		// Initialize Storage
		m_DistImg = Mat::zeros(obsHeight,obsWidth,CV_32FC1);

		for (int i = 0; i < NUM_MODELS; ++i) {
			m_Mean[i] = new float[modelWidth * modelHeight];
			m_Var[i] = new float[modelWidth * modelHeight];
			m_Age[i] = new float[modelWidth * modelHeight];

			m_Mean_Temp[i] = new float[modelWidth * modelHeight];
			m_Var_Temp[i] = new float[modelWidth * modelHeight];
			m_Age_Temp[i] = new float[modelWidth * modelHeight];

			memset(m_Mean[i], 0, sizeof(float) * modelWidth * modelHeight);
			memset(m_Var[i], 0, sizeof(float) * modelWidth * modelHeight);
			memset(m_Age[i], 0, sizeof(float) * modelWidth * modelHeight);
		}
		m_ModelIdx = new int[modelWidth * modelHeight];

		// update with homography I
		double h[16][9];
	    for (int j = 0; j < 16; ++j) {
		  for (int ii = 0; ii < 9; ++ii) {
			h[j][ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		 }
	    }

		motionCompensate(h,0);
		update(pInputImg);

	}

	void motionCompensate(double h[16][9],int frame_num) {

		int curModelWidth = modelWidth;
		int curModelHeight = modelHeight;

		// compensate models for the current view
		for (int j = 0; j < curModelHeight; ++j) {
			for (int i = 0; i < curModelWidth; ++i) {
				float X, Y;
				float W = 1.0;
				X = BLOCK_SIZE * i + BLOCK_SIZE / 2.0;
				Y = BLOCK_SIZE * j + BLOCK_SIZE / 2.0;
				float newW = 0;
				float newX = 0;
				float newY = 0;
				// newW = X;
				// newX = X;
				// newY = X;
				// transformed coordinates with h
				if(X<obsWidth/4 && Y<obsHeight/4)
				{
				newW = h[0][6] * X + h[0][7] * Y + h[0][8];
				newX = (h[0][0] * X + h[0][1] * Y + h[0][2]) / newW;
				newY = (h[0][3] * X + h[0][4] * Y + h[0][5]) / newW;
				}
				else if(X<obsWidth/2&&X>=obsWidth/4&&Y<obsHeight/4)
				{
				newW = h[1][6] * X + h[1][7] * Y + h[1][8];
				newX = (h[1][0] * X + h[1][1] * Y + h[1][2]) / newW;
				newY = (h[1][3] * X + h[1][4] * Y + h[1][5]) / newW;
				}
				else if(X<obsWidth/4*3&&X>=obsWidth/2&&Y<obsHeight/4)
				{
				newW = h[2][6] * X + h[2][7] * Y + h[2][8];
				newX = (h[2][0] * X + h[2][1] * Y + h[2][2]) / newW;
				newY = (h[2][3] * X + h[2][4] * Y + h[2][5]) / newW;
				}
				else if(X<obsWidth&&X>=obsWidth/4*3&&Y<obsHeight/4)
				{
				newW = h[3][6] * X + h[3][7] * Y + h[3][8];
				newX = (h[3][0] * X + h[3][1] * Y + h[3][2]) / newW;
				newY = (h[3][3] * X + h[3][4] * Y + h[3][5]) / newW;
				}

				else if(X<obsWidth/4&&Y<obsHeight/2&&Y>=obsHeight/4)
				{
				newW = h[4][6] * X + h[4][7] * Y + h[4][8];
				newX = (h[4][0] * X + h[4][1] * Y + h[4][2]) / newW;
				newY = (h[4][3] * X + h[4][4] * Y + h[4][5]) / newW;
				}
				else if(X<obsWidth/2&&X>=obsWidth/4&&Y<obsHeight/2&&Y>=obsHeight/4)
				{
				newW = h[5][6] * X + h[5][7] * Y + h[5][8];
				newX = (h[5][0] * X + h[5][1] * Y + h[5][2]) / newW;
				newY = (h[5][3] * X + h[5][4] * Y + h[5][5]) / newW;
				}
				else if(X<obsWidth/4*3&&X>=obsWidth/2&&Y<obsHeight/2&&Y>=obsHeight/4)
				{
				newW = h[6][6] * X + h[6][7] * Y + h[6][8];
				newX = (h[6][0] * X + h[6][1] * Y + h[6][2]) / newW;
				newY = (h[6][3] * X + h[6][4] * Y + h[6][5]) / newW;
				}
				else if(X<obsWidth&&X>=obsWidth/4*3&&Y<obsHeight/2&&Y>=obsHeight/4)
				{
				newW = h[7][6] * X + h[7][7] * Y + h[7][8];
				newX = (h[7][0] * X + h[7][1] * Y + h[7][2]) / newW;
				newY = (h[7][3] * X + h[7][4] * Y + h[7][5]) / newW;
				}
				else if(X<obsWidth/4&&Y<obsHeight/4*3&&Y>=obsHeight/2)
				{
				newW = h[8][6] * X + h[8][7] * Y + h[8][8];
				newX = (h[8][0] * X + h[8][1] * Y + h[8][2]) / newW;
				newY = (h[8][3] * X + h[8][4] * Y + h[8][5]) / newW;
				}
				else if(X<obsWidth/2&&X>=obsWidth/4&&Y<obsHeight/4*3&&Y>=obsHeight/2)
				{
				newW = h[9][6] * X + h[9][7] * Y + h[9][8];
				newX = (h[9][0] * X + h[9][1] * Y + h[9][2]) / newW;
				newY = (h[9][3] * X + h[9][4] * Y + h[9][5]) / newW;
				}
				else if(X<obsWidth/4*3&&X>=obsWidth/2&&Y<obsHeight/4*3&&Y>=obsHeight/2)
				{
				newW = h[10][6] * X + h[10][7] * Y + h[10][8];
				newX = (h[10][0] * X + h[10][1] * Y + h[10][2]) / newW;
				newY = (h[10][3] * X + h[10][4] * Y + h[10][5]) / newW;
				}
				else if(X<obsWidth&&X>=obsWidth/4*3&&Y<obsHeight/4*3&&Y>=obsHeight/2)
				{
				newW = h[11][6] * X + h[11][7] * Y + h[11][8];
				newX = (h[11][0] * X + h[11][1] * Y + h[11][2]) / newW;
				newY = (h[11][3] * X + h[11][4] * Y + h[11][5]) / newW;
				}
				else if(X<obsWidth/4&&Y<obsHeight&&Y>=obsHeight/4*3)
				{
				newW = h[12][6] * X + h[12][7] * Y + h[12][8];
				newX = (h[12][0] * X + h[12][1] * Y + h[12][2]) / newW;
				newY = (h[12][3] * X + h[12][4] * Y + h[12][5]) / newW;
				}
				else if(X<obsWidth/2&&X>=obsWidth/4&&Y<obsHeight&&Y>=obsHeight/4*3)
				{
				newW = h[13][6] * X + h[13][7] * Y + h[13][8];
				newX = (h[13][0] * X + h[13][1] * Y + h[13][2]) / newW;
				newY = (h[13][3] * X + h[13][4] * Y + h[13][5]) / newW;
				}
				else if(X<obsWidth/4*3&&X>=obsWidth/2&&Y<obsHeight&&Y>=obsHeight/4*3)
				{
				newW = h[14][6] * X + h[14][7] * Y + h[14][8];
				newX = (h[14][0] * X + h[14][1] * Y + h[14][2]) / newW;
				newY = (h[14][3] * X + h[14][4] * Y + h[14][5]) / newW;
				}
				else if(X<obsWidth&&X>=obsWidth/4*3&&Y<obsHeight&&Y>=obsHeight/4*3)
				{
				newW = h[15][6] * X + h[15][7] * Y + h[15][8];
				newX = (h[15][0] * X + h[15][1] * Y + h[15][2]) / newW;
				newY = (h[15][3] * X + h[15][4] * Y + h[15][5]) / newW;
				}


				// float newW = h[6] * X + h[7] * Y + h[8];
				// float newX = (h[0] * X + h[1] * Y + h[2]) / newW;
				// float newY = (h[3] * X + h[4] * Y + h[5]) / newW;
				// newW = h[15][6] * X + h[15][7] * Y + h[15][8];
				// newX = (h[15][0] * X + h[15][1] * Y + h[15][2]) / newW;
				// newY = (h[15][3] * X + h[15][4] * Y + h[15][5]) / newW;


				float newI = newX / BLOCK_SIZE;
				float newJ = newY / BLOCK_SIZE;

				int idxNewI = floor(newI);
				int idxNewJ = floor(newJ);

				float di = newI - ((float)(idxNewI) + 0.5);
				float dj = newJ - ((float)(idxNewJ) + 0.5);

				float w_H = 0.0;
				float w_V = 0.0;
				float w_HV = 0.0;
				float w_self = 0.0;
				float sumW = 0.0;

				int idxNow = i + j * modelWidth;

#define WARP_MIX
				// For Mean and Age
				{
					float temp_mean[4][NUM_MODELS];
					float temp_age[4][NUM_MODELS];
					memset(temp_mean, 0, sizeof(float) * 4 * NUM_MODELS);
					memset(temp_age, 0, sizeof(float) * 4 * NUM_MODELS);
#ifdef WARP_MIX
					// Horizontal Neighbor
					if (di != 0) {
						int idx_new_i = idxNewI;
						int idx_new_j = idxNewJ;
						idx_new_i += di > 0 ? 1 : -1;
						if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
							w_H = fabs(di) * (1.0 - fabs(dj));
							sumW += w_H;
							int idxNew = idx_new_i + idx_new_j * modelWidth;
							for (int m = 0; m < NUM_MODELS; ++m) {
								temp_mean[0][m] = w_H * m_Mean[m][idxNew];
								temp_age[0][m] = w_H * m_Age[m][idxNew];
							}
						}
					}
					// Vertical Neighbor
					if (dj != 0) {
						int idx_new_i = idxNewI;
						int idx_new_j = idxNewJ;
						idx_new_j += dj > 0 ? 1 : -1;
						if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
							w_V = fabs(dj) * (1.0 - fabs(di));
							sumW += w_V;
							int idxNew = idx_new_i + idx_new_j * modelWidth;
							for (int m = 0; m < NUM_MODELS; ++m) {
								temp_mean[1][m] = w_V * m_Mean[m][idxNew];
								temp_age[1][m] = w_V * m_Age[m][idxNew];
							}
						}
					}
					// HV Neighbor
					if (dj != 0 && di != 0) {
						int idx_new_i = idxNewI;
						int idx_new_j = idxNewJ;
						idx_new_i += di > 0 ? 1 : -1;
						idx_new_j += dj > 0 ? 1 : -1;
						if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
							w_HV = fabs(di) * fabs(dj);
							sumW += w_HV;
							int idxNew = idx_new_i + idx_new_j * modelWidth;
							for (int m = 0; m < NUM_MODELS; ++m) {
								temp_mean[2][m] = w_HV * m_Mean[m][idxNew];
								temp_age[2][m] = w_HV * m_Age[m][idxNew];
							}
						}
					}
#endif
					// Self
					if (idxNewI >= 0 && idxNewI < curModelWidth && idxNewJ >= 0 && idxNewJ < curModelHeight) {
						w_self = (1.0 - fabs(di)) * (1.0 - fabs(dj));
						sumW += w_self;
						int idxNew = idxNewI + idxNewJ * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_mean[3][m] = w_self * m_Mean[m][idxNew];
							temp_age[3][m] = w_self * m_Age[m][idxNew];
						}
					}

					if (sumW > 0) {
						for (int m = 0; m < NUM_MODELS; ++m) {
#ifdef WARP_MIX
							m_Mean_Temp[m][idxNow] = (temp_mean[0][m] + temp_mean[1][m] + temp_mean[2][m] + temp_mean[3][m]) / sumW;
							m_Age_Temp[m][idxNow] = (temp_age[0][m] + temp_age[1][m] + temp_age[2][m] + temp_age[3][m]) / sumW;
#else
							m_Mean_Temp[m][idxNow] = temp_mean[3][m] / sumW;
							m_Age_Temp[m][idxNow] = temp_age[3][m] / sumW;
#endif
						}
					}
				}

				// For Variance
				{
					float temp_var[4][NUM_MODELS];
					memset(temp_var, 0, sizeof(float) * 4 * NUM_MODELS);
#ifdef WARP_MIX
					// Horizontal Neighbor
					if (di != 0) {
						int idx_new_i = idxNewI;
						int idx_new_j = idxNewJ;
						idx_new_i += di > 0 ? 1 : -1;
						if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
							int idxNew = idx_new_i + idx_new_j * modelWidth;
							for (int m = 0; m < NUM_MODELS; ++m) {
								// temp_var[0][m] = w_H * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
								float tmp=m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew];
								temp_var[0][m] = w_H * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * tmp * tmp);
							}
						}
					}
					// Vertical Neighbor
					if (dj != 0) {
						int idx_new_i = idxNewI;
						int idx_new_j = idxNewJ;
						idx_new_j += dj > 0 ? 1 : -1;
						if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
							int idxNew = idx_new_i + idx_new_j * modelWidth;
							for (int m = 0; m < NUM_MODELS; ++m) {
								// temp_var[1][m] = w_V * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
								float tmp=m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew];
								temp_var[1][m] = w_V * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * tmp * tmp);
							}
						}
					}
					// HV Neighbor
					if (dj != 0 && di != 0) {
						int idx_new_i = idxNewI;
						int idx_new_j = idxNewJ;
						idx_new_i += di > 0 ? 1 : -1;
						idx_new_j += dj > 0 ? 1 : -1;
						if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
							int idxNew = idx_new_i + idx_new_j * modelWidth;
							for (int m = 0; m < NUM_MODELS; ++m) {
								// temp_var[2][m] = w_HV * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
								float tmp=m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew];
								temp_var[2][m] = w_HV * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * tmp * tmp);
							}
						}
					}
 #endif
					// Self
					if (idxNewI >= 0 && idxNewI < curModelWidth && idxNewJ >= 0 && idxNewJ < curModelHeight) {
						int idxNew = idxNewI + idxNewJ * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							// temp_var[3][m] = w_self * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
							temp_var[3][m] = w_self * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew])*(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew]));
						}
					}

					if (sumW > 0) {
						for (int m = 0; m < NUM_MODELS; ++m) {
#ifdef WARP_MIX
							m_Var_Temp[m][idxNow] = (temp_var[0][m] + temp_var[1][m] + temp_var[2][m] + temp_var[3][m]) / sumW;
#else
							m_Var_Temp[m][idxNow] = (temp_var[3][m]) / sumW;
#endif
						}
					}

				}

				// Limitations and Exceptions
				for (int m = 0; m < NUM_MODELS; ++m) {
					m_Var_Temp[m][i + j * modelWidth] = MAX(m_Var_Temp[m][i + j * modelWidth], MIN_BG_VAR);
				}
				if (idxNewI < 1 || idxNewI >= modelWidth - 1 || idxNewJ < 1 || idxNewJ >= modelHeight - 1) {
					for (int m = 0; m < NUM_MODELS; ++m) {
						m_Var_Temp[m][i + j * modelWidth] = INIT_BG_VAR;
						m_Age_Temp[m][i + j * modelWidth] = 0;
					}
				} else {
					for (int m = 0; m < NUM_MODELS; ++m) {
						m_Age_Temp[m][i + j * modelWidth] =
						    MIN(m_Age_Temp[m][i + j * modelWidth] * exp(-VAR_DEC_RATIO * MAX(0.0, m_Var_Temp[m][i + j * modelWidth] - VAR_MIN_NOISE_T)), MAX_BG_AGE);
					}
				}
			}
		}

	}

	void update(const Mat& pOutputImg) {
		clock_t t1,t2,t3,t4,t5,t6;
		// cv::cvtColor(pOutputImg, pOutputImg, cv::COLOR_BGR2GRAY);
		int curModelWidth = modelWidth;
		int curModelHeight = modelHeight;
		mask = Mat::zeros(obsHeight, obsWidth, CV_8UC1);
		//////////////////////////////////////////////////////////////////////////
		// Find Matching Model
		memset(m_ModelIdx, 0, sizeof(int) * modelHeight * modelWidth);
        t1=clock();
		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++) {
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++) {

				// base (i,j) for this block
				int idx_base_i;
				int idx_base_j;
				idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
				idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;

				float cur_mean = 0;
				float elem_cnt = 0;
				for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
					for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

						int idx_i = idx_base_i + ii;
						int idx_j = idx_base_j + jj;

						if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
							continue;

						cur_mean += pOutputImg.at<uchar>(idx_j, idx_i);
						elem_cnt += 1.0;
					}
				}	//loop for pixels
				cur_mean /= elem_cnt;

				//////////////////////////////////////////////////////////////////////////
				// Make Oldest Idx to 0 (swap)
				int oldIdx = 0;
				float oldAge = 0;
				for (int m = 0; m < NUM_MODELS; ++m) {
					float fAge = m_Age_Temp[m][bIdx_i + bIdx_j * modelWidth];

					if (fAge >= oldAge) {
						oldIdx = m;
						oldAge = fAge;
					}
				}
				if (oldIdx != 0) {
					m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth] = m_Mean_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth];
					m_Mean_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth] = cur_mean;

					m_Var_Temp[0][bIdx_i + bIdx_j * modelWidth] = m_Var_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth];
					m_Var_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth] = INIT_BG_VAR;

					m_Age_Temp[0][bIdx_i + bIdx_j * modelWidth] = m_Age_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth];
					m_Age_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth] = 0;
				}
				//////////////////////////////////////////////////////////////////////////
				// Select Model 
				// Check Match against 0

				if ((cur_mean - m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth])*(cur_mean - m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth]) < VAR_THRESH_MODEL_MATCH * m_Var_Temp[0][bIdx_i + bIdx_j * modelWidth]) {
					m_ModelIdx[bIdx_i + bIdx_j * modelWidth] = 0;
				}
				// Check Match against 1
				else if ((cur_mean - m_Mean_Temp[1][bIdx_i + bIdx_j * modelWidth])*(cur_mean - m_Mean_Temp[1][bIdx_i + bIdx_j * modelWidth])< VAR_THRESH_MODEL_MATCH * m_Var_Temp[1][bIdx_i + bIdx_j * modelWidth]) {
					m_ModelIdx[bIdx_i + bIdx_j * modelWidth] = 1;
				}
				// If No match, set 1 age to zero and match = 1
				else {
					m_ModelIdx[bIdx_i + bIdx_j * modelWidth] = 1;
					m_Age_Temp[1][bIdx_i + bIdx_j * modelWidth] = 0;
				}

			}
		}		// loop for models
        t2=clock();
		// update with current observation
		float obs_mean[NUM_MODELS];
		float obs_var[NUM_MODELS];
        t3=clock();
		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++) {
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++) {

				// base (i,j) for this block
				int idx_base_i;
				int idx_base_j;
				idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
				idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;

				int nMatchIdx = m_ModelIdx[bIdx_i + bIdx_j * modelWidth];

				// obtain observation mean
				memset(obs_mean, 0, sizeof(float) * NUM_MODELS);
				int nElemCnt[NUM_MODELS];
				memset(nElemCnt, 0, sizeof(int) * NUM_MODELS);
				for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
					for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

						int idx_i = idx_base_i + ii;
						int idx_j = idx_base_j + jj;

						if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
							continue;

						obs_mean[nMatchIdx] += pOutputImg.at<uchar>(idx_j, idx_i);
						++nElemCnt[nMatchIdx];
					}
				}
				for (int m = 0; m < NUM_MODELS; ++m) {

					if (nElemCnt[m] <= 0) {
						m_Mean[m][bIdx_i + bIdx_j * modelWidth] = m_Mean_Temp[m][bIdx_i + bIdx_j * modelWidth];
					} else {
						// learning rate for this block
						float age = m_Age_Temp[m][bIdx_i + bIdx_j * modelWidth];
						float alpha = age / (age + 1.0);

						obs_mean[m] /= ((float)nElemCnt[m]);
						// update with this mean
						if (age < 1) {
							m_Mean[m][bIdx_i + bIdx_j * modelWidth] = obs_mean[m];
						} else {
							m_Mean[m][bIdx_i + bIdx_j * modelWidth] = alpha * m_Mean_Temp[m][bIdx_i + bIdx_j * modelWidth] + (1.0 - alpha) * obs_mean[m];
						}

					}
				}
			}
		}
        t4=clock();
		
		t5=clock();
		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++) {
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++) {
				// TODO: OPTIMIZE THIS PART SO THAT WE DO NOT CALCULATE THIS (LUT)
				// base (i,j) for this block
				int idx_base_i;
				int idx_base_j;
				// idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
				// idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;
				idx_base_i = bIdx_i * BLOCK_SIZE;
				idx_base_j = bIdx_j * BLOCK_SIZE;
				int nMatchIdx = m_ModelIdx[bIdx_i + bIdx_j * modelWidth];

				// obtain observation variance
				memset(obs_var, 0, sizeof(float) * NUM_MODELS);
				int nElemCnt[NUM_MODELS];
				memset(nElemCnt, 0, sizeof(int) * NUM_MODELS);
				for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
					for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

						int idx_i = idx_base_i + ii;
						int idx_j = idx_base_j + jj;
						nElemCnt[nMatchIdx]++;

						if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight) {
							continue;
						}

						float pixelDist = 0.0;
						float fDiff = pOutputImg.at<uchar>(idx_j, idx_i)- m_Mean[nMatchIdx][bIdx_i + bIdx_j * modelWidth];
						// pixelDist += pow(fDiff, (int)2);
                        pixelDist += fDiff * fDiff;
						// m_DistImg.at<float>(idx_j, idx_i) = pow(pOutputImg.at<uchar>(idx_j, idx_i) - m_Mean[0][bIdx_i + bIdx_j * modelWidth], (int)2);
						float diff=pOutputImg.at<uchar>(idx_j, idx_i) - m_Mean[0][bIdx_i + bIdx_j * modelWidth];
                        m_DistImg.at<float>(idx_j, idx_i) = diff*diff;

						if (m_Age_Temp[0][bIdx_i + bIdx_j * modelWidth] > 1) {

							BYTE valOut = m_DistImg.at<float>(idx_j, idx_i) > VAR_THRESH_FG_DETERMINE * m_Var_Temp[0][bIdx_i + bIdx_j * modelWidth] ? 255 : 0;
							mask.at<uchar>(idx_j, idx_i) = valOut;
		

						}

						obs_var[nMatchIdx] = MAX(obs_var[nMatchIdx], pixelDist);
					}
				}

				for (int m = 0; m < NUM_MODELS; ++m) {
					if (nElemCnt[m] > 0) {
						float age = m_Age_Temp[m][bIdx_i + bIdx_j * modelWidth];
						float alpha = age / (age + 1.0);
						if (age == 0) {
							m_Var[m][bIdx_i + bIdx_j * modelWidth] = MAX(obs_var[m], INIT_BG_VAR);
						} else {
							float alpha_var = alpha;	
							m_Var[m][bIdx_i + bIdx_j * modelWidth] = alpha_var * m_Var_Temp[m][bIdx_i + bIdx_j * modelWidth] + (1.0 - alpha_var) * obs_var[m];
							m_Var[m][bIdx_i + bIdx_j * modelWidth] = MAX(m_Var[m][bIdx_i + bIdx_j * modelWidth], MIN_BG_VAR);
						}
						m_Age[m][bIdx_i + bIdx_j * modelWidth] = m_Age_Temp[m][bIdx_i + bIdx_j * modelWidth] + 1.0;
						m_Age[m][bIdx_i + bIdx_j * modelWidth] = MIN(m_Age[m][bIdx_i + bIdx_j * modelWidth], MAX_BG_AGE);
					} else {
						m_Var[m][bIdx_i + bIdx_j * modelWidth] = m_Var_Temp[m][bIdx_i + bIdx_j * modelWidth];
						m_Age[m][bIdx_i + bIdx_j * modelWidth] = m_Age_Temp[m][bIdx_i + bIdx_j * modelWidth];
					}
				}

			}
		}
	    t6=clock();
		double run1=t2-t1;
		double run2=t4-t3;
		double run3=t6-t5;
		std::cout<<"Run1:"<<run1/CLOCKS_PER_SEC<<" "<<"Run2:"<<run2/CLOCKS_PER_SEC<<" "<<"Run3:"<<run3/CLOCKS_PER_SEC<<std::endl;
	 }
};

#endif				// _PROB_MODEL_H_
