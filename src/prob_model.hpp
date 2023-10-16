
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
#include    <CL/cl.h>
/************************************************************************/
/*  Necessary includes for this Algorithm                               */
/************************************************************************/

#include "params.hpp"
using namespace cv;

class ProbModel {

 public:

	Mat m_DistImg;
    Mat mask;
	float *m_Mean[NUM_MODELS];
	float *m_Var[NUM_MODELS];
	float *m_Age[NUM_MODELS];

	float *m_Mean_Temp[NUM_MODELS];
	float *m_Var_Temp[NUM_MODELS];
	float *m_Age_Temp[NUM_MODELS];

	float *m_Mean_TempTTT[NUM_MODELS];
	float *m_Var_TempTTT[NUM_MODELS];
	float *m_Age_TempTTT[NUM_MODELS];

    float *di;
	float *dj;
    float *idxNewI;
	float *idxNewJ;

	int *m_ModelIdx;

	int modelWidth;
	int modelHeight;

	int obsWidth;
	int obsHeight;
	cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
	cl_kernel h_kernel = 0;
    cl_mem memObjects[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    cl_int errNum;


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
			if (m_Mean_TempTTT[i] != 0) {
				delete m_Mean_TempTTT[i];
				m_Mean_TempTTT[i] = 0;
			}
			if (m_Var_TempTTT[i] != 0) {
				delete m_Var_TempTTT[i];
				m_Var_TempTTT[i] = 0;
			}
			if (m_Age_TempTTT[i] != 0) {
				delete m_Age_TempTTT[i];
				m_Age_TempTTT[i] = 0;
			}
		}
		if (m_ModelIdx != 0) {
			delete m_ModelIdx;
			m_ModelIdx = 0;
		}

	}

	void init(Mat pInputImg) {

		uninit();

		obsWidth = pInputImg.cols;
		obsHeight = pInputImg.rows;

		modelWidth = obsWidth / BLOCK_SIZE;
		modelHeight = obsHeight / BLOCK_SIZE;

		/////////////////////////////////////////////////////////////////////////////
		// Initialize Storage
		m_DistImg = Mat::zeros(obsHeight,obsWidth,CV_32FC1);

	    di= new float[modelWidth * modelHeight];
        dj= new float[modelWidth * modelHeight];
        idxNewI=new float[modelWidth * modelHeight];
	    idxNewJ=new float[modelWidth * modelHeight];

        memset(di, 0, sizeof(float) * modelWidth * modelHeight);
	    memset(dj, 0, sizeof(float) * modelWidth * modelHeight);
	    memset(idxNewI, 0, sizeof(float) * modelWidth * modelHeight);
        memset(idxNewJ, 0, sizeof(float) * modelWidth * modelHeight);

		for (int i = 0; i < NUM_MODELS; ++i) {
			m_Mean[i] = new float[modelWidth * modelHeight];
			m_Var[i] = new float[modelWidth * modelHeight];
			m_Age[i] = new float[modelWidth * modelHeight];

			m_Mean_Temp[i] = new float[modelWidth * modelHeight];
			m_Var_Temp[i] = new float[modelWidth * modelHeight];
			m_Age_Temp[i] = new float[modelWidth * modelHeight];

			m_Mean_TempTTT[i] = new float[modelWidth * modelHeight];
			m_Var_TempTTT[i] = new float[modelWidth * modelHeight];
			m_Age_TempTTT[i] = new float[modelWidth * modelHeight];


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

    // 一、选择OpenCL平台并创建一个上下文
    context = CreateContext();

    // 二、 创建设备并创建命令队列
    commandQueue = CreateCommandQueue(context, &device);

    // 三、创建和构建程序对象
    program = CreateProgram(context, device, "compensate.cl");
    // 四、 创建OpenCL内核并分配内存空间
    kernel = clCreateKernel(program, "compensate_kernel", NULL);
	h_kernel = clCreateKernel(program, "h_cal_kernel", NULL);
	// motionCompensate(h,0);
	update(pInputImg);

	}

    cl_context CreateContext()
{

    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // 选择可用的平台中的第一个
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // 创建一个OpenCL上下文环境
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
        NULL, NULL, &errNum);

    return context;
}

    // 二、 创建设备并创建命令队列
    cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device)
{
    cl_int errNum;
    cl_device_id* devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // 获取设备缓冲区大小
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

    // 为设备分配缓存空间
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

    // 选取可用设备中的第一个
    commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

    // 三、创建和构建程序对象
    cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char* srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
        (const char**)&srcStr,
        NULL, NULL);

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    return program;
}

    // 创建和构建程序对象
    bool CreateMemObjects(cl_context context, cl_mem memObjects[13],
  		double *h,
        float *m_Mean,
        float *m_Var,
        float *m_Age,
        float *m_Mean_Temp,
        float *m_Var_Temp,
        float *m_Age_Temp,
        float *m_Mean1,
        float *m_Var1,
        float *m_Age1,
        float *m_Mean_Temp1,
        float *m_Var_Temp1,
        float *m_Age_Temp1
        )
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(double) * 16 * 9, h, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var, NULL);
    memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age, NULL);
    memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean_Temp, NULL);
    memObjects[5] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var_Temp, NULL);
    memObjects[6] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age_Temp, NULL);
    memObjects[7] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean1, NULL);
    memObjects[8] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var1, NULL);
    memObjects[9] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age1, NULL);
    memObjects[10] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean_Temp1, NULL);
    memObjects[11] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var_Temp1,NULL);
    memObjects[12] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age_Temp1, NULL);
    return true;
}

// 释放OpenCL资源
    void Cleanup(cl_context context, cl_command_queue commandQueue,
    cl_program program, cl_kernel kernel, cl_mem memObjects[16])
{
    for (int i = 0; i < 3; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}

	void motionCompensate(double *h) {

		int curModelWidth = modelWidth;
		int curModelHeight = modelHeight;
	// 	clock_t t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,ts,te;
	// ts=clock();
    // t1=clock();
    if (!CreateMemObjects(context, memObjects,h,
                          m_Mean[0],m_Var[0],m_Age[0],m_Mean_Temp[0],m_Var_Temp[0],m_Age_Temp[0],
                          m_Mean[1],m_Var[1],m_Age[1],m_Mean_Temp[1],m_Var_Temp[1],m_Age_Temp[1])) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
    }
	// t2=clock();

	// t3=clock();
    // 五、 设置内核数据并执行内核
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[3]);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[4]);
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[5]);
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &memObjects[6]);
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &memObjects[7]);
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &memObjects[8]);
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &memObjects[9]);
    errNum |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &memObjects[10]);
    errNum |= clSetKernelArg(kernel, 11, sizeof(cl_mem), &memObjects[11]);
    errNum |= clSetKernelArg(kernel, 12, sizeof(cl_mem), &memObjects[12]);
    size_t globalWorkSize[1] = { (size_t)modelWidth * (size_t)modelHeight };
    size_t localWorkSize[1] = {10};
    // t4=clock();

    // t5=clock();
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
        globalWorkSize, localWorkSize,
        0, NULL, NULL);	
    // t6=clock();

    // t7=clock();
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[4], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Mean_Temp[0],
        0, NULL, NULL);
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[5], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Var_Temp[0],
        0, NULL, NULL);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[6], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Age_Temp[0],
        0, NULL, NULL);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[10], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Mean_Temp[1],
        0, NULL, NULL);
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[11], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Var_Temp[1],
        0, NULL, NULL);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[12], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Age_Temp[1],
        0, NULL, NULL);
    // t8=clock();

	// t9=clock();
    for (int i = 0; i < 16; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
//     t10=clock();
// 	te=clock();
// double t=t2-t1;
// double tt=t4-t3;
// double ttt=t6-t5;
// double tttt=t8-t7;
// double ttttt=t10-t9;
// double tttttt=te-ts;
// std::cout<<"CreateMemObjects:"<<t/CLOCKS_PER_SEC<<" "<<"clSetKernelArg:"<<tt/CLOCKS_PER_SEC<<" "<<"clEnqueueNDRangeKernel:"<<ttt/CLOCKS_PER_SEC<<" "<<"clEnqueueReadBuffer:"<<tttt/CLOCKS_PER_SEC<<
// " "<<"clReleaseMemObject:"<<ttttt/CLOCKS_PER_SEC<<" "<<"all:"<<tttttt/CLOCKS_PER_SEC<<std::endl;
}

	void update(const Mat& pOutputImg) {
		// clock_t t1,t2,t3,t4,t5,t6;
		// cv::cvtColor(pOutputImg, pOutputImg, cv::COLOR_BGR2GRAY);
		int curModelWidth = modelWidth;
		int curModelHeight = modelHeight;
		mask = Mat::zeros(obsHeight, obsWidth, CV_8UC1);
		//////////////////////////////////////////////////////////////////////////
		memset(m_ModelIdx, 0, sizeof(int) * modelHeight * modelWidth);
        // t1=clock();
		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++) {
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++) {

				int idx_base_i;
				int idx_base_j;
				idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
				idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;

				float cur_mean = 0;
				float elem_cnt = 0;
				//计算block均值
				for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
					for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

						int idx_i = idx_base_i + ii;
						int idx_j = idx_base_j + jj;

						if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
							continue;

						cur_mean += pOutputImg.at<uchar>(idx_j, idx_i);
						elem_cnt += 1.0;
					}
				}	
				cur_mean /= elem_cnt;

				//////////////////////////////////////////////////////////////////////////
				//查看模型的age，比较模型0和模型1的age，确定oldIdx，oldAge
				int oldIdx = 0;
				float oldAge = 0;
				for (int m = 0; m < NUM_MODELS; ++m) {
					float fAge = m_Age_Temp[m][bIdx_i + bIdx_j * modelWidth];

					if (fAge >= oldAge) {
						oldIdx = m;
						oldAge = fAge;
					}
				}
				//model1 age大于model0 age，model0必为错误的，将model1的模型给到model0，并使model1初始化
				if (oldIdx != 0) {
					m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth] = m_Mean_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth];
					m_Mean_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth] = cur_mean;

					m_Var_Temp[0][bIdx_i + bIdx_j * modelWidth] = m_Var_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth];
					m_Var_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth] = INIT_BG_VAR;

					m_Age_Temp[0][bIdx_i + bIdx_j * modelWidth] = m_Age_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth];
					m_Age_Temp[oldIdx][bIdx_i + bIdx_j * modelWidth] = 0;
				}

                //结合上一步，oldIdx=0时，正常根据均值判断属于哪个模型
				//oldIdx=1时，if中判断相当于判断当前点时候属于模型0(原模型1)，elseif必定会满足条件，相当于重新建立一个模型，else与elseif功能是相同的
				if ((cur_mean - m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth])*(cur_mean - m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth]) < VAR_THRESH_MODEL_MATCH * m_Var_Temp[0][bIdx_i + bIdx_j * modelWidth]) {
					m_ModelIdx[bIdx_i + bIdx_j * modelWidth] = 0;
				}else if ((cur_mean - m_Mean_Temp[1][bIdx_i + bIdx_j * modelWidth])*(cur_mean - m_Mean_Temp[1][bIdx_i + bIdx_j * modelWidth])< VAR_THRESH_MODEL_MATCH * m_Var_Temp[1][bIdx_i + bIdx_j * modelWidth]) {
					m_ModelIdx[bIdx_i + bIdx_j * modelWidth] = 1;
				}else {
					m_ModelIdx[bIdx_i + bIdx_j * modelWidth] = 1;
					m_Age_Temp[1][bIdx_i + bIdx_j * modelWidth] = 0;
				}
					if (m_ModelIdx[bIdx_i + bIdx_j * modelWidth]==1) {
						m_Mean[0][bIdx_i + bIdx_j * modelWidth] = m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth];
						float age = m_Age_Temp[1][bIdx_i + bIdx_j * modelWidth];
						float alpha = age / (age + 1.0);
						if (age < 1) {
							m_Mean[1][bIdx_i + bIdx_j * modelWidth] = cur_mean;
						} else {
							m_Mean[1][bIdx_i + bIdx_j * modelWidth] = alpha * m_Mean_Temp[1][bIdx_i + bIdx_j * modelWidth] + (1.0 - alpha) * cur_mean;
						}
					} else{
						m_Mean[1][bIdx_i + bIdx_j * modelWidth] = m_Mean_Temp[1][bIdx_i + bIdx_j * modelWidth];
						float age = m_Age_Temp[0][bIdx_i + bIdx_j * modelWidth];
						float alpha = age / (age + 1.0);
						if (age < 1) {
							m_Mean[0][bIdx_i + bIdx_j * modelWidth] = cur_mean;
						} else {
							m_Mean[0][bIdx_i + bIdx_j * modelWidth] = alpha * m_Mean_Temp[0][bIdx_i + bIdx_j * modelWidth] + (1.0 - alpha) * cur_mean;
						}
					} 


			}
		}

		float obs_var[NUM_MODELS];
		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++) {
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++) {
				int idx_base_i;
				int idx_base_j;
				idx_base_i = bIdx_i * BLOCK_SIZE;
				idx_base_j = bIdx_j * BLOCK_SIZE;
				int nMatchIdx = m_ModelIdx[bIdx_i + bIdx_j * modelWidth];
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
                        pixelDist += fDiff * fDiff;
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
						if (age== 0||fabs(age)<0.000000001) {
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
	    // t6=clock();
		// double run1=t2-t1;
		// double run2=t4-t3;
		// double run3=t6-t5;
		// std::cout<<"Run1:"<<run1/CLOCKS_PER_SEC<<" "<<"Run2:"<<run2/CLOCKS_PER_SEC<<" "<<"Run3:"<<run3/CLOCKS_PER_SEC<<std::endl;
	 }
};

#endif				// _PROB_MODEL_H_