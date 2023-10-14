
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
    bool CreateMemObjects(cl_context context, cl_mem memObjects[16],
        float *di,float *dj,float *idxNewI,float *idxNewJ,	
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
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, di, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, dj, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, idxNewI, NULL);
    memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, idxNewJ, NULL);
    memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean, NULL);
    memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var, NULL);
    memObjects[6] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age, NULL);
    memObjects[7] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean_Temp, NULL);
    memObjects[8] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var_Temp, NULL);
    memObjects[9] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age_Temp, NULL);
    memObjects[10] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean1, NULL);
    memObjects[11] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var1, NULL);
    memObjects[12] = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age1, NULL);
    memObjects[13] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean_Temp1, NULL);
    memObjects[14] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var_Temp1,NULL);
    memObjects[15] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
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

	void motionCompensate(double h[16][9]) {

		int curModelWidth = modelWidth;
		int curModelHeight = modelHeight;
		// compensate models for the current view
		clock_t t1,t2,t3,t4,t5,t6,t7,t8;
t1=clock();
t3=clock();
		for (int j = 0; j < curModelHeight; ++j) {
			for (int i = 0; i < curModelWidth; ++i) {
				float X, Y;
				float W = 1.0;
				X = BLOCK_SIZE * i + BLOCK_SIZE / 2.0;
				Y = BLOCK_SIZE * j + BLOCK_SIZE / 2.0;
				float newW = 0;
				float newX = 0;
				float newY = 0;
				int idxNow = i + j * modelWidth;

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

				float newI = newX / BLOCK_SIZE;
				float newJ = newY / BLOCK_SIZE;

				idxNewI[i + j * modelWidth] = floor(newI);
				idxNewJ[i + j * modelWidth] = floor(newJ);

				di[i + j * modelWidth] = newI - (idxNewI[i + j * modelWidth] + 0.5);
				dj[i + j * modelWidth] = newJ - (idxNewJ[i + j * modelWidth] + 0.5);			

			}
		}
    // 创建内存对象

    if (!CreateMemObjects(context, memObjects, di,dj,idxNewI,idxNewJ,
                          m_Mean[0],m_Var[0],m_Age[0],m_Mean_Temp[0],m_Var_Temp[0],m_Age_Temp[0],
                          m_Mean[1],m_Var[1],m_Age[1],m_Mean_Temp[1],m_Var_Temp[1],m_Age_Temp[1])) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
    }
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
    errNum |= clSetKernelArg(kernel, 13, sizeof(cl_mem), &memObjects[13]);
    errNum |= clSetKernelArg(kernel, 14, sizeof(cl_mem), &memObjects[14]);
    errNum |= clSetKernelArg(kernel, 15, sizeof(cl_mem), &memObjects[15]);
    size_t globalWorkSize[1] = { (size_t)modelWidth * (size_t)modelHeight };
    size_t localWorkSize[1] = {10};
t4=clock();

t5=clock();
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
        globalWorkSize, localWorkSize,
        0, NULL, NULL);	
t6=clock();

t7=clock();
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[7], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Mean_Temp[0],
        0, NULL, NULL);
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[8], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Var_Temp[0],
        0, NULL, NULL);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[9], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Age_Temp[0],
        0, NULL, NULL);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[13], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Mean_Temp[1],
        0, NULL, NULL);
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[14], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Var_Temp[1],
        0, NULL, NULL);
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[15], CL_TRUE,
        0, modelWidth * modelHeight * sizeof(float), m_Age_Temp[1],
        0, NULL, NULL);

    for (int i = 0; i < 16; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
t8=clock();
t2=clock();
// double t=t2-t1;
// double tt=t4-t3;
// double ttt=t6-t5;
// double tttt=t8-t7;
// std::cout<<"opencl:"<<t/CLOCKS_PER_SEC<<" "<<"cpu2gpu:"<<tt/CLOCKS_PER_SEC<<" "<<"cal:"<<ttt/CLOCKS_PER_SEC<<" "<<"gpu2cpu:"<<tttt/CLOCKS_PER_SEC<<std::endl;
}

	void update(const Mat& pOutputImg) {
		// clock_t t1,t2,t3,t4,t5,t6;
		// cv::cvtColor(pOutputImg, pOutputImg, cv::COLOR_BGR2GRAY);
		int curModelWidth = modelWidth;
		int curModelHeight = modelHeight;
		mask = Mat::zeros(obsHeight, obsWidth, CV_8UC1);
		//////////////////////////////////////////////////////////////////////////
		// Find Matching Model
		memset(m_ModelIdx, 0, sizeof(int) * modelHeight * modelWidth);
        // t1=clock();
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
        // t2=clock();
		// update with current observation
		float obs_mean[NUM_MODELS];
		float obs_var[NUM_MODELS];
        // t3=clock();
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
        // t4=clock();
		
		// t5=clock();
		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++) {
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++) {
				// TODO: OPTIMIZE THIS PART SO THAT WE DO NOT CALCULATE THIS (LUT)
				// base (i,j) for this block
				int idx_base_i;
				int idx_base_j;
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