
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
    cl_mem memObjects[9] = {0,0,0,0,0,0,0,0,0};
    cl_int errNum;


 public:
	 ProbModel() {

		for (int i = 0; i < NUM_MODELS; ++i) {
			m_Mean[i] = 0;
			m_Var[i] = 0;
			m_Age[i] = 0;
		} 
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
		}
		Cleanup(context, commandQueue, program, kernel, memObjects);
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

		for (int i = 0; i < NUM_MODELS; ++i) {
			m_Mean[i] = new float[modelWidth * modelHeight];
			m_Var[i] = new float[modelWidth * modelHeight];
			m_Age[i] = new float[modelWidth * modelHeight];
			memset(m_Mean[i], 0, sizeof(float) * modelWidth * modelHeight);
			memset(m_Var[i], 0, sizeof(float) * modelWidth * modelHeight);
			memset(m_Age[i], 0, sizeof(float) * modelWidth * modelHeight);
		}
		// update with homography I
		double *h=new double[16*9];
	    for (int j = 0; j < 16; ++j) {
		  for (int ii = 0; ii < 9; ++ii) {
			h[j*16+9] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
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
	motionCompensate_init(h,pInputImg);
    delete h;
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

    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
        NULL, NULL, &errNum);

    return context;
}

    cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device)
{
    cl_int errNum;
    cl_device_id* devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

  
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);


    commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

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

    bool CreateMemObjects_init(cl_context context, cl_mem memObjects[9],
  		double *h,
        float *m_Mean,
        float *m_Var,
        float *m_Age,
        float *m_Mean1,
        float *m_Var1,
        float *m_Age1,
	    float *image,
		float *mmask
        )
    {
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(double) * 16 * 9, h, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var, NULL);
    memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age, NULL);
    memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Mean1, NULL);
    memObjects[5] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Var1, NULL);
    memObjects[6] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * modelWidth * modelHeight, m_Age1, NULL);
	memObjects[7] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 1920 * 1080, image,NULL);
    memObjects[8] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 1920 * 1080, mmask, NULL);
    return true;
    }   

    bool CreateMemObjects(cl_context context, cl_mem memObjects[9],
  		double *h,
	    float *image,
		float *mmask
        )
    {
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(double) * 16 * 9, h, NULL);
	memObjects[7] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 1920 * 1080, image,NULL);
    memObjects[8] = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 1920 * 1080, mmask, NULL);
    return true;
    }   

void Cleanup(cl_context context, cl_command_queue commandQueue,
    cl_program program, cl_kernel kernel, cl_mem memObjects[9])
{
    for (int i = 0; i < 9; i++) {
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

void motionCompensate_init(double *h,const Mat& pOutputImg) {
int curModelWidth = modelWidth;
	int curModelHeight = modelHeight;
    Mat img2;
    pOutputImg.convertTo(img2, CV_32FC1); 
    int img_length = img2.total() * img2.channels();
    float* image = new float[img_length]();
    float* mmask = new float[img_length]();
    std::memcpy(image, img2.ptr<float>(0), img_length * sizeof(float));
    CreateMemObjects_init(context, memObjects,
	                      h,
                          m_Mean[0],m_Var[0],m_Age[0],
                          m_Mean[1],m_Var[1],m_Age[1],
						  image,mmask);
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[3]);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[4]);
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[5]);
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &memObjects[6]);
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &memObjects[7]);
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &memObjects[8]);
    size_t globalWorkSize[1] = { (size_t)modelWidth * (size_t)modelHeight };
    size_t localWorkSize[1] = {100};
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
        globalWorkSize, localWorkSize,
        0, NULL, NULL);	
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[8], CL_TRUE,
        0, 1920 *1080 * sizeof(float), mmask,
        0, NULL, NULL);

    if (memObjects[0] != 0)
            clReleaseMemObject(memObjects[0]);
	if (memObjects[7] != 0)
            clReleaseMemObject(memObjects[7]);
	if (memObjects[8] != 0)
            clReleaseMemObject(memObjects[8]);
	mask= cv::Mat(pOutputImg.rows, pOutputImg.cols, CV_32FC1, mmask);
	mask.convertTo(mask, CV_8UC1); 
    delete image;
    delete mmask;
}

void motionCompensate(double *h,const Mat& pOutputImg) {
    int curModelWidth = modelWidth;
	int curModelHeight = modelHeight;
    Mat img2;
    pOutputImg.convertTo(img2, CV_32FC1); 
    int img_length = img2.total() * img2.channels();
    float* image = new float[img_length]();
    float* mmask = new float[img_length]();
    std::memcpy(image, img2.ptr<float>(0), img_length * sizeof(float));

    CreateMemObjects(context, memObjects,h,image,mmask);
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[3]);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[4]);
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[5]);
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &memObjects[6]);
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &memObjects[7]);
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &memObjects[8]);
    size_t globalWorkSize[1] = { (size_t)modelWidth * (size_t)modelHeight };
    size_t localWorkSize[1] = {100};
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
        globalWorkSize, localWorkSize,
        0, NULL, NULL);	

	errNum = clEnqueueReadBuffer(commandQueue, memObjects[8], CL_TRUE,
        0, 1920 *1080 * sizeof(float), mmask,
        0, NULL, NULL);

    if (memObjects[0] != 0)
            clReleaseMemObject(memObjects[0]);
	if (memObjects[7] != 0)
            clReleaseMemObject(memObjects[7]);
	if (memObjects[8] != 0)
            clReleaseMemObject(memObjects[8]);
	mask= cv::Mat(pOutputImg.rows, pOutputImg.cols, CV_32FC1, mmask);
	mask.convertTo(mask, CV_8UC1); 

delete image;
delete mmask;
}

void update(const Mat& pOutputImg) {
	 }
};

#endif				// _PROB_MODEL_H_