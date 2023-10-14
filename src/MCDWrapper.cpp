#ifndef	_MCDWRAPPER_CPP_
#define	_MCDWRAPPER_CPP_

#include <ctime>
#include <cstring>
#include "MCDWrapper.hpp"
#include "params.hpp"

MCDWrapper::MCDWrapper()
{
}

MCDWrapper::~MCDWrapper()
{
}

void
 MCDWrapper::Init(const UMat& imgGray)
{

	frm_cnt = 0;
	Mat mat_imgGray;
	imgGray.copyTo(mat_imgGray);
	m_LucasKanade.Init(imgGray.cols,imgGray.rows);
	BGModel.init(mat_imgGray);
	imgGrayPrev=imgGray.clone();
}
cv::Point2f MCDWrapper::compensate(cv::Point2f point,double(*h)[9])
{
		       float newW = 0;
		       float newX = 0;
		       float newY = 0;
			   int modelWidth=imgGrayPrev.cols;
			   int modelHeight=imgGrayPrev.rows;
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

std::vector<cv::Rect> MCDWrapper::Run(const UMat & imgGray)
{   clock_t startTime1,endTime1;
	clock_t startTime2,endTime2;
	clock_t startTime3,endTime3;
	clock_t startTime4,endTime4;
	clock_t startTime5,endTime5;
	clock_t startRunTrack,endRunTrack;
	clock_t startGetHomography,endGetHomography;
	clock_t startGetmotionCompensate,endmotionCompensate;


	/*************************
	计算KLT 单应矩阵H 模型补偿
    **************************/
	startTime1 = clock();
	frm_cnt++;
	double h[16][9];
	startRunTrack=clock();
	m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
	endRunTrack=clock();

	m_LucasKanade.GetHomography(h);

    startGetmotionCompensate=clock();
	BGModel.motionCompensate(h);
    endmotionCompensate=clock();

	double time1 =endRunTrack-startRunTrack;
	double time3 =endmotionCompensate-startGetmotionCompensate;

    std::cout<<"RunTrack:"<<time1/CLOCKS_PER_SEC<<" "<<"motionCompensate:"<<time3/CLOCKS_PER_SEC<<std::endl;

	Mat H(3, 3, CV_32F);
    memcpy(H.data,h, sizeof(float) * 9);
	endTime1 = clock();

    /*************************
	模型更新 前景提取
    **************************/
	startTime2 = clock();
	Mat Temp;
	imgGray.copyTo(Temp);
	BGModel.update(Temp);
	endTime2 = clock();

    /*************************
	形态学处理并绘制初始框 
    **************************/
	startTime3 = clock();
	vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
	vector<Rect> bgs_tracked_res;
	vector<Rect>bgs_tracked_list_point;
    vector<Rect>new_bgs_tracked_res;

	UMat thresh;
	thresh=UMat::zeros(imgGray.rows, imgGray.cols, CV_8UC1);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	medianBlur(BGModel.mask, BGModel.mask, 3);
	// blur(BGModel.mask, BGModel.mask, Size(3, 3));
    dilate(BGModel.mask, thresh, kernel, Point(-1, -1), 1);
    erode(thresh, thresh, kernel, Point(-1, -1), 1);
	// medianBlur(thresh, thresh, 3);
    background_res.clear();
	// startTime3 = clock();
    findContours(thresh.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) < 40)
            continue;
        Rect rect = boundingRect(contours[i]);
		background_res.push_back(rect);
    }

   	endTime3 = clock(); 
    /************************************
	tracker追踪 通过运动连续性对运动目标框进行筛选
    *************************************/
	startTime4 = clock();
	//更新所有trackerbox状态
	for (size_t i = 0; i < bgs_tracked_list.size(); i++) {
        bgs_tracked_list[i].update_status=false;
	}
	//初始化tracker list
    if (bgs_tracked_list.empty()) {
        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) < 40)
                continue;
            Rect bbox = boundingRect(contours[i]);
            Tracker tracker(bbox.x, bbox.y - 2, bbox.width, bbox.height + 2);
            bgs_tracked_list.push_back(tracker);
        }
    }else{
	//更新tracker list中匹配框的坐标,将未匹配的加入list
	for (size_t i = 0; i < background_res.size(); i++) {	
			Rect box = background_res[i];
			cv::Point2f new_point=compensate(cv::Point2f(box.x,box.y),h);
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

	//删除tracker list中missCounter >= 3的box
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

	//更新tracker list中box的missCounter,hitCounter,将结果加入bgs_tracked_res
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

	//计算unique删除重复的box
    unique_bgs_tracked_res.clear();
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
	endTime4 = clock();

    /************************************
	计算均方误差追踪 通过相似性对运动目标框进行筛选
    *************************************/
	startTime5 = clock();
	new_bgs_tracked_res.clear();
	for (size_t i = 0; i < unique_bgs_tracked_res.size(); i++) {	
			Rect box = unique_bgs_tracked_res[i];
			cv::Point2f new_point=compensate(cv::Point2f(box.x,box.y),h);
			float newX = new_point.x;
			float newY = new_point.y;
			if (newX < 0) {
                newX = 0;
            }
            if (newX + box.width >= imgGray.cols) {
                newX = imgGray.cols - box.width -1;
            }
            if (newY < 0) {
                newY = 0;
            }
            if (newY+box.height >= imgGray.rows) {
                newY = imgGray.rows - box.height -1;
            }
			if(box.height>=imgGray.rows || box.width>=imgGray.cols)
			{
				continue;
			}
		    bgs_tracked_list_point.push_back(Rect(int(newX), int(newY), box.width, box.height));


			UMat old_bbox_tmp = imgGray(box);
			UMat new_bbox_tmp = imgGrayPrev(Rect(int(newX), int(newY), box.width, box.height));
            Mat old_bbox;
			Mat new_bbox;
            old_bbox_tmp.copyTo(old_bbox);
			new_bbox_tmp.copyTo(new_bbox);

			if (old_bbox.rows != new_bbox.rows || old_bbox.cols != new_bbox.cols) {
                 continue;
            }else {
			double mse=0;
			for(int i=0;i<old_bbox.cols;i++)
			{
				for(int j=0;j<old_bbox.rows;j++)
				{
					uchar a=old_bbox.at<uchar>(j,i);
					uchar b=new_bbox.at<uchar>(j,i);
					double diff=a-b;
					double rr=diff*diff;
					mse+=rr;
				}
			}
			mse=mse/(old_bbox.cols*old_bbox.rows);
                if (mse > 250.0) {
                    new_bgs_tracked_res.push_back(box);
                }
            }
	}
	endTime5 = clock();

    // imgGray.copyTo(imgGrayPrev);
	imgGrayPrev=imgGray.clone();
	double t1 =endTime1-startTime1;
	double t2 =endTime2-startTime2;
	double t3 =endTime3-startTime3;
	double t4 =endTime4-startTime4;
	double t5 =endTime5-startTime5;
    std::cout<<"KLT:"<<t1/CLOCKS_PER_SEC<<" "<<"BGM:"<<t2/CLOCKS_PER_SEC<<" "<<"TH:"<<t3/CLOCKS_PER_SEC<<" "<<"TRACKER:"<<t4/CLOCKS_PER_SEC<<" "<<"MSE:"<<t5/CLOCKS_PER_SEC<<std::endl;
	// std::cout<<"TOTAL:"<<t1/CLOCKS_PER_SEC+t2/CLOCKS_PER_SEC+t3/CLOCKS_PER_SEC+t4/CLOCKS_PER_SEC+t5/CLOCKS_PER_SEC<<std::endl;
	return  new_bgs_tracked_res;

}

#endif				// _MCDWRAPPER_CPP_
