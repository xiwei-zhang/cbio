#ifndef __detectROI_h
#define __detectROI_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

void detectROI(Mat imC, Mat imout, int T){
	// initialization
	int ma(0), mav,map;
	vector<Mat> planes;
	split(imC, planes);
	Mat imtemp1 = Mat::zeros(imout.rows,imout.cols,CV_8U);
	Mat imtemp2(imout.rows,imout.cols,CV_32S);

	// add 3 channels and do a threshold
	for( int y = 0; y < imtemp1.rows; y++ ){
		uchar * line0 = planes[0].ptr<uchar>(y);
		uchar * line1 = planes[1].ptr<uchar>(y);
		uchar * line2 = planes[2].ptr<uchar>(y);
		uchar * line3 = imtemp1.ptr<uchar>(y);
		for( int x = 0; x < imtemp1.cols; x++ ){
			line3[x] = ((line0[x]+line1[x]+line2[x])>T)?255:0;
		}
	}

	// keep the most large elements
	Label(imtemp1,imtemp2,6);

	for( int y = 0; y < imtemp2.rows; y++ ){
		int* line_y = imtemp2.ptr<int>(y);
		for( int x = 0; x < imtemp2.cols; x++ ){
			if (line_y[x]>ma)
				ma = line_y[x];
		}
	}

	if (ma==1) imtemp1.copyTo(imout,imtemp1);
	else{
		int *hist = new int[ma];
		memset(hist,0,sizeof(int)*ma);
		for( int y = 0; y < imtemp2.rows; y++ ){
			int* line_y = imtemp2.ptr<int>(y);
			for( int x = 0; x < imtemp2.cols; x++ ){
				if (line_y[x]==0) continue;
				hist[line_y[x]-1]++;
			}
		}
		mav = 0;
		for(int i=0; i<ma; i++){
			if (hist[i]>mav){
				mav = hist[i];
				map = i;
			}
		}
		compare(imtemp2,(map+1),imout,0);
	}
	threshold(imout,imout,10,255,0);

	// for some special case , set border of image at 0
	for (int i=0; i<imout.cols; i++){
		for (int j=0; j<imout.rows; j++){
			if (i==0 || j==0 || i==(imout.cols-1) || j==(imout.rows-1))
				imout.at<uchar>(j,i)=0;
		}
	}
}

int *sizeEstimate(Mat imROI){
	int *imInfo = new int[4];
	int x;
	bool f(false);

	for (int i=0; i<imROI.cols; i++){
		for (int j=0; j<imROI.rows; j++){
			if (imROI.at<uchar>(j,i)!=0){
				x=i;
				f=true;
				break;
			}
		}
		if (f) break;
	}

	if (x>imROI.cols/4){ // some image totally black
		memset(imInfo,0,sizeof(int)*4);
		return imInfo;
	}

	//// For eophtha 
	imInfo[0] = imROI.cols-2*x;
	imInfo[1] = round(imInfo[0]/5.0);
	imInfo[2] = round(imInfo[0]/888.0*12);
	imInfo[3] = round(imInfo[0]/888.0*9);
	//// For diaret1
	/*imInfo[0] = imROI.cols-2*x;
	imInfo[1] = round(imInfo[0]/6.0);
	imInfo[2] = round(imInfo[0]/96.0);
	imInfo[3] = round(imInfo[0]/140.0);*/
	cout<<imInfo[0]<<" "<<imInfo[1]<<" "<<imInfo[2]<<" "<<imInfo[3]<<endl;
	return imInfo;
	
}

#endif