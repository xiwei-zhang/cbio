#ifndef __preprocessing_h
#define __preprocessing_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "TOfft.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

void detectBorderReflect(Mat imBlue, Mat imROI, Mat imout){
	float f = imBlue.cols/512.0;
	int imSizeR[2] = {imBlue.rows/f,512};
	Mat imBR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imROIR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	resize(imBlue,imBR,imBR.size());
	resize(imROI,imROIR,imROIR.size());
	Mat imtempR1 = imBR.clone();
	Mat imtempR2 = imBR.clone();
	Mat imtempR3 = imBR.clone();
	Mat imoutR1 = imBR.clone();
	int *iminfoR =  sizeEstimate(imROIR);

	// for some special case , set border of image at 0
	for (int i=0; i<imROIR.cols; i++){
		for (int j=0; j<imROIR.rows; j++){
			if (i==0 || j==0 || i==(imROIR.cols-1) || j==(imROIR.rows-1))
				imROIR.at<uchar>(j,i)=0;
		}
	}

	// I. get reflections,  using Blue channel.
	int meanB = meanValue(imBR,1);
	imCompare(imROIR,0,0,meanB,imBR,imBR);
	//fastMeanFilter(imBR,50,imtempR1);
	blur(imBR,imtempR1,Size(49,49));
	subtract(imBR,imtempR1,imtempR2);
	imCompare(imROIR,0,0,0,imtempR2,imtempR1);

	Erode(imROIR,imtempR2,6,1);
	subtract(imROIR,imtempR2,imtempR3);
	RecUnderBuild(imtempR1,imtempR3,imtempR2,6);
	subtract(imtempR1,imtempR2,imtempR3);

	threshold(imtempR2,imtempR1,3-1,255,0);
	threshold(imtempR3,imoutR1,5-1,255,0);

	Open(imtempR1,imtempR3,6,1);

	lengthOpening(imtempR3,imtempR1,100,512*512,7,1);

	// II. remove not reflections
	//After detection of border reflections, the ROI is divided into 2 parts.
	//Outer circle and Inner circle
	//If the reflections appear in outer circle, it's ok. Otherwise, should be removed
	float inner(0.0f),outer(0.0f),r;
	fastErode(imROIR,imtempR2,6,iminfoR[0]/10);
	for(int j=0; j<imBR.rows; j++){
		for (int i=0; i<imBR.cols; i++){
			if(imtempR1.at<uchar>(j,i)>0){
				if(imtempR2.at<uchar>(j,i)>0) inner++;
				else outer++;
			}
		}
	}
	if (outer==0) r=1.0f;
	else r=inner/outer;
	//cout<<"r: "<<r<<endl;
	if(r>0.25 && r<=0.5){
		RecUnderBuild(imtempR1,imtempR2,imtempR3,6);
		subtract(imtempR1,imtempR3,imtempR2);
		resize(imtempR2,imout,imout.size());
		threshold(imout,imout,150-1,255,0);
	}
	else if(r>0.5)
		imout = Mat::zeros(imout.rows,imout.cols,CV_8U);
	else {
		resize(imtempR1,imout,imout.size());
		threshold(imout,imout,150-1,255,0);
	}
	//imwrite("imtemp2.png",imtempR2);
	//imwrite("imBR.png",imBR);
	//imwrite("imtemp3.png",imtempR3);

	delete[] iminfoR;
}

void getASF(Mat imin, Mat imASF, Mat imROI, int startSize, int endSize, int nbASF){
	Mat imtemp1 = imin.clone();
	float itv = (endSize-startSize+1)/float(nbASF);
	if (itv<1.0) itv = 1.0;
	imin.copyTo(imASF);
	for (int i=startSize; i<endSize; int(i+=itv)){
		fastClose(imASF,imtemp1,6,i);
		fastOpen(imtemp1,imASF,6,i);
	}
}
#endif