#ifndef __EX_h
#define __EX_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "detectROI.h"
#include "TOfft.h"
#include "ccAnalyse.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;


void getSaturation(vector<Mat> imC, Mat imSaturation){
	/************************************************************************/
	/*  Get saturation channel                                              */
	/************************************************************************/
	imSaturation = Mat::zeros(imC[0].rows,imC[0].cols,CV_8U);
	priority_queue<int> temp;  // temp is priority queue, the first element is always the bigest one
	int s,max;
	for (int j=0; j<imC[0].rows; j++){
		for (int i=0; i<imC[0].cols; i++){
			for (int k=0; k<3; k++){
				temp.push(imC[k].at<uchar>(j,i));
			}
			max = temp.top();
			temp.pop();
			temp.pop();
			if (max==0) s=0;
			else s = (int)((max-temp.top())/(float)max*255);
			imSaturation.at<uchar>(j,i) = s;
			temp.pop();
		}
	}
	imwrite("imSaturation.png",imSaturation);
}


void getParabola(Mat imgreen, Mat imROI, Mat imOpticDisc, Mat imRefZone, Mat imRefZoneL,  vector<Mat> vessels, vector<Mat> vesselProperty, int* ODCenterS, int* imInfo ){
	/************************************************************************/
	/* Detect reflection zone, a double parabola pass optic disc                                                                     */
	/************************************************************************/
	int ODCenter[2];
	imOpticDisc.setTo(0);
	// a. reconstruct OD image
	float f = imgreen.cols/512.0f;
	if (ODCenterS[0]==-1){
		ODCenter[0]=-1; ODCenter[1]=-1;
		imOpticDisc.setTo(0);
		imRefZone.setTo(0);
		imRefZoneL.setTo(0);
	}
	else{
		ODCenter[0] = ODCenterS[0]*f; ODCenter[1] = ODCenterS[1]*f;
		Point center(ODCenter[0],ODCenter[1]);
		circle(imOpticDisc,center,imInfo[1]/3*2,255,-1);
		imwrite("imOD.png",imOpticDisc);

		float f2 = imInfo[0]/512.0f; // resize (fix diameter of FOV == 512 )
		int ODCenter_[2]={(int)ODCenter[0]/f2, (int)ODCenter[1]/f2};
		int size_[2] = {(int)imgreen.cols/f2, (int)imgreen.rows/f2};

		Mat imROI_ = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imgreen_ = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imVesselWidth_ = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imtmp1 = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imtmp2 = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imtmp3 = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imtmp4 = Mat::zeros(size_[1],size_[0],CV_8U);
		Mat imtemp1 = imgreen.clone();

		resize(vesselProperty[2],imVesselWidth_,imROI_.size());
		resize(imROI,imROI_,imROI_.size());
		resize(imgreen,imgreen_,imgreen_.size());
		int *imInfo_ = sizeEstimate(imROI_);

		// Select vessel skeleton by its width
		imCompare(imVesselWidth_,7,3,255,0,imtmp1);
		imCompare(imVesselWidth_,9,3,255,0,imtmp2);
		fastDilate(imtmp1,imtmp3,6,imInfo_[2]/3);
		fastDilate(imtmp2,imtmp4,6,imInfo_[2]/2);
		imSup(imtmp3,imtmp4,imtmp3);

		// Fit parabola
		int la[7] = {50,75,100,200,300,400,500}, v1[7]={0,0,0,0,0,0,0},v2[7]={0,0,0,0,0,0,0};
		int maxV(0),maxP(0),Vn, LR=0;
		float x,y1,y2;
		for (int k=0; k<7; k++){
			imtmp1 = Mat::zeros(imtmp1.rows,imtmp1.cols,CV_8U);
			for (int i=0; i<imtmp1.cols*5; i++){
				x = i/5.0f;
				if (x>ODCenter_[0]) continue;
				if (abs(x-ODCenter_[0])<size_[0]){
					y1 = round(sqrt(la[k]*abs(x-ODCenter_[0])) + ODCenter_[1]);
					y2 = round(-sqrt(la[k]*abs(x-ODCenter_[0])) + ODCenter_[1]);
					if (y1>=0 && y1<imtmp1.rows && x>=0 && x<imtmp1.cols) imtmp1.at<uchar>(int(y1),int(x))=255;
					if (y2>=0 && y2<imtmp1.rows && x>=0 && x<imtmp1.cols) imtmp1.at<uchar>(int(y2),int(x))=255;
				}
			}
			fastDilate(imtmp1,imtmp2,6,imInfo_[0]/15);
			imInf(imtmp2,imtmp3,imtmp1);
			for (int j=0; j<imtmp1.rows; j++){
				for (int i=0; i<imtmp1.cols; i++){
					if(imtmp1.at<uchar>(j,i)>0) v1[k]++;
				}
			}
		}
		for (int k=0; k<7; k++){
			imtmp1 = Mat::zeros(imtmp1.rows,imtmp1.cols,CV_8U);
			for (int i=0; i<imtmp1.cols*5; i++){
				x = i/5.0f;
				if (x<=ODCenter_[0]) continue;
				if (abs(x-ODCenter_[0])<size_[0]){
					y1 = round(sqrt(la[k]*abs(x-ODCenter_[0])) + ODCenter_[1]);
					y2 = round(-sqrt(la[k]*abs(x-ODCenter_[0])) + ODCenter_[1]);
					if (y1>=0 && y1<imtmp1.rows && x>=0 && x<imtmp1.cols) imtmp1.at<uchar>(int(y1),int(x))=255;
					if (y2>=0 && y2<imtmp1.rows && x>=0 && x<imtmp1.cols) imtmp1.at<uchar>(int(y2),int(x))=255;
				}
			}
			fastDilate(imtmp1,imtmp2,6,imInfo_[0]/15);
			imInf(imtmp2,imtmp3,imtmp1);
			for (int j=0; j<imtmp1.rows; j++){
				for (int i=0; i<imtmp1.cols; i++){
					if(imtmp1.at<uchar>(j,i)>0) v2[k]++;
				}
			}
		}
		for (int i=0; i<7; i++){
			if (maxV<=v1[i]){maxV=v1[i]; maxP=i; LR=0;}
		}
		for (int i=0; i<7; i++){
			if (maxV<=v2[i]){maxV=v2[i]; maxP=i; LR=1;}
		}
		cout<<maxV<<" "<<maxP<<" "<<LR<<endl;


		// Restore the parabola to original image
		imtmp1 = Mat::zeros(imtmp1.rows,imtmp1.cols,CV_8U);
		for (int i=0; i<imtmp1.cols*5; i++){
			x = i/5.0f;
			if (abs(x-ODCenterS[0])<imtmp1.cols){
				y1 = round(sqrt(la[maxP]*abs(x-ODCenterS[0])) + ODCenterS[1]);
				y2 = round(-sqrt(la[maxP]*abs(x-ODCenterS[0])) + ODCenterS[1]);
				if (y1>=0 && y1<imtmp1.rows && x>=0 && x<imtmp1.cols) imtmp1.at<uchar>(int(y1),int(x))=255;
				if (y2>=0 && y2<imtmp1.rows && x>=0 && x<imtmp1.cols) imtmp1.at<uchar>(int(y2),int(x))=255;
			}
		}
		int f(0);
		float PE;
		for (int i=0; i<imtmp1.cols; i++){
			if (imtmp1.at<uchar>(0,i)!=0){
				PE = float(i)/imtmp1.cols;
				f = 1;
				break;
			}
		}
		if (f==0){
			for (int j=0; j<imtmp1.rows; j++){
				if (imtmp1.at<uchar>(j,0)!=0){
					PE = float(j)/imtmp1.rows;
					f = 2;
					break;
				}
			}
		}

		int xe,ye;
		float a;
		if (f==1){
			xe = int(PE*imgreen.cols);
			ye = 0;
		}
		else if(f==2){
			ye = int(PE*imgreen.rows);
			xe = 0;
		}
		a = abs((ye-ODCenter[1])*(ye-ODCenter[1])/(xe-ODCenter[0]));

		imtemp1 = Mat::zeros(imtemp1.rows,imtemp1.cols,CV_8U);
		for (int i=0; i<imtemp1.cols*5; i++){
			x = i/5.0f;
			if (abs(x-ODCenter[0])<imtemp1.cols){
				y1 = round(sqrt(a*abs(x-ODCenter[0])) + ODCenter[1]);
				y2 = round(-sqrt(a*abs(x-ODCenter[0])) + ODCenter[1]);
				if (y1>=0 && y1<imtemp1.rows && x>=0 && x<imtemp1.cols) imtemp1.at<uchar>(int(y1),int(x))=255;
				if (y2>=0 && y2<imtemp1.rows && x>=0 && x<imtemp1.cols) imtemp1.at<uchar>(int(y2),int(x))=255;
			}
		}
		fastDilate(imtemp1,imRefZoneL,6,imInfo[0]/12);
		imwrite("imRefZoneL.png",imRefZoneL);

		imtemp1 = Mat::zeros(imtemp1.rows,imtemp1.cols,CV_8U);
		for (int i=0; i<imtemp1.cols*5; i++){
			x = i/5.0f;
			if (LR==0){
				if (x>ODCenter[0]) continue;
			}
			else{
				if (x<=ODCenter[0]) continue;
			}
			if (abs(x-ODCenter[0])<imtemp1.cols){
				y1 = round(sqrt(a*abs(x-ODCenter[0])) + ODCenter[1]);
				y2 = round(-sqrt(a*abs(x-ODCenter[0])) + ODCenter[1]);
				if (y1>=0 && y1<imtemp1.rows && x>=0 && x<imtemp1.cols) imtemp1.at<uchar>(int(y1),int(x))=255;
				if (y2>=0 && y2<imtemp1.rows && x>=0 && x<imtemp1.cols) imtemp1.at<uchar>(int(y2),int(x))=255;
			}
		}
		fastDilate(imtemp1,imRefZone,6,imInfo[0]/12);

		int leftEnd = ODCenter[0] - imInfo[0]/4;
		int rightEnd = ODCenter[0] + imInfo[0]/4;
		if (leftEnd<0) leftEnd = 0;
		if (rightEnd>=imtemp1.cols) rightEnd = imtemp1.cols - 1;

		for(int i=0; i<imtemp1.cols; i++){
			for (int j=0; j<imtemp1.rows; j++){
				if (i<leftEnd || i>rightEnd)
					imRefZone.at<uchar>(j,i) = 0;
			}
		}
		imwrite("imRefZone.png",imRefZone);

	}
}

void preprocessing(Mat imgreen, Mat imROI, Mat imOpticDisc, Mat imRefZone, Mat imRefZoneL, Mat imVessel, Mat imMeanL, Mat imBrightPart, Mat imFiltered, int* imInfo ){
	Mat imtemp1 = imgreen.clone();
	Mat imtemp2 = imgreen.clone();
	Mat imtemp3 = imgreen.clone();
	Mat imtemp4 = imgreen.clone();
	Mat immark1 = imgreen.clone();
	Mat imVesselMean = imgreen.clone();

	// a. remove high pic noise
	fastErode(imROI,imtemp3,6,imInfo[3]/2);
	subtract(imROI,imtemp3,imtemp4);

	Open(imgreen,imtemp1,6,1);
	subtract(imgreen,imtemp1,imtemp2);
	threshold(imtemp2,imtemp3,30,255,0);

	RecUnderBuild(imtemp3,imtemp4,imtemp2,6);
	subtract(imtemp3,imtemp2,imtemp3);
	binAreaSelection(imtemp3,imtemp2,6,(imInfo[3]/3+1));

	Dilate(imtemp2,imtemp3,8,1);
	imCompare(imtemp3,50,1,imtemp1,imgreen,imgreen);
	imInf(imROI,imgreen,imgreen);

	
	// b. extend border
	fastDilate(imgreen,imtemp1,6,imInfo[2]*2);
	Erode(imROI,imtemp2,6,5);
	imCompare(imtemp2,0,1,imgreen,imtemp1,imgreen);
	
	// c. inpaint
	fastClose(imgreen,imtemp1,6,imInfo[2]);
	fastErode(imtemp1,imtemp2,6,imInfo[2]+1);
	imSup(imtemp2,imgreen,immark1);

	
	//Erode(imROI,imtemp2,6,3);
	//imInf(imtemp2,immark1,imtemp1);
	//imwrite("inpaint.png",imtemp1);


	// d. compensation from local grey level, reflect zones and vessel
	//fastMeanFilter(imVessel,10,imVesselMean);
	blur(imVessel,imVesselMean,Size(10,10));
	imInf(imROI,imVesselMean,imVesselMean);
	int *hist1 = histogram(imMeanL);
	int *hist2 = histogram(imVesselMean);
	imCompare(imVessel,1,1,hist2[257],imVesselMean,imVesselMean);

	imCompare(imBrightPart,5,4,0,imBrightPart,imtemp1);
	int meanOnSet2 = meanValue(imtemp1,1);

	int compns = meanOnSet2 + 10;
	if (compns<15) compns = 15;
	if (compns>30) compns = 30;
	cout<<"COMPNS "<<compns<<endl;

   
	imtemp1.setTo(0);
	float v1,v2;
	for (int j=0; j<imgreen.rows; j++){
		for (int i=0; i<imgreen.cols; i++){
			if (imMeanL.at<uchar>(j,i)>0){
				v1 = float(imMeanL.at<uchar>(j,i))/hist1[257];
				v2 = float(imVesselMean.at<uchar>(j,i))/hist2[257];
				imtemp1.at<uchar>(j,i) = (v1>v2?v1:v2)*compns;
			}
		}
	}
	imCompare(imRefZoneL,0,1,compns,imtemp1,imtemp1);
	imCompare(imRefZone,0,1,60,imtemp1,imtemp1);

	subtract(immark1,imtemp1,imtemp2);
	RecUnderBuild(immark1,imtemp2,imtemp1,6);
	Maxima(imtemp1,imtemp2,6);  // get maximums from filtered image. used to reconstruction

	RecUnderBuild(imtemp2,imRefZone,imtemp1,6);
	subtract(imtemp2,imtemp1,imtemp2);
	imCompare(imtemp2,0,1,255,imtemp1,imtemp3);
	RecUnderBuild(immark1,imtemp3,imFiltered,6);

	imwrite("prefiltered.png",imFiltered); // filtered image

}

void getCandidates(Mat imROI, Mat imgreen, Mat imFiltered, Mat imVessel, Mat imBorderReflection, Mat imOpticDisc, Mat imMeanL, Mat imCandiMain, Mat imCandiSmall, Mat imVAR, Mat imUO, int *imInfo){
	Mat imtemp1 = imFiltered.clone();
	Mat imtemp2 = imFiltered.clone();
	Mat imtemp3 = imFiltered.clone();
	Mat imtemp4 = imFiltered.clone();
	Mat imtemp5 = imFiltered.clone();

	// a.1 Ultimate opening
	clock_t UO1=clock();
	UltimateOpening(imFiltered,imUO,6,imInfo[0]/3,1);
	imwrite("imUO.png",imUO);
	clock_t UO2=clock();
	cout<<"    UUUUUO time: "<<double(diffclock(UO2,UO1))<<"ms"<<endl;

	//fastErode(imROI,imtemp1,6,3);
	//subtract(imROI,imtemp1,imtemp2);
	//imSup(imtemp2,imBorderReflection,imtemp1);  // remove border reflection
	//imSup(imtemp1,imOpticDisc,imtemp1);  // remove OD
	//RecUnderBuild(imUO,imOpticDisc,imtemp2,6);
	//imCompare(imtemp2,imUO,0,0,imUO,imtemp3);

	//// a.2 threshold on UO
	//// defaut th : 2;  for big, imInfo[1]^2 : 5;   for small, (imInfo[2]*2)^2 : 1
	//grayAreaSelection(imtemp3,imtemp1,6,(imInfo[1]/2)*(imInfo[1]/2));
	//threshold(imtemp1,imtemp4,4,255,0);
	//imCompare(imtemp3,imtemp1,0,0,imtemp3,imtemp3);
	//imCompare(imtemp3,2,3,255,imtemp4,imtemp4);

	//binAreaSelection(imtemp3,imtemp1,6,(imInfo[2]*2)*(imInfo[2]*2));
	//subtract(imtemp3,imtemp1,imtemp2);
	//imCompare(imtemp2,1,3,255,imtemp4,imtemp4);

	// b local VAR
	getVAR(imFiltered,imVAR,imInfo[2]);
	imwrite("imVAR.png",imVAR);
	//threshold(imVAR,imtemp1,1,255,0);
	//imwrite("z1.png",imtemp1);
	//imwrite("z4.png",imtemp4);


	// c large mean filter
	// c.1 remove noise
	fastErode(imROI,imtemp1,6,3);
	subtract(imROI,imtemp1,imtemp2);
	imSup(imtemp2,imBorderReflection,imtemp1);  // remove border reflection
	imSup(imtemp1,imOpticDisc,imtemp5);  // remove OD
	RecUnderBuild(imFiltered,imtemp5,imtemp2,6);
	imCompare(imtemp2,imFiltered,0,0,imFiltered,imtemp1);


	// c.2 mean filter
	float f = imInfo[0]/512.0f; // resize (fix diameter of FOV == 512 )
	int size_[2] = {(int)imgreen.cols/f, (int)imgreen.rows/f};
	Mat imROI_ = Mat::zeros(size_[1],size_[0],CV_8U);
	Mat imgreen_ = Mat::zeros(size_[1],size_[0],CV_8U);
	Mat imtmp1 = imgreen_.clone();
	Mat imtmp2 = imgreen_.clone();
	Mat imtmp3 = imgreen_.clone();

	resize(imROI,imROI_,imROI_.size());
	resize(imFiltered,imgreen_,imgreen_.size());
	fastDilate(imgreen_,imtmp1,6,20);
	imCompare(imROI_,1,1,imgreen_,imtmp1,imgreen_);
	//fastMeanFilter(imgreen_,50,imtmp1);
	blur(imgreen_,imtmp1,Size(49,49));
	imInf(imROI_,imtmp1,imtmp2);
	resize(imtmp2,imMeanL,imMeanL.size());

	// c.3 reconstruction to get candidates
	subtract(imtemp1, imMeanL, imtemp3);
	threshold(imtemp3,imtemp2,9,255,0);
	fastDilate(imtemp2,imtemp1,6,imInfo[2]/2);
	imCompare(imtemp1,0,1,0,imgreen,imtemp3);
	RecUnderBuild(imgreen,imtemp3,imtemp4,6);
	subtract(imgreen,imtemp4,imtemp1);

	// c.4 contrast and area selection
	int mm = meanValue(imtemp1,1);
	if (mm>5) mm=5;
	else if (mm<3) mm=3;
	threshold(imtemp1,imtemp2,mm-1,255,0);
	binAreaSelection(imtemp2,imtemp3,6,imInfo[3]);
	subtract(imtemp2,imtemp3,imtemp3);
	imInf(imtemp3,imtemp1,imCandiMain);
	fastErode(imROI,imtemp1,6,imInfo[3]);
	imInf(imtemp1,imCandiMain,imCandiMain);
	imwrite("imCandiMain.png",imCandiMain);

	// d small candidate
	lengthOpening(imgreen,imtemp2,imInfo[3],imInfo[3]*imInfo[3],0,1);
	subtract(imgreen,imtemp2,imtemp3);

	fastDilate(imCandiMain,imtemp1,6,imInfo[2]);
	imCompare(imtemp1,0,1,4,7,imtemp2);
	imCompare(imtemp3,imtemp2,1,255,0,imtemp4);
	binAreaSelection(imtemp4,imtemp1,6,5); //imInfo[2]/3);
	subtract(imtemp4,imtemp1,imtemp2);
	RecUnderBuild(imtemp3,imtemp2,imCandiSmall,6);
	fastErode(imROI,imtemp1,6,imInfo[3]);
	imInf(imtemp1,imCandiSmall,imCandiSmall);
	imSup(imVessel,imtemp5,imtemp5);
	RecUnderBuild(imCandiSmall,imtemp5,imtemp1,6);
	subtract(imCandiSmall,imtemp1,imCandiSmall);
	imwrite("imCandiSmall.png",imCandiSmall);
}


void detectEX(vector<Mat> imC, Mat imROI, Mat imBorderReflection, Mat imBrightPart, Mat imGT, vector<Mat> vessels, vector<Mat> vesselProperty, int* imInfo, int* ODCenterS){

	clock_t begin=clock();
	Mat imgreen = imC[1].clone();
	imC[1].copyTo(imgreen);
	Mat imtemp1 = imgreen.clone();
	Mat imtemp2 = imgreen.clone();
	Mat imtemp3 = imgreen.clone();
	Mat imtemp4 = imgreen.clone();
	Mat imSaturation = imgreen.clone();
	Mat imROIs = imgreen.clone();
	Mat imReflection = imgreen.clone();
	Mat imRefZone = imgreen.clone();
	Mat imRefZoneL = imgreen.clone();
	Mat imOpticDisc = imgreen.clone();
	Mat imMeanL = imgreen.clone(); // mean filter with large kernel size
	Mat imFiltered = imgreen.clone();
	Mat imCandiMain = imgreen.clone();
	Mat imCandiSmall = imgreen.clone();
	Mat imVAR = imgreen.clone();
	Mat imUO = imgreen.clone();

	// restore vessel mask
	Mat imVessel = Mat::zeros(imgreen.rows,imgreen.cols,CV_8U);
	resize(vessels[0],imVessel,imVessel.size());
	threshold(imVessel,imVessel,100,255,0);
	imwrite("imVessel.png",imVessel);

	// Get mean images
	float f = imInfo[0]/512.0f; // resize (fix diameter of FOV == 512 )
	int size_[2] = {(int)imgreen.cols/f, (int)imgreen.rows/f};
	Mat imROI_ = Mat::zeros(size_[1],size_[0],CV_8U);
	Mat imgreen_ = Mat::zeros(size_[1],size_[0],CV_8U);
	Mat imtmp1 = imgreen_.clone();
	Mat imtmp2 = imgreen_.clone();
	Mat imtmp3 = imgreen_.clone();

	resize(imROI,imROI_,imROI_.size());
	resize(imgreen,imgreen_,imgreen_.size());
	fastDilate(imgreen_,imtmp1,6,20);
	imCompare(imROI_,1,1,imgreen_,imtmp1,imgreen_);
	//fastMeanFilter(imgreen_,50,imtmp1);
	blur(imgreen_,imtmp1,Size(49,49));
	imInf(imROI_,imtmp1,imtmp2);
	resize(imtmp2,imMeanL,imMeanL.size());

	// Get saturation channel
	clock_t ex1=clock();
	getSaturation(imC,imSaturation);

	// Restore OD
	getParabola(imgreen, imROI, imOpticDisc, imRefZone, imRefZoneL, vessels, vesselProperty, ODCenterS, imInfo);

	clock_t ex2=clock();
	cout<<" Parabola time: "<<double(diffclock(ex2,ex1))<<"ms"<<endl;

	preprocessing(imgreen, imROI,imOpticDisc, imRefZone, imRefZoneL, imVessel, imMeanL, imBrightPart, imFiltered, imInfo);

	clock_t ex3=clock();
	cout<<" Prepro time: "<<double(diffclock(ex3,ex2))<<"ms"<<endl;

	getCandidates(imROI, imgreen, imFiltered, imVessel, imBorderReflection, imOpticDisc, imMeanL, imCandiMain, imCandiSmall, imVAR, imUO, imInfo);

	clock_t ex4=clock();
	cout<<" getCandidate time: "<<double(diffclock(ex4,ex3))<<"ms"<<endl;

	int *hist = histogram(imCandiMain);
	if (hist[257]==0){
		ofstream myfile;
		myfile.open("EXcandis1.txt");
		myfile.close();
		myfile.open("EXcandis2.txt");
		myfile.close();
		imtemp1.setTo(0);
		imwrite("imEXD.png",imtemp1);
	}
	else{
		const int t_area =round( 0.00587*imInfo[0]-1.28);
		const int t_area2 = imInfo[3]*imInfo[3]/4;

		ccAnalyse(imCandiMain,  imCandiSmall, imgreen, imFiltered,  imROI, imVAR, imUO, imSaturation, imVessel, imGT, "EXcandis1.txt","imEXD.png",t_area);


		/*imtemp1.setTo(0);
		ccAnalyse(imtemp2,  imtemp1, imgreen, imFiltered,  imROI, imVAR, imUO, imSaturation, imVessel, imGT, "EXcandis2.txt","imEXD2.png",t_area2);*/
	}
	clock_t ex5=clock();
	cout<<" ccAnalyse time: "<<double(diffclock(ex5,ex4))<<"ms"<<endl;

	//
	//
	//////----------------------------------------------------
	////// I. Saturation, reflection mask, and blur level
	//int LV_blur;
	//Mat imSaturation = Mat::zeros(imC[0].rows,imC[0].cols,CV_8U);
	//priority_queue<int> temp;  // temp is priority queue, the first element is always the bigest one
	//int s,max;
	//for (int j=0; j<imC[0].rows; j++){
	//	for (int i=0; i<imC[0].cols; i++){
	//		for (int k=0; k<3; k++){
	//			temp.push(imC[k].at<uchar>(j,i));
	//		}
	//		max = temp.top();
	//		temp.pop();
	//		temp.pop();
	//		if (max==0) s=0;
	//		else s = (int)((max-temp.top())/(float)max*255);
	//		imSaturation.at<uchar>(j,i) = s;
	//		temp.pop();
	//	}
	//}
	//imwrite("imSaturation.png",imSaturation);

	//getASF(imSaturation,imtemp1,imROI,1,imInfo[1]/4,5);
	//
	//subtract(imtemp1,imSaturation,imtemp2);
	//fastErode(imROI,imROIs,6,imInfo[2]/4);
	//imInf(imROIs,imtemp2,imtemp2);
	////imwrite("imtemp2.png",imtemp2);

	//threshold(imtemp2,imtemp1,24,255,0);
	//threshold(imSaturation,imtemp2,100,255,1);
	//imSup(imtemp2,imtemp1,imReflection);
	//imInf(imROIs,imReflection,imtemp1);

	//binAreaSelection(imtemp1,imtemp2,6,imInfo[2]*imInfo[2]);
	//subtract(imtemp1,imtemp2,imReflection);
	//imwrite("imReflection.png",imReflection);

	//// blur level
	//int s1(0),s2(0);
	//for (int j=0; j<imC[0].rows; j++){
	//	for (int i=0; i<imC[0].cols; i++){
	//		if(imReflection.at<uchar>(j,i)>0) s1++;
	//		if(imROI.at<uchar>(j,i)>0) s2++;
	//	}
	//}
	//if ((float(s1)/s2) > 0.1) LV_blur=1;
	//else LV_blur=0;

	//cout<<"blur level: "<<LV_blur<<" "<<float(s1)/s2<<endl;
	//////----------------------------------------------------




	//////----------------------------------------------------
	////// III. Pre filter

	clock_t end=clock();
	cout<<"part time: "<<double(diffclock(end,begin))<<"ms"<<endl;
	//////----------------------------------------------------
	cout<<"helloWorld!"<<endl;
}

#endif
