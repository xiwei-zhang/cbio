#include <stdio.h>
#include <iostream>
#include <time.h>


// opencv lib
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//#incldue "imgproc/include/opencv2/imgproc/imgproc.hpp"
//#incldue "highgui/include/opencv2/highgui/highgui.hpp"

// Teleophta lib
#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "OpticDisc.h"
#include "vessel.h"
#include "detectROI.h"
#include "filter.h"
#include "TOfft.h"
#include "preprocessing.h"
#include "MA.h"
#include "EX.h"


using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;



int main( int argc, char** argv )
{
	clock_t begin=clock();
	const char* imagename;
	if (argc > 1) imagename = argv[1];
	else {
		cout<<"give an image!!"<<endl;
		return 0;
	}

	Mat imin = imread(imagename); // the newer cvLoadImage alternative, MATLAB-style function
	vector<Mat> planes; // Vector is template vector class, similar to STL's vector. It can store matrices too.
	split(imin, planes); // split the image into separate color planes
	Mat imgreen = planes[1];
	Mat imred = planes[2];
	Mat imblue = planes[0];
	Mat imtemp1 = imgreen.clone();
	Mat imtemp2 = imgreen.clone();
	Mat imtemp3 = imgreen.clone();

	Mat imGTc;
	Mat imGT = imgreen.clone();
	if (argc>2){
		imGTc = imread(argv[2]);
		vector<Mat> imGTv;
		split(imGTc, imGTv);
		imGTv[1].copyTo(imGT);
	}
	else imGT = Mat::zeros(imgreen.rows, imgreen.cols, CV_8U);


	// Mitos-atypia
	imwrite("imout.png",imgreen);
	


	// mean filters test //
	//clock_t mean0=clock();
	//fastMeanFilter(imgreen,10,imtemp1);
	//imwrite("fastmean.png",imtemp1);
	//clock_t mean1=clock();
	//cout<<"fastmean: "<<double(diffclock(mean1,mean0))<<"ms"<<endl;


	//blur(imgreen,imtemp2,Size(10,10));
	//imwrite("CVblur.png",imtemp2);
	//clock_t mean2=clock();
	//cout<<"fastmean: "<<double(diffclock(mean2,mean1))<<"ms"<<endl;
	//UltimateOpening(imgreen,imtemp1,6,1000,1);

//	//======================================================================
//	// 0. Segment ROI
//	Mat imROI = imgreen.clone();
//	detectROI(imin, imROI, 10);
//	int *imInfo = sizeEstimate(imROI);
//	//imwrite("imROI.png",imROI);
//	//cout<<"==== 0. Detect ROI completed"<<endl<<endl;
//	clock_t mid1=clock();
//	cout<<"ROI time: "<<double(diffclock(mid1,begin))<<"ms"<<endl;
//	////======================================================================
//
//	////======================================================================
//	// I. preprocessing
//	// I.0 clean background
//	imCompare(imROI,0,0,0,imred,imred);
//	imCompare(imROI,0,0,0,imgreen,imgreen);
//	imCompare(imROI,0,0,0,imblue,imblue);
//	imwrite("imGreen.png",imgreen);
//	imwrite("imBlue.png",imblue);
//	imwrite("imRed.png",imred);
//	vector<Mat> imC;
//	imC.push_back(imred);
//	imC.push_back(imgreen);
//	imC.push_back(imblue);
//
//
//	// I.1 border reflection
//	Mat imBorderReflection;
//	if (1){
//		imBorderReflection = imgreen.clone();
//		detectBorderReflect(imblue, imROI, imBorderReflection);
//		//imwrite("imBorderRef.png",imBorderReflection);
//	}
//	else{
//		imin = imread("imBorderRef.png");
//		vector<Mat> planes2;
//		split(imin, planes2); // split the image into separate color planes
//		imBorderReflection = planes2[1];
//		imwrite("imBorderRef.png",imBorderReflection);
//	}
//
//	clock_t mid2=clock();
//	cout<<"border reflection time: "<<double(diffclock(mid2,mid1))<<"ms"<<endl;
//
//	// I.2 Get residue of ASF (green channel), useful for the rest processing
//	Mat imASF,imBrightPart;
//	if (1){
//		imASF = imgreen.clone();
//		imBrightPart = imgreen.clone();
//		getASF(imgreen,imtemp1,imROI,1,imInfo[2],10);
//		subtract(imtemp1,imgreen,imASF);
//		fastErode(imROI,imtemp3,6,imInfo[2]/4);
//		imInf(imtemp3,imASF,imASF);
//		imwrite("imASF.png",imASF);
//
//		subtract(imgreen,imtemp1,imBrightPart);
//		imInf(imtemp3,imBrightPart,imBrightPart);
//		imwrite("imBrightPart.png",imBrightPart);
//		/*getASF(imtemp1,imtemp2,imROI,imInfo[2]+imInfo[2]*2/5,imInfo[2]*3,4);
//		subtract(imtemp2,imgreen,imASFL);
//		imInf(imtemp3,imASFL,imASFL);
//		imwrite("imASFL.png",imASFL);*/
//	}
//	else{
//		imASF = imgreen.clone();
//		//imASFL = imgreen.clone();
//		imin = imread("imASF.png");
//		vector<Mat> planes2;
//		split(imin, planes2); // split the image into separate color planes
//		planes2[1].copyTo(imASF);
//
//		imin = imread("imBrightPart.png");
//		split(imin, planes2); // split the image into separate color planes
//		planes2[1].copyTo(imBrightPart);
//	}
//
//	clock_t mid3=clock();
//	cout<<"ASF time: "<<double(diffclock(mid3,mid2))<<"ms"<<endl;
//
//
//	// I.3 resize images (taking 1440x960 as reference)
//	float f = imInfo[0]/888.0f;
//	int imSizeR[2] = {(int)imin.rows/f,(int)imin.cols/f};
//	Mat imgreenR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	Mat imblueR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	Mat imredR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	Mat imASFR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	Mat imBrightPartR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	Mat imROIR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	Mat imBorderReflectionR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
//	resize(imgreen,imgreenR,imgreenR.size());
//	resize(imblue,imblueR,imblueR.size());
//	resize(imred,imredR,imredR.size());
//	resize(imROI,imROIR,imROIR.size());
//	resize(imASF,imASFR,imASFR.size());
//	resize(imBrightPart,imBrightPartR,imBrightPartR.size());
//	resize(imBorderReflection,imBorderReflectionR,imBorderReflectionR.size());
//	imC.push_back(imredR);
//	imC.push_back(imgreenR);
//	imC.push_back(imblueR);
//	int *imInfoR =  sizeEstimate(imROIR);
//	cout<<"==== I. Preprocessing completed"<<endl<<endl;
//	//======================================================================
//
//
//	//======================================================================
//	// II. vessel segmentation and analyse
//	vector<Mat> vessels;
//	vector<Mat> vesselProperty;
//	if(1){
//		vessels = segmentVessel(imASFR,imROIR,imInfoR);
//		vesselProperty = vesselAnalyse(vessels[0],imROIR,imASFR,imInfoR);
//		//imwrite("imMainVessel.png",vessels[0]);
//		//imwrite("imAsfDv2Rec.png",vessels[1]);
//		//imwrite("imRecTh.png",vessels[2]);
//		//
//		//imwrite("imSK.png",vesselProperty[0]);
//		//imwrite("imVCut.png",vesselProperty[1]);
//		//imwrite("imVWidth.png",vesselProperty[2]);
//		//imwrite("imVOrient.png",vesselProperty[3]);
//		//imwrite("imVInt.png",vesselProperty[4]);
//
//	}
//	else{
//		Mat imMainVessel = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imSK = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imVWidth = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imVOrient = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imVInt = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imAsfDv2Rec = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imRecTh = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		Mat imVCut = Mat::zeros(imgreenR.rows, imgreenR.cols, CV_8U);
//		vector<Mat> planes3;
//		imin = imread("imMainVessel.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imMainVessel);
//		imin = imread("imSK.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imSK);
//		imin = imread("imVWidth.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imVWidth);
//		imin = imread("imVOrient.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imVOrient);
//		imin = imread("imVInt.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imVInt);
//		imin = imread("imAsfDv2Rec.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imAsfDv2Rec);
//		imin = imread("imRecTh.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imRecTh);
//		imin = imread("imVCut.png");
//		split(imin, planes3);
//		planes3[1].copyTo(imVCut);
//
//		vessels.push_back(imMainVessel);
//		vessels.push_back(imAsfDv2Rec);
//		vessels.push_back(imRecTh);
//
//		vesselProperty.push_back(imSK);
//		vesselProperty.push_back(imVCut);
//		vesselProperty.push_back(imVWidth);
//		vesselProperty.push_back(imVOrient);
//		vesselProperty.push_back(imVInt);
//
//	}
//
//	clock_t mid4=clock();
//	cout<<"Vessel time: "<<double(diffclock(mid4,mid3))<<"ms"<<endl;
//
//	////======================================================================
//
//
//	////======================================================================
//	//// III. Optic Disc localization
//	int *ODCenter;
//	if(1){
//		ODCenter = detectOD(imC,imROI,imBorderReflection,vessels,vesselProperty);
//		cout<<"OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
//		cout<<"==== III. Optic Disc detection completed"<<endl<<endl;
//	}
//	else{
//		ODCenter = new int[2];
//		ODCenter[0]= 319;//198;
//		ODCenter[1]= 153;//179;
//	}
//
//	clock_t mid5=clock();
//	cout<<"OD time: "<<double(diffclock(mid5,mid4))<<"ms"<<endl;
//	////======================================================================
//
//	////======================================================================
//	//// temporaly, for exudates
//
//
//
//	////======================================================================
//
//
//	////======================================================================
//	//// VI.a EX detection
//	detectEX(imC, imROI, imBorderReflection, imBrightPart,imGT, vessels, vesselProperty, imInfo, ODCenter);
//	clock_t mid6=clock();
//	cout<<"Exudate time: "<<double(diffclock(mid6,mid5))<<"ms"<<endl;
//	////======================================================================
//
//
//	//////======================================================================
//	////// VI. MA detection
//	//detectMA(imgreenR,imROIR,imASFR,imBorderReflectionR,imBrightPartR,vessels,vesselProperty,imInfoR,ODCenter);
//	//////======================================================================
//	//
//
//
//
//	clock_t end=clock();
//	cout<<"Total time: "<<double(diffclock(end,begin))<<"ms"<<endl;
//
//
//	//waitKey();
//	//cin.get();

	return 0;
	// all the memory will automatically be released by Vector<>, Mat and Ptr<> destructors.
}
