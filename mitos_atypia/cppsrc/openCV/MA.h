#ifndef __MA_h
#define __MA_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "detectROI.h"
#include "TOfft.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;


class HMCand{
public:
	int posit[2];
	list<int> p[2];

	HMCand();
};

HMCand::HMCand(){
	posit[0] = 0;
	posit[1] = 0;
}


void detectMA(Mat imgreenR, Mat imROIR, Mat imASFR, Mat imBorderReflectionR, Mat imBrightPart, vector<Mat> vessels, vector<Mat> vesselProperty, int* imInfoR, int* ODCenter){

	int imSizeR[2] = {imgreenR.rows,imgreenR.cols};
	Mat imtempR1 = imgreenR.clone();
	Mat imtempR2 = imgreenR.clone();
	Mat imtempR3 = imgreenR.clone();
	Mat imMACandi = imgreenR.clone();
	Mat imBN = imgreenR.clone();
	Mat imDarkPart = imgreenR.clone();
	Mat imHMCandi = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imROIRs = imgreenR.clone();
	Erode(imROIR,imROIRs,6,2);

	//========================================================
	// Reconstruction of OD
	Mat imOD=Mat::zeros(imgreenR.rows,imgreenR.cols,CV_8U);
	Mat imMC=Mat::zeros(imgreenR.rows,imgreenR.cols,CV_8U);
	int ODCenterL[2]={-1,-1}, MCCenterL[2]={-1,-1};
	if(ODCenter[0]!=-1){
		float f = imgreenR.cols/512.0f;
		ODCenterL[0] = ODCenter[0]*f; ODCenterL[1] = ODCenter[1]*f;
		MCCenterL[0] = ODCenter[2]*f; MCCenterL[1] = ODCenter[3]*f;
		Point center(ODCenter[0]*f,ODCenter[1]*f);
		circle(imOD,center,imInfoR[1]/3*2,255,-1);
		center.x = MCCenterL[0];
		center.y = MCCenterL[1]; 
		circle(imMC,center,imInfoR[1]/4,255,-1);
	}
	imwrite("imOD.png",imOD);
	imwrite("imMC.png",imMC);
	cout<<"Large ODCenter: "<<ODCenterL[0]<<" "<<ODCenterL[1]<<endl;
	cout<<"Large MCCenter: "<<MCCenterL[0]<<" "<<MCCenterL[1]<<endl;
	//========================================================

	//========================================================
	// Bright noise
	int meanV = meanValue(imgreenR,1);// fill background
	imCompare(imROIR,0,0,meanV,imgreenR,imtempR1);
	fastMeanFilter(imtempR1,imInfoR[1]/3*2,imtempR2);
	subtract(imgreenR,imtempR2,imtempR1);
	fastMeanFilter(imtempR1,imInfoR[3]*3,imtempR2);
	imCompare(imROIR,0,0,0,imtempR2,imBN);

	imtempR3= Mat::zeros(imtempR1.rows,imtempR1.cols,CV_8U);
	Point center(imgreenR.cols/2,imgreenR.rows/2);
	circle(imtempR3,center,imInfoR[1],255,-1);
	imSup(imtempR3,imOD,imtempR1);
	RecUnderBuild(imgreenR,imtempR1,imtempR2,6);
	subtract(imgreenR,imtempR2,imtempR3);
	RecUnderBuild(imtempR3,imBorderReflectionR, imtempR1,6);
	threshold(imtempR1,imtempR2,4,255,0);
	imSup(imtempR2,imBorderReflectionR,imBorderReflectionR);
	imwrite("imBorderRefL.png",imBorderReflectionR);
	imwrite("imBN.png",imBN);
	//========================================================


	//========================================================
	// Get candidates
	subtract(vessels[2],vessels[0],imtempR1);
	imCompare(imROIRs,0,0,0,imtempR1,imtempR1);
	// remove od region
	RecUnderBuild(imtempR1,imOD,imtempR3,6);
	subtract(imtempR1,imtempR3,imMACandi);
	imwrite("imMACandi.png",imMACandi);
	//========================================================


	//========================================================
	// Re-segment vessel (take large structure into account)
	
	//threshold(imBrightPart,imtempR1,3,255,0); // original version 3
	//binAreaSelection(imtempR1,imtempR2,6,10);
	//subtract(imtempR1,imtempR2,imtempR3);
	//fastDilate(imtempR3,imtempR2,6,4);
	//imCompare(imtempR2,255,0,0,imgreenR,imtempR1);
	//RecUnderBuild(imgreenR,imtempR1,imtempR3,6);
	//imwrite("BNimtemp3.png",imtempR3);
	//-----
	// 01 23 new
	imCompare(imBN,3,1,0,imgreenR,imtempR1);
	int tempV = meanValue(imtempR1,1);
	imCompare(imBN,3,1,tempV/2,imgreenR,imtempR3);
	//-----


	//fastErode(imgreenR,imtempR1,6,5);
	//fastDilate(imtempR1,imtempR2,6,6);
	//imInf(imtempR2,imgreenR,imtempR3);

	imCompare(imOD,255,0,meanV/2,imtempR3,imtempR3);
	imCompare(imROIR,0,0,meanV/2,imtempR3,imtempR3);
	imCompare(imBorderReflectionR,255,0,1,imtempR3,imtempR3);
	
	imwrite("imtemp3.png",imtempR3);

	fastMeanFilter(imtempR3,100,imtempR2);
	subtract(imtempR2,imtempR3,imtempR1);

	//Dilate(imASFRL,imtempR1,6,2);
	//Erode(imtempR1,imtempR2,6,3);
	//imSup(imtempR2,imASFRL,imtempR3);

	subtract(imtempR1,imOD,imtempR1);
	subtract(imtempR1,imBorderReflectionR,imtempR1);

	imInf(imROIRs,imtempR1,imDarkPart);
	//new ----
	imCompare(imBN,3,1,0,imDarkPart,imDarkPart);
	

	imwrite("imtempDarkpart.png",imDarkPart);
	float nbp(0),temp(0);
	for (int j=0; j<vessels[0].rows; j++){
		for (int i=0; i<vessels[0].cols; i++){
			if (vesselProperty[0].at<uchar>(j,i)==0) continue; // no vessel, continue
			if (vesselProperty[2].at<uchar>(j,i)>=1 && vesselProperty[2].at<uchar>(j,i)<=10){
				nbp++;
				temp+=imDarkPart.at<uchar>(j,i);
			}
		}
	}
	int meanVesselV;
	if (nbp==0) {
		cout<<"no vessel, set default: 5"<<endl;
		meanVesselV = 5;
	}
	else {
		meanVesselV = round(temp/nbp);
		cout<<"mean vessel intensity: "<<meanVesselV<<endl;
	}
	//meanVesselV = 5;
	Close(imDarkPart,imtempR1,6,1);
	imtempR1.copyTo(imDarkPart);
	threshold(imDarkPart,imtempR1,meanVesselV,255,0);
	imSup(imtempR1,vessels[0],imtempR2);
	binAreaSelection(imtempR2,imtempR1,6,50);
	subtract(imtempR2,imtempR1,imtempR2);
	imwrite("imVessel_big.png",imtempR2);
	
	imSup(imDarkPart,imASFR,imtempR3);
	imwrite("imASF_final.png",imtempR3);

	vesselProperty.clear();
	vesselProperty = vesselAnalyse(imtempR2,imROIR,imtempR3,imInfoR);

	imwrite("imSK2.png",vesselProperty[0]);
	imwrite("imVCut2.png",vesselProperty[1]);
	imwrite("imVWidth2.png",vesselProperty[2]);
	imwrite("imVOrient2.png",vesselProperty[3]);
	imwrite("imVInt2.png",vesselProperty[4]);
	//========================================================





	//========================================================
	// Jan. 31 new
	int C_maxWidth = 11;

	// 1. get candidate points
	for (int j=0; j<vesselProperty[0].rows; j++){
		for (int i=0; i<vesselProperty[0].cols; i++){
			if (vesselProperty[1].at<uchar>(j,i)!=2) continue; // no vessel, continue
			if (imOD.at<uchar>(j,i)>0) continue; // in the region of OD, continue
			if (vesselProperty[4].at<uchar>(j,i)<=6) continue; // too dark, continue
			if (vesselProperty[2].at<uchar>(j,i)>C_maxWidth){
				imHMCandi.at<uchar>(j,i)=255;
			}
		}
	}
	imwrite("imHMCandi.png",imHMCandi);

	// 2. Max tree analysis, to get CC for the candi points
	int *hist = histogram(imtempR3);
	int lenH = hist[257]+1;
	Mat imstate = Mat::zeros( imtempR3.rows, imtempR3.cols, CV_32S);
	Mat imtemp = Mat::zeros( imtempR3.rows, imtempR3.cols, CV_8U);
	subtract(imstate,2,imstate); // initiate to -2
	mxt maxTree(imtempR3,imstate);
	int h(0);
	maxTree.flood_h(h,imtempR3, imstate, 6);

	layer **node = new layer* [lenH];
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [maxTree.Nnodes[i]];
	}
	getRelations(maxTree,node,imtempR3,imstate,lenH,0);

	int hh,ii,hp,ip,x,y,xx,yy;
	queue<int> q[2];
	int se=6;
	imtempR1 = Mat::zeros(imtempR1.rows, imtempR1.cols,CV_8U);
	imtempR2 = Mat::zeros(imtempR1.rows, imtempR1.cols,CV_8U);
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	for (int j=0; j<vesselProperty[0].rows; j++){
		for (int i=0; i<vesselProperty[0].cols; i++){
			if (imHMCandi.at<uchar>(j,i)==255 && imtempR2.at<uchar>(j,i)==0){
				HMCand candi;
				candi.posit[0] = i;
				candi.posit[1] = j;
				q[0].push(i);q[1].push(j);
				candi.p[0].push_back(i);candi.p[1].push_back(j);
				imtempR2.at<uchar>(j,i)=1;
				hh = imtempR3.at<uchar>(j,i);
				ii = imstate.at<int>(j,i);
				while(1){
					while(!q[0].empty()){
						xx = q[0].front();
						yy = q[1].front();
						q[0].pop();
						q[1].pop();

						int mod = y%2;
						for (int k=0; k<se; ++k){
							if (mod==0){
								x = xx + se_even[k][1];
								y = yy + se_even[k][0];
							}
							else{
								x = xx + se_odd[k][1];
								y = yy + se_odd[k][0];
							}
							if (x<0 || x>=imtempR1.cols || y<0 || y>=imtempR1.rows) continue;
							if (imtempR3.at<uchar>(y,x)>=hh && imtempR2.at<uchar>(y,x)==0){
								q[0].push(x);q[1].push(y);
								candi.p[0].push_back(x);candi.p[1].push_back(y);
								imtempR2.at<uchar>(y,x)=1;
							}
						}
					}
				}
			}
		}
	}
	//========================================================




	////========================================================
	//// Detection of Hemorrhages
	//int C_maxWidth = 11;
	//int C_medianWidth = 8;
	//int C_intens = 17;
	//int x,y;
	//int count_vw, count_endP, count_cnntP, vLable ;
	//
	//// Labling vessel
	//Mat imSkLabel(imtempR1.rows,imtempR1.cols,CV_32S);
	//Label(vesselProperty[0],imSkLabel,6);

	//// Segment small isolated vessel
	//// 2 part: 1.short vessel with high intensity; 2. middle length vessel without intensity constrain
	//threshold(vesselProperty[4],imtempR2,1,255,0);
	//binAreaSelection(imtempR2,imtempR1,6,30);
	//imCompare(vesselProperty[4],15,2,0,imtempR1,imtempR1);

	//binAreaSelection(vesselProperty[0],imtempR2,6,100);

	//imSup(imtempR1,imtempR2,imtempR1);
	//imwrite("imtemp1.png",imtempR1);


	//// Analyse each pixel in skeleton 
	//for (int j=0; j<vesselProperty[0].rows; j++){
	//	for (int i=0; i<vesselProperty[0].cols; i++){
	//		if (vesselProperty[1].at<uchar>(j,i)!=2) continue; // no vessel, continue
	//		if (imOD.at<uchar>(j,i)>0) continue; // in the region of OD, continue
	//		if (vesselProperty[4].at<uchar>(j,i)<=6) continue; // too dark, continue
	//		
	//		// count 3 types of point:
	//		// end points, connecting points, and wide vessel points
	//		if (vesselProperty[2].at<uchar>(j,i)>C_medianWidth){
	//			count_vw=0; count_cnntP=0; count_endP=0; vLable=imSkLabel.at<int>(j,i);
	//			for (int n=-imInfoR[2]; n<=imInfoR[2]; n++){
	//				for (int m=-imInfoR[2]; m<=imInfoR[2]; m++){
	//					x=i+m; y=j+n;
	//					if (x<0 || x>=imgreenR.cols || y<0 || y>=imgreenR.rows) continue;
	//					if (imSkLabel.at<int>(y,x)!=vLable) continue; //not belong to the same vessel
	//					if (vesselProperty[1].at<uchar>(y,x)==1) count_endP++;
	//					if (vesselProperty[2].at<uchar>(y,x)>(C_maxWidth+1)) count_vw++;
	//					if (vesselProperty[1].at<uchar>(y,x)==3) count_cnntP++;
	//				}
	//			}
	//		}

	//		int xx=854, yy=74;
	//		if(i==xx && j==yy) cout<<"FFFFF "<<(int)vesselProperty[2].at<uchar>(j,i)<<" "<<count_cnntP<<" "<<count_endP<<" "<<count_vw<<endl;
	//
	//		// Very wide points
	//		if (vesselProperty[2].at<uchar>(j,i)>C_maxWidth){
	//			///// decide is it a candidate
	//			// 1. No OD
	//			if (ODCenter[0]==-1){
	//				if(count_endP>0 && count_cnntP==0 && count_vw>1) imHMCandi.at<uchar>(j,i)=255;
	//				if(count_endP>0 && count_vw>10) imHMCandi.at<uchar>(j,i)=255;
	//			}
	//			// 2. If it is close to the OD, which means it could be a segment of big vessel
	//			else if (sqrt(float((ODCenterL[0]-i)*(ODCenterL[0]-i) + (ODCenterL[1]-j)*(ODCenterL[1]-j)))<(imInfoR[0]/3)
	//				|| abs(ODCenterL[0]-i)<imInfoR[1]){
	//				if(count_endP>1 && count_vw>10 && vesselProperty[2].at<uchar>(j,i)>20) {imHMCandi.at<uchar>(j,i)=255;}// cout<<"f1"<<endl;}
	//				if(count_endP>0 && count_cnntP==0 && count_vw>10) {imHMCandi.at<uchar>(j,i)=255; }//cout<<"f2"<<endl;}
	//				if(i==xx && j==yy) cout<<"F1"<<endl;
	//			}
	//			else{
	//				if(i==xx && j==yy) cout<<"F2"<<endl;
	//				if(count_endP>0 && count_vw>10) {imHMCandi.at<uchar>(j,i)=255; if(i==xx && j==yy) cout<<"F2B"<<endl;}//cout<<"f4"<<endl;}
	//			}
	//			if(count_endP>0 && count_cnntP==0 && count_vw>1 && vesselProperty[2].at<uchar>(j,i)>16) {imHMCandi.at<uchar>(j,i)=255; if(i==xx && j==yy) cout<<"F3a"<<endl;}//cout<<"f3"<<endl;}
	//			if(count_endP>1 &&count_cnntP>9) {imHMCandi.at<uchar>(j,i)=255; if(i==xx && j==yy) cout<<"F3b"<<endl;}// cout<<"f5"<<endl;}
	//			if(count_cnntP>3 && count_vw>20 ) {imHMCandi.at<uchar>(j,i)=255;}
	//		}

	//		// small isolated "vessel" maybe hemorrhage
	//		if(i==1007 && j==243) cout<<count_cnntP<<" "<<(int)vesselProperty[2].at<uchar>(j,i)<<endl;
	//		if (imHMCandi.at<uchar>(j,i)==255) continue;
	//		if (vesselProperty[2].at<uchar>(j,i)<C_maxWidth && count_cnntP>0) continue;
	//		if(vesselProperty[2].at<uchar>(j,i)>C_medianWidth && imtempR1.at<uchar>(j,i)>0){
	//			if(vesselProperty[2].at<uchar>(j,i)>(C_maxWidth+1)) imHMCandi.at<uchar>(j,i)=255;
	//			else if (vesselProperty[4].at<uchar>(j,i)>C_intens) imHMCandi.at<uchar>(j,i)=255;
	//			else if (ODCenter[0]==-1) {imHMCandi.at<uchar>(j,i)=255; }//cout<<"f6"<<endl;}
	//			else{
	//				float dist = sqrt(float((ODCenterL[0]-i)*(ODCenterL[0]-i) +(ODCenterL[1]-j)*(ODCenterL[1]-j)));
	//				if (dist>300)  {imHMCandi.at<uchar>(j,i)=255;}//cout<<"f7"<<endl;}
	//			}
	//		}
	//	}
	//}


	//// Final part, reconstruct to CC
	//imwrite("imHMCandiMarker.png",imHMCandi);
	//RecUnderBuild(imtempR3,imHMCandi,imtempR2,6);

	//imwrite("imtemp2.png",imtempR2);

	//int *hist = histogram(imtempR2);
	//int *countV = new int[(hist[257]+1)];
	//memset(countV,0,sizeof(int)*(hist[257]+1));
	//for (int j=0; j<vesselProperty[0].rows; j++){
	//	for (int i=0; i<vesselProperty[0].cols; i++){
	//		if (vesselProperty[0].at<uchar>(j,i)==0) continue; // no vessel, continue
	//		if (imOD.at<uchar>(j,i)>0) continue; // in the region of OD, continue
	//		if (vesselProperty[2].at<uchar>(j,i)<=8){ // new 8 -> 12
	//			countV[imtempR2.at<uchar>(j,i)]++;
	//		}
	//	}
	//}
	//int th1(255),th2(0),Th(0);
	//for (int i=hist[257]; i>0;i--){
	//	cout<<i<<" "<<countV[i]<<"  ";
	//	if (countV[i]<40) th1=i;
	//	else break;
	//}
	//cout<<endl;

	//imCompare(imtempR2,3,2,0,imtempR2,imtempR2);
	//th2 = autoThresholdMaxVar(imtempR2,1);
	//Th = (th1+th2)/2; //max(th1,th2);
	//if (th1>=th2) Th=th1;
	//else Th = round(float(th1+th2))/2;
	//if (Th<4) Th=4;
	//
	//threshold(imtempR2,imHMCandi,Th-1,255,0);
	//RecUnderBuild(imHMCandi,imMC,imtempR2,6);
	//subtract(imHMCandi,imtempR2,imtempR3);
	//cout<<"final threshold: "<<th1<<" "<<th2<<" "<<Th<<endl;

	//// Final remove wrong segmented vessel
	//Dilate(imtempR3,imHMCandi,6,2);
	//Label(imHMCandi,imSkLabel,8);
	//int maxv(0);
	//for (int j=0; j<vesselProperty[0].rows; j++){
	//	for (int i=0; i<vesselProperty[0].cols; i++){
	//		if (imSkLabel.at<int>(j,i) == 0) continue;
	//		if (imSkLabel.at<int>(j,i)>=maxv) maxv=imSkLabel.at<int>(j,i);
	//	}
	//}
	//
	///*
	//if (vesselProperty[1].at<uchar>(y,x)==1) count_endP++;
	//if (vesselProperty[2].at<uchar>(y,x)>(C_maxWidth+1)) count_vw++;
	//if (vesselProperty[1].at<uchar>(y,x)==3) count_cnntP++;
	//*/
	//if (maxv!=0){
	//	maxv++;
	//	int **vlist = new int*[maxv];
	//	for (int i=0; i<maxv; i++){ // 0:count_vw  1:count_endP  2:count_cnntP
	//		vlist[i] = new int[4];
	//		memset(vlist[i],0,sizeof(int)*4);
	//	}
	//	for (int j=0; j<vesselProperty[0].rows; j++){
	//		for (int i=0; i<vesselProperty[0].cols; i++){
	//			if (imSkLabel.at<int>(j,i) == 0) continue;
	//			if (vesselProperty[2].at<uchar>(j,i) == 0) continue;
	//			if (vesselProperty[2].at<uchar>(j,i)<=10) vlist[imSkLabel.at<int>(j,i)][0]++; // new 8 -> 12
	//			if (vesselProperty[1].at<uchar>(j,i)==1) vlist[imSkLabel.at<int>(j,i)][1]++;
	//			if (vesselProperty[1].at<uchar>(j,i)==3) vlist[imSkLabel.at<int>(j,i)][2]++;
	//			if (vesselProperty[2].at<uchar>(j,i)>10) vlist[imSkLabel.at<int>(j,i)][3]++;
	//		}
	//	}
	//	for (int i=0; i<maxv; i++){
	//		cout<<vlist[i][0]<<" "<<vlist[i][1]<<" "<<vlist[i][2]<<" "<<vlist[i][3]<<endl;
	//		if (vlist[i][0]>20 && vlist[i][2]<=3 && vlist[i][1]<=1 && (float(vlist[i][3])/vlist[i][0])<0.65) vlist[i][0]=0;
	//		if (vlist[i][0]>60) vlist[i][0]=0;
	//		else vlist[i][0]=255;
	//	}
	//	imtempR1 = Mat::zeros(imtempR1.rows,imtempR1.cols,CV_8U);
	//	for (int j=0; j<vesselProperty[0].rows; j++){
	//		for (int i=0; i<vesselProperty[0].cols; i++){
	//			if (imSkLabel.at<int>(j,i) == 0) continue;
	//			imtempR1.at<uchar>(j,i) = vlist[imSkLabel.at<int>(j,i)][0];
	//		}
	//	}
	//	imtempR1.copyTo(imHMCandi);
	//}
	//RecUnderBuild(imtempR3,imHMCandi,imtempR1,6);
	//imtempR1.copyTo(imHMCandi);
	//imwrite("imtemp3.png",imtempR3);
	//imwrite("imHMCandi.png",imHMCandi);

	////========================================================

	////RecOverBuild(imgreenR,imtempR1,imtempR2,6);
	////subtract(imtempR2,imgreenR,imtempR3);
	////imwrite("imtemp2.png",imtempR2);
	////imwrite("imtemp3.png",imtempR3);
	cout<<"Hello world"<<endl;
}

#endif