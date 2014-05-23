#ifndef __vessel_h
#define __vessel_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
vector<Mat> segmentVessel(Mat imASFR, Mat imROIR, int* imInfoR){
	
	Mat imVesselR = imASFR.clone();
	Mat imtempR1 = imASFR.clone();
	Mat imtempR2 = imASFR.clone();
	Mat imout1 = imASFR.clone(); // divided and rec
	Mat imout2 = imASFR.clone(); // thresholded


	//=======================================================
	// pre filtering: removing reflection in the vessel
	fastClose(imASFR,imtempR1,6,2);
	fastOpen(imtempR1,imtempR2,6,3);
	imSup(imtempR2,imASFR,imtempR1);
	subtract(imtempR1,imASFR,imtempR2);
	imCompare(imtempR2,5,3,imtempR1,imASFR,imASFR);
	
	//imwrite("z1.png",imASFR);
	
	// threshold
	imCompare(imROIR,0,1,1,0,imtempR1);
	int s = sum(imtempR1)[0];
	int s2(0),th2(0);
	int *hist = histogram(imASFR);
	for (int i=hist[257]; i>hist[256]; i--) {
		s2 += hist[i];
		if (float(s2)/s>0.13) {th2 = i+1; break;}
	}
	//cout<<"threshold area: "<<th2<<endl;
	int th1 =5;
	if (th2>th1) th1=th2;
	//cout<<"Threshold at : "<<th1<<endl;

	/*threshold(imASFR,imtempR1,th1,255,0);// old version
	fastErode(imtempR1,imtempR2,6,imInfoR[2]/4);
	RecUnderBuild(imtempR1,imtempR2,imtempR4,6);
	Close(imtempR4,imVesselR,6,2);*/
	//imwrite("vesselMask.png",imVesselR);
	//----- 2013 01 07 new -------
	divide(imASFR,2,imtempR1);
	RecUnderBuild(imASFR,imtempR1,imout1,6);
	threshold(imout1,imout2,th1-1,255,0);
	lengthOpening(imout2,imtempR1,imInfoR[2]*1.5,imInfoR[2]*imInfoR[2],2,1);
	Close(imtempR1,imVesselR,6,2);
	//----------------------------

	vector<Mat> imVessels;
	imVessels.push_back(imVesselR);
	imVessels.push_back(imout1);
	imVessels.push_back(imout2);
	return(imVessels);
	//=======================================================
}



vector<Mat> vesselAnalyse(Mat imVesselR, Mat imROIR, Mat imASFR, int* imInfoR){

	int imSizeR[2] = {imVesselR.rows, imVesselR.cols};

	Mat imSK = imVesselR.clone();
	Mat imtempR1 = imVesselR.clone();
	Mat imtempR2 = imVesselR.clone();
	Mat imtempR3 = imVesselR.clone();
	Mat imtempR4 = imVesselR.clone();

	// threshold
	imCompare(imROIR,0,1,1,0,imtempR1);
	int s = sum(imtempR1)[0];
	int s2(0),th2(0);
	int *hist = histogram(imASFR);
	for (int i=hist[257]; i>hist[256]; i--) {
		s2 += hist[i];
		if (float(s2)/s>0.13) {th2 = i+1; break;}
	}
	int th1 =5;
	if (th2>th1) th1=th2;
	int thMax = autoThresholdMaxVar(imASFR,1);
	//cout<<"Threshold max : "<<thMax<<" "<<th1<<endl;
	//=======================================================
	// skeleton
	Distance(imVesselR,imtempR2,4);
	Maxima(imtempR2,imtempR4,4);
	SkeletonWithAnchor(imVesselR,imtempR4,imSK,6);
	//imwrite("imSkeleten.png",imSK);
	//=======================================================
	

	//=======================================================
	// skeleton vessel analyze
	for (int j=0; j<imSizeR[0];++j){
		for (int i=0; i<imSizeR[1];++i){
			if (imSK.at<uchar>(j,i)>0)
				imtempR1.at<uchar>(j,i)=3;  // imout initialized by sk
			else
				imtempR1.at<uchar>(j,i) = 0;
		}
	}

	int **se_even = nl(6,1);
	int **se_odd = nl(6,0);

	// look up table for Hex dist at 2
	int se_even2[12][2], se_odd2[12][2]; 
	se_even2[0][1] = 2; se_even2[0][0] = 0; // col, row 
	se_even2[1][1] = 1; se_even2[1][0] = -1;
	se_even2[2][1] = 1;	se_even2[2][0] = -2;
	se_even2[3][1] = 0;	se_even2[3][0] = -2;
	se_even2[4][1] = -1;se_even2[4][0] = -2;
	se_even2[5][1] = -2;se_even2[5][0] = -1;
	se_even2[6][1] = -2;se_even2[6][0] = 0;
	se_even2[7][1] = -2;se_even2[7][0] = 1;
	se_even2[8][1] = -1;se_even2[8][0] = 2;
	se_even2[9][1] = 0; se_even2[9][0] = 2;
	se_even2[10][1] = 1;se_even2[10][0] = 2;
	se_even2[11][1] = 1;se_even2[11][0] = 1;

	se_odd2[0][1] = 2;	se_odd2[0][0] = 0;
	se_odd2[1][1] = 2;	se_odd2[1][0] = -1;
	se_odd2[2][1] = 1;	se_odd2[2][0] = -2;	
	se_odd2[3][1] = 0;	se_odd2[3][0] = -2;
	se_odd2[4][1] = -1;	se_odd2[4][0] = -2;
	se_odd2[5][1] = -1;	se_odd2[5][0] = -1;
	se_odd2[6][1] = -2;	se_odd2[6][0] = 0;
	se_odd2[7][1] = -1;	se_odd2[7][0] = 1;
	se_odd2[8][1] = -1;	se_odd2[8][0] = 2;
	se_odd2[9][1] = 0;	se_odd2[9][0] = 2;
	se_odd2[10][1] = 1;	se_odd2[10][0] = 2;
	se_odd2[11][1] = 2;	se_odd2[11][0] = 1;

	// look up table of end points and normal points
	int table[64] = {0};
	table[0] = 1;
	table[1] = 1;
	table[2] = 1;
	table[3] = 3;
	table[4] = 1;
	table[6] = 3;
	table[8] = 1;
	table[12] = 3;
	table[16] = 1;
	table[24] = 3;
	table[32] = 1;
	table[33] = 3;
	table[48] = 3;

	table[5] = 2;
	table[9] = 2;
	table[10] = 2;
	table[17] = 2;
	table[18] = 2;
	table[20] = 2;
	table[34] = 2;
	table[36] = 2;
	table[40] = 2;

	int index, x, y,mod;
	queue<int> temp, bifX, bifY;
	bool findEnd;
	// first pass, detect endpoints 
	for(int j=0; j<imSizeR[0]; ++j){
		mod = j%2;
		for(int i=0; i<imSizeR[1]; ++i){
			if (imtempR1.at<uchar>(j,i)==0) continue;
			findEnd=false;
			index = 0;
			for (int k=0; k<6; ++k){
				if (mod==0){
					x = i + se_even[k][1];
					y = j + se_even[k][0];
				}
				else{
					x = i + se_odd[k][1];
					y = j + se_odd[k][0];
				}
				if (x<0 || x>=imSizeR[1] || y<0 || y>=imSizeR[0]) continue;
				if (imtempR1.at<uchar>(y,x)>0){
					index = index + (int)pow(2.0f,k);
				}
				if (imtempR1.at<uchar>(y,x)==1) findEnd=true;
			}
			if (table[index]==2)
				imtempR1.at<uchar>(j,i)=2;
			else if(table[index]==1)
				imtempR1.at<uchar>(j,i)=1;
			if(table[index]==3 || imtempR1.at<uchar>(j,i)==3 || imtempR1.at<uchar>(j,i)==1){
				float D6[6]={0};
				int nbP(0);
				for (int k=0; k<12; k++){
					if (mod==0){
						x = i+se_even2[k][1];
						y = j+se_even2[k][0];
					}
					else{
						x = i+se_odd2[k][1];
						y = j+se_odd2[k][0];
					}
					if (x<0 || x>=imSizeR[1] || y<0 || y>=imSizeR[0]) continue;
					if (imtempR1.at<uchar>(y,x)>0){
						nbP++;
						switch (k){
						case 0:
							D6[0]+=0.5; D6[5]+=0.5;
							break;
						case 1:
							D6[0]+=1;
							break;
						case 2:
							D6[0]+=0.5; D6[1]+=0.5;
							break;
						case 3:
							D6[1]+=1;
							break;
						case 4:
							D6[1]+=0.5; D6[2]+=0.5;
							break;
						case 5:
							D6[2]+=1;
							break;
						case 6:
							D6[2]+=0.5; D6[3]+=0.5;
							break;
						case 7:
							D6[3]+=1;
							break;
						case 8:
							D6[3]+=0.5; D6[4]+=0.5;
							break;
						case 9:
							D6[4]+=1;
							break;
						case 10:
							D6[4]+=0.5; D6[5]+=0.5;
							break;
						case 11:
							D6[5]+=1;
							break;
						}
					}
				}
				int edge0(0),edge1(0);
				for (int k=0; k<6; k++){
					if (D6[k]>0) edge0++;
					if (D6[k]>=1) edge1++;
				}
				if (nbP<=1){
					if (findEnd) imtempR1.at<uchar>(j,i)=2;
					else imtempR1.at<uchar>(j,i)=1;
				}
				else if (nbP==2)// 2 points: yes
					imtempR1.at<uchar>(j,i)=2;
				else if(nbP==3 && edge0<=4 && edge1<=1)// 3 points: <= 4 edge and 1 edge >1  yes
					imtempR1.at<uchar>(j,i)=2;
				else if(nbP==3 && edge0<=3 && edge1<=2)// 3 points: <= 3 edge and 2 edge >1  yes
					imtempR1.at<uchar>(j,i)=2;
				else if(nbP==4 && edge0<=4 && edge1<=2)// 4 points: <= 4 edge and 2 edge >1  yes
					imtempR1.at<uchar>(j,i)=2;
				else if(nbP>4)
					imtempR1.at<uchar>(j,i)=2;
			}
		}
	}

	// remove small branch.
	for (int j=0; j<imSizeR[0];++j){
		mod = j%2;
		for (int i=0; i<imSizeR[1];++i){
			if (imtempR1.at<uchar>(j,i)>0 && imtempR1.at<uchar>(j,i)!=2){
				// first circle (dist 1)
				for (int k=0; k<6; ++k){
					if (mod==0){
						x = i + se_even[k][1];
						y = j + se_even[k][0];
					}
					else{
						x = i + se_odd[k][1];
						y = j + se_odd[k][0];
					}
					if (x<0 || x>=imSizeR[1] || y<0 || y>=imSizeR[0]) continue;
					if(imtempR1.at<uchar>(j,i)==1){ // if near to a connect point or an end point, remove
						if (imtempR1.at<uchar>(y,x)==3 || imtempR1.at<uchar>(y,x)==1){
							imtempR1.at<uchar>(j,i)=2; 
							break;
						}
					}
					else if(imtempR1.at<uchar>(j,i)==3){
						if (imtempR1.at<uchar>(y,x)==1){
							imtempR1.at<uchar>(y,x)=2;
						}
						if (imtempR1.at<uchar>(y,x)==3){
							imtempR1.at<uchar>(j,i)=2;
						}
					}
				}
				// second circle (dist 2)
				for (int k=0; k<12; ++k){
					if (mod==0){
						x = i + se_even2[k][1];
						y = j + se_even2[k][0];
					}
					else{
						x = i + se_odd2[k][1];
						y = j + se_odd2[k][0];
					}
					if (x<0 || x>=imSizeR[1] || y<0 || y>=imSizeR[0]) continue;
					if(imtempR1.at<uchar>(j,i)==1){ // if near to a connect point or an end point, remove
						if (imtempR1.at<uchar>(y,x)==3 || imtempR1.at<uchar>(y,x)==1){
							imtempR1.at<uchar>(j,i)=2; 
							break;
						}
					}
					else if(imtempR1.at<uchar>(j,i)==3){
						if (imtempR1.at<uchar>(y,x)==1){
							imtempR1.at<uchar>(y,x)=2;
						}
						if (imtempR1.at<uchar>(y,x)==3){
							imtempR1.at<uchar>(j,i)=2;
						}
					}
				}
			}
		}
	}

	//imwrite("imVesselCut.png",imtempR1);
	//=======================================================


	//=======================================================
	// width measurement
	imtempR2 = Mat::zeros(imtempR1.rows, imtempR1.cols, CV_8U);
	imtempR3 = Mat::zeros(imtempR1.rows, imtempR1.cols, CV_8U);
	imtempR4 = Mat::zeros(imtempR1.rows, imtempR1.cols, CV_8U);
	int i_,j_;
	for(int j=0; j<imSizeR[0]; ++j){
		mod = j%2;
		for(int i=0; i<imSizeR[1]; ++i){
			if (imtempR1.at<uchar>(j,i)==0) continue;
			//if (imtempR1.at<uchar>(j,i)!=2) continue;
			// initialization 
			int len1D[6] = {0}, vOrig = imASFR.at<uchar>(j,i), vDiff(0);
			int meanInt(0);

			for (int k=0; k<6; ++k){
				i_ = i; j_ = j;
				bool f=true;
				while (f){
					if (mod==0){
						x = i_ + se_even[k][1];
						y = j_ + se_even[k][0];
					}
					else{
						x = i_ + se_odd[k][1];
						y = j_ + se_odd[k][0];
					}
					if (x<0 || x>=imSizeR[1] || y<0 || y>=imSizeR[0]) break;
					if (imASFR.at<uchar>(y,x)>=th1) {
						if (abs(vOrig-imASFR.at<uchar>(y,x)) < thMax && imASFR.at<uchar>(y,x)>4){
							len1D[k]++;
							meanInt+=imASFR.at<uchar>(y,x);
							i_ = x;
							j_ = y;
						}
						else f = false;
					}
					else f=false;
				}
			}
			//if (i==1007 && j==243) cout<<len1D[0]<<" "<<len1D[1]<<" "<<len1D[2]<<" "<<len1D[3]<<" "<<len1D[4]<<" "<<len1D[5]<<endl;
			int minW(100),minP,vw[3];
			vw[0] = len1D[0]+len1D[3];
			vw[1] = len1D[1]+len1D[4];
			vw[2] = len1D[2]+len1D[5];
			for (int w=0; w<3; w++){
				if (vw[w]<=minW){
					minW = vw[w];
					minP = w;
				}
			}

			// In the case of bifurcate like "-<" or ">-"
			if ((len1D[0]+len1D[2]+len1D[4])<9){
				len1D[0]=(len1D[0]==0?1:len1D[0]);
				len1D[2]=(len1D[2]==0?1:len1D[2]);
				len1D[4]=(len1D[4]==0?1:len1D[4]);
				if ( (len1D[3]/len1D[0])>=3 && (len1D[5]/len1D[2])>=3 && (len1D[1]/len1D[4])>=3 )
					minW = 5;
			}
			else if ((len1D[1]+len1D[3]+len1D[5])<9){
				len1D[1]=(len1D[1]==0?1:len1D[1]);
				len1D[3]=(len1D[3]==0?1:len1D[3]);
				len1D[5]=(len1D[5]==0?1:len1D[5]);
				if ( (len1D[0]/len1D[3])>=3 && (len1D[2]/len1D[5])>=3 && (len1D[4]/len1D[1])>=3 )
					minW = 5;
			}

			//if (i==755 && j==690) cout<<"DGDG "<<meanInt<<" "<<vw[0]<<" "<<vw[1]<<" "<<vw[2]<<endl;
			if ((vw[0]+vw[1]+vw[2])==0) meanInt = 0;
			else meanInt /= (vw[0]+vw[1]+vw[2]);
			imtempR2.at<uchar>(j,i)=minW;
			imtempR3.at<uchar>(j,i)=minP+1;
			imtempR4.at<uchar>(j,i)=meanInt;
		}
	}
	//imwrite("imVWidth.png",imtempR2);
	//imwrite("imVOrient.png",imtempR3);
	//imwrite("imVInt.png",imtempR4);
	//=======================================================

	// return
	vector<Mat> vOut;
	vOut.push_back(imSK);
	vOut.push_back(imtempR1);
	vOut.push_back(imtempR2);
	vOut.push_back(imtempR3);
	vOut.push_back(imtempR4);
	
	return vOut;

	//=======================================================
	// generate width mask
	/*list<int> widthV;
	int midV,size,interval,meanVI;
	interval = imInfoR[1]/2;
	int res = iminR.rows%interval;
	int lenR = (iminR.rows - res)/interval+2;
	int *coorR = new int[lenR];
	coorR[0] = 0;
	for (int i=1; i<lenR; i++) coorR[i] = coorR[i-1] + interval;
	coorR[lenR-1] = iminR.rows;

	res = iminR.cols%interval;
	int lenC = (iminR.cols - res)/interval+2;
	int *coorC = new int[lenC];
	coorC[0] = 0;
	for (int i=1; i<lenC; i++) coorC[i] = coorC[i-1] + interval;
	coorC[lenC-1] = iminR.cols;

	j_ = 0;
	i_ = 0;
	for (int j=1; j<lenR; j++){
	for (int i=1; i<lenC; i++){
	j_ = coorR[j-1];
	i_ = coorC[i-1];
	meanVI=0;
	for (int n=j_; n<coorR[j]; n++){
	for (int m=i_; m<coorC[i]; m++){
	if (imtempR2.at<uchar>(n,m)>0 && imtempR2.at<uchar>(n,m)<=imInfoR[2]){
	widthV.push_back(imtempR2.at<uchar>(n,m));
	meanVI += imtempR4.at<uchar>(n,m);
	}
	}
	}
	widthV.sort();
	if (widthV.size()==0) {midV = 0; meanVI = 0;}
	else {
	meanVI /= (int)widthV.size();
	size = (int)(widthV.size()+1)/4;
	for (int l=0; l<size; l++){
	widthV.pop_back();
	}
	midV = widthV.back();
	}

	for (int n=j_; n<coorR[j]; n++){
	for (int m=i_; m<coorC[i]; m++){
	imtempR3.at<uchar>(n,m) = midV;
	imtempR1.at<uchar>(n,m) = meanVI;
	}
	}
	i_ = coorC[i];
	j_ = coorC[j];
	widthV.clear();
	}
	}
	imwrite("imWMask.png",imtempR3);
	imwrite("imVIMask.png",imtempR1);*/
	//=======================================================

}

#endif
