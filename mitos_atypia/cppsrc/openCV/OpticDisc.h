#ifndef __OpticDisc_h
#define __OpticDisc_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "detectROI.h"
#include "TOfft.h"
#include "preprocessing.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

int* detectOD(vector<Mat> imC, Mat imROI, Mat imBorderReflection, vector<Mat> vessels, vector<Mat> vesselProperty){
	// 0. Initialization
	int *ODCenter = new int[4];
	ODCenter[0] = -1; ODCenter[1] = -1;  ODCenter[2] = -1;  ODCenter[3] = -1;// initialized by -1
	Mat imRed = imC[0];
	Mat imGreen = imC[1];
	Mat imBlue = imC[2];

	float f = imC[0].cols/512.0;
	int imSizeR[2] = {imC[0].rows/f,512};
	Mat imRedR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U); // im*R means Reduced
	Mat imGreenR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imBlueR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imROIR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imBorderReflectionR = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	resize(imC[0],imRedR,imRedR.size());
	resize(imC[1],imGreenR,imGreenR.size());
	resize(imC[2],imBlueR,imBlueR.size());
	resize(imROI,imROIR,imROIR.size());
	resize(imBorderReflection,imBorderReflectionR,imBorderReflectionR.size());
	int *iminfoR =  sizeEstimate(imROIR);
	vector<Mat> imCR;
	imCR.push_back(imRedR);
	imCR.push_back(imGreenR);
	imCR.push_back(imBlueR);

	Mat imtemp1 = imRedR.clone();
	Mat imtemp2 = imRedR.clone();
	Mat imtemp3 = imRedR.clone();
	Mat imMean3 = imRedR.clone();
	
	int C_area = iminfoR[1]*iminfoR[1]/5*3;
	//cout<<"area critere: "<<C_area<<endl;

	//=======================================================
	// Part 1. Intensity information. Get candidates
	// add 3 channels and do a threshold
	for( int y = 0; y < imMean3.rows; y++ ){
		uchar * line0 = imCR[0].ptr<uchar>(y);
		uchar * line1 = imCR[1].ptr<uchar>(y);
		uchar * line2 = imCR[2].ptr<uchar>(y);
		uchar * line3 = imMean3.ptr<uchar>(y);
		for( int x = 0; x < imMean3.cols; x++ ){
			line3[x] = (line0[x]+line1[x]+line2[x])/3;
		}
	}
	//imwrite("imMean3.png",imMean3);

	// remove vessel
	fastClose(imMean3,imtemp1,6,iminfoR[2]);
	fastOpen(imtemp1,imtemp2,6,iminfoR[2]+1);
	imSup(imtemp2,imMean3,imMean3);
	//imwrite("ODimmean2.png",imMean3);

	// fill background
	int meanV = meanValue(imMean3,1);
	imCompare(imROIR,0,0,meanV,imMean3,imtemp1);

	// mean filter
	//fastMeanFilter(imtemp1,50,imtemp2);
	blur(imtemp1,imtemp2,Size(49,49));

	// remove background variation
	subtract(imMean3,imtemp2,imtemp1);
	imInf(imROIR,imtemp1,imtemp2);
	//imwrite("ODtemp1.png",imtemp2);
	////--- FFT bug fix
	//for (int j=(imtemp1.rows/10*9); j<imtemp1.rows; j++){
	//	for (int i=0; i<imtemp1.cols; i++){
	//		imtemp2.at<uchar>(j,i)=0;
	//	}
	//}
	////----
	imtemp2.copyTo(imtemp3);
	//imwrite("imOD_removeBG.png",imtemp2);

	// remove bright border
	imCompare(imBorderReflectionR,0,1,0,imtemp2,imtemp1);

	// remove vessel like bright structure
	lengthOpening(imtemp1,imtemp2,10,50*50,0,0);
	//imwrite("imOD_filter.png",imtemp2);

	// mean filter II
	//fastMeanFilter(imtemp2,50,imtemp1);
	blur(imtemp2,imtemp1,Size(49,49));
	//imwrite("imOD_mean.png",imtemp1);
		
	// area selection from the top of the histogram
	int *hist = histogram(imtemp1);
	int s(0);
	for (int h=255; h>0; h--){
		if (s>C_area){
			threshold(imtemp1,imtemp2,h+1,255,0);
			break;
		}
		s+=hist[h];
	}
	//imwrite("imtemp2.png",imtemp2);

	//=======================================================

	// Part 2: vessel information
	int win1[2] = {200,100}; // height , width
	int x,y;
	float nbP,sWindow(win1[0]*win1[1]),maxVD(0);
	int maxVDp1(0),maxVDp2(0);
	int nbPv(0),nbPd1(0),nbPd2(0);//number of pixels of 3 orientation
	//Mat imtempV1 = Mat::zeros(vessels[0].rows,vessels[0].cols,CV_8U);
	//Mat imtempV2 = vessels[0].clone();
	int *vWidthLinePrj = new int[vessels[0].cols];
	memset(vWidthLinePrj,0,sizeof(int)*vessels[0].cols);
	
	for (int j=0; j<vessels[0].rows; j++){
		for (int i=0; i<vessels[0].cols; i++){
			if (vesselProperty[0].at<uchar>(j,i)==0) continue;
			// vessel density within a moving window
			nbP = 0.0f;
			for(int n=-win1[0]/2; n<win1[0]/2; n++){
				for(int m=-win1[1]/2; m<win1[1]/2; m++){
					x = i+m; y = j+n;
					if (x<0 || x>=vessels[0].cols || y<0 || y>=vessels[0].rows){
						x = (i+m+vessels[0].cols)%vessels[0].cols;
						y = (j+n+vessels[0].rows)%vessels[0].rows;
					}
					if(vessels[0].at<uchar>(y,x)>0) nbP++;
				}
			}
			if (nbP/sWindow>maxVD){
				maxVD = nbP/sWindow;
				maxVDp1 = i;
				maxVDp2 = j;
			}
			
			// vessel width
			// if vessel is vertical && width enter 5~15
			if (vesselProperty[3].at<uchar>(j,i)==1 && vesselProperty[2].at<uchar>(j,i)<=15 && vesselProperty[2].at<uchar>(j,i)>5){
				vWidthLinePrj[i] += vesselProperty[2].at<uchar>(j,i);
			}

			// vessel orientation
			switch((int)vesselProperty[3].at<uchar>(j,i)){
			case 1:
				nbPv++;
				break;
			case 2:
				nbPd1++;
				break;
			case 3:
				nbPd2++;
				break;
			}
		}
	}

	// mean and max for line
	int maxVW(0),maxVWp(0),temp;
	for (int i=0; i<vessels[0].cols; i++){
		temp = 0;
		for (int m=-10; m<10; m++){
			temp += vWidthLinePrj[(i+m+vessels[0].cols)%vessels[0].cols];
			if(vWidthLinePrj[(i+m+vessels[0].cols)%vessels[0].cols]>0) nbP++;
		}
		if (temp>maxVW){ maxVW=temp; maxVWp=i;}
	}
	
	//Transfer to 512:
	f = vessels[0].cols/512.0;
	maxVWp/=f; maxVDp1/=f; maxVDp2/=f;
	//cout<<"maxVW p: "<<maxVWp<<endl;
	//cout<<"maxVD p: "<<maxVDp1<<" "<<maxVDp2<<endl;
	//cout<<"number of pixels in 3 orientations: "<< nbPv<<" "<<nbPd1<<" "<<nbPd2<<endl;


	//=======================================================
	// Part 3 Detect the OD position (or NOT presented)
	/************************************************************************/
	/* We have the following types of information:
		1. Intensity ( the brightest part in the image, which also has a reasonable area
		2. Vessel Density (in a rectangular moving window)
		3. Vessel width (mainly the vertical vessels)
		4. Number of pixels in Skeleton image in 3 orientation (if there is little vertical
			pixels and the ratio between vertical and diagonals are too small, there maybe no 
			OD.*/
	/************************************************************************/
	imtemp3.copyTo(imMean3);
	fastDilate(imBorderReflectionR,imtemp1,6,10);
	// remove border reflection
	for(int j=0; j<imMean3.rows; j++){
		for (int i=0; i<imMean3.cols; i++){
			if (imtemp1.at<uchar>(j,i)>0)
				imMean3.at<uchar>(j,i) = 1;
		}
	}
	//imwrite("imMean3b.png",imMean3);
	
	int maxIntp(-1);
	float maxInt(0);
	// First let's consider condition 4
	if (nbPv<300 || ((nbPd1+nbPd2)/nbPv)>5){
		cout<<"No OD presented, too little vertical vessels"<<endl;
	}
	// Vessel Width and Vessel Density are agreed
	else if (abs(maxVWp - maxVDp1)<60){
		int startp = (maxVWp + maxVDp1)/2-50;
		int endp = (maxVWp + maxVDp1)/2+50;
		if (startp<0) startp=0;
		if (endp >= imMean3.cols) endp = imMean3.cols-1;
		int len=endp-startp+1;
		for (int j=0; j<imMean3.rows; j++){
			nbP=0;
			temp = 0;
			for (int i=startp; i<=endp; i++){
				for (int m=-30; m<=30; m++){
					x = i;
					y = (j+m+imMean3.rows) % imMean3.rows;
					temp += imMean3.at<uchar>(y,x);
					if (imMean3.at<uchar>(y,x)>0) nbP++;
				}
			}
			//cout<<j<<" "<<temp/nbP<<"  ";
			if(nbP>500){
				temp/=nbP;
				if (temp>maxInt){maxInt=temp; maxIntp=j;}
			}
		}
		cout<<endl;

		//cout<<"maxInt p: "<<maxIntp<<" value: "<<maxInt<<endl;
		if (imtemp2.at<uchar>(maxIntp,(maxVWp + maxVDp1)/2)>0){
			ODCenter[0] = (maxVWp + maxVDp1)/2;
			ODCenter[1] = maxIntp;
			cout<<"FID: S All agreed! OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
		}
		else{
			bool agreed(false);
			for (int i=startp; i<=endp; i++){
				for (int j=maxIntp-30; j<=maxIntp+30; j++){
					if (j<0 || j>=imMean3.rows) continue;
					if (imtemp2.at<uchar>(j,i)>0){
						agreed = true;
						ODCenter[0]=i;
						ODCenter[1]=j;
						break;
					}
				}
				if(agreed) break;
			}
			if (agreed) cout<<"FID: S All agreed! OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
			else {
				/*ODCenter[0] = (maxVWp + maxVDp1)/2;
				ODCenter[1] = maxIntp;
				cout<<"FID: A Vessel Density and width agreed! OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;*/
				cout<<"oops! No OD detected"<<endl;
			}
		}
	}

	// Vessel Width and Vessel Density are NOT agreed
	else{
		// first do the same thing using only Vessel width
		maxInt=0,maxIntp=0;
		int startp = maxVWp-50;
		int endp = maxVWp+50;
		if (startp<0) startp=0;
		if (endp >= imMean3.cols) endp = imMean3.cols-1;
		for (int j=0; j<imMean3.rows; j++){
			temp = 0;
			nbP = 0;
			for (int i=startp; i<=endp; i++){
				for (int m=-30; m<=30; m++){
					x = i;
					y = (j+m+imMean3.rows) % imMean3.rows;
					temp += imMean3.at<uchar>(y,x);
					if (imMean3.at<uchar>(y,x)>0) nbP++;
				}
			}
			if(nbP!=0){
				temp/=nbP;
				if (temp>maxInt){maxInt=temp; maxIntp=j;}
			}

		}

		if (imtemp2.at<uchar>(maxIntp,maxVWp)>0){
			ODCenter[0] = maxVWp;
			ODCenter[1] = maxIntp;
			cout<<"FID: B Vessel width and Intensity agreed! OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
		}
		else{
			bool agreed(false);
			for (int i=startp; i<=endp; i++){
				for (int j=maxIntp-30; j<=maxIntp+30; j++){
					if (j<0 || j>=imMean3.rows) continue;
					if (imtemp2.at<uchar>(j,i)>0){
						agreed = true;
						ODCenter[0]=i;
						ODCenter[1]=j;
						break;
					}
				}
				if(agreed) break;
			}
			if (agreed) cout<<"FID: B Vessel width and Intensity agreed! OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;

			else{
				// Secondly, try the same thing with Vessel density
				maxInt=0,maxIntp=0;
				int startp = maxVDp1-50;
				int endp = maxVDp1+50;
				if (startp<0) startp=0;
				if (endp >= imMean3.cols) endp = imMean3.cols-1;
				for (int j=0; j<imMean3.rows; j++){
					temp = 0;
					nbP = 0;
					for (int i=startp; i<=endp; i++){
						for (int m=-30; m<=30; m++){
							x = i;
							y = (j+m+imMean3.rows) % imMean3.rows;
							temp += imMean3.at<uchar>(y,x);
							if (imMean3.at<uchar>(y,x)>0) nbP++;
						}
					}
					if(nbP!=0){
						temp/=nbP;
						if (temp>maxInt){maxInt=temp; maxIntp=j;}
					}
				}
				if (imtemp2.at<uchar>(maxIntp,maxVDp1)>0){
					ODCenter[0] = maxVDp1;
					ODCenter[1] = maxIntp;
					cout<<"FID: B Vessel density and Intensity agreed! OD center:  "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
				}
				else{
					bool agreed(false);
					for (int i=startp; i<=endp; i++){
						for (int j=maxIntp-30; j<=maxIntp+30; j++){
							if (j<0 || j>=imMean3.rows) continue;
							if (imtemp2.at<uchar>(j,i)>0){
								agreed = true;
								ODCenter[0]=i;
								ODCenter[1]=j;
								break;
							}
						}
						if(agreed) break;
					}
					if (agreed) cout<<"FID: B Vessel density and Intensity agreed! OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
					else cout<<"oops! Nothing is agreed. NO OD detected! "<<endl;
				}
			}
		}
	}
	
	// Final correction
	if(ODCenter[0]!=-1){
		bool flag(false);
		imtemp1 = Mat::zeros(imtemp1.rows, imtemp1.cols, CV_8U);
		imtemp1.at<uchar>(ODCenter[1],ODCenter[0])=255;
		RecUnderBuild(imtemp2,imtemp1,imtemp3,6);
		Distance(imtemp3,imtemp1,6);
		int *hist = histogram(imtemp1);
		if(hist[257]!=0){
			for(int j=0; j<imMean3.rows; j++){
				for (int i=0; i<imMean3.cols; i++){
					if (imtemp1.at<uchar>(j,i)==hist[257]){
						ODCenter[0]=i;
						ODCenter[1]=j;
						flag = true;
						break;
					}
				}
				if (flag) break;
			}
		}
		if(flag)
			cout<<"Corrected OD center: "<<ODCenter[0]<<" "<<ODCenter[1]<<endl;
	}


	//=============================================================================
	// Macula
	if(ODCenter[0]!=-1){
		// first get candidate by intensity
		getASF(imGreenR,imtemp1,imROIR,1,iminfoR[1]/2,10);
		subtract(imtemp1,imGreenR,imtemp2);
		fastErode(imROIR,imtemp3,6,iminfoR[2]/4);
		imInf(imtemp3,imtemp2,imtemp1);
		lengthOpening(imtemp1,imtemp2,iminfoR[1],iminfoR[1]*iminfoR[1]/2,0,0);
		subtract(imtemp1,imtemp2,imtemp3);

		threshold(imtemp3,imtemp1,7,255,0);
		binAreaSelection(imtemp1,imtemp2,6,70);
		subtract(imtemp1,imtemp2,imtemp2);
		
		lengthOpening(imtemp2,imtemp1,iminfoR[1],iminfoR[1]*iminfoR[1]/2,2,2);
		subtract(imtemp2,imtemp1,imtemp3);

		// second selection
		float stdDist = iminfoR[0]/7*3;
		Dilate(imtemp3,imtemp1,6,2);
		Distance(imtemp1,imtemp2,6);
		Maxima(imtemp2,imtemp3,6);
		queue<int> MCCenterL[2];
		for(int j=0; j<imtemp1.rows; j++){
			for (int i=0; i<imtemp1.cols; i++){
				if (imtemp3.at<uchar>(j,i)>0){
					float dist1 = sqrt(float((ODCenter[0]-i)*(ODCenter[0]-i) + (ODCenter[1]-j)*(ODCenter[1]-j)));
					float dist2 = abs(ODCenter[0]-i);
					if (dist1<(stdDist+50) && dist1>(stdDist-50) && dist2<(stdDist+50) && dist2>(stdDist-50)){
						MCCenterL[0].push(i);
						MCCenterL[1].push(j);
					}
				}
			}
		}
		int MCCenter[2] = {-1,-1};
		if (MCCenterL[0].size()==1){
			MCCenter[0] = MCCenterL[0].front();
			MCCenter[1] = MCCenterL[1].front();
		}
		else if (MCCenterL[0].size()>1){
			int minv(999);
			while(!MCCenterL[0].empty()){
				if (imGreenR.at<uchar>(MCCenterL[1].front(),MCCenterL[0].front()) <= minv){
					minv = imGreenR.at<uchar>(MCCenterL[1].front(),MCCenterL[0].front());
					MCCenter[0] = MCCenterL[0].front();
					MCCenter[1] = MCCenterL[1].front();
				}
				MCCenterL[0].pop();
				MCCenterL[1].pop();
			}
		}
		
		
		// If not detected (normally in the border)
		if (MCCenter[0]==-1){
			fastErode(imROIR,imtemp1,6,iminfoR[1]/4);
			imCompare(imtemp1,255,0,0,imGreenR,imtemp2);
			fastDilate(imtemp2,imtemp1,6,iminfoR[2]/2);
			fastErode(imtemp1,imtemp3,6,iminfoR[2]/2+1);
			imSup(imtemp3,imtemp2,imtemp1);
			//imwrite("MCimtemp1.png",imtemp1); // get a circle
			// test the circle, try to find MC
			int LR,jStart,jEnd,yy;
			queue<int>vMean,posit;
			if (ODCenter[0]<=imtemp1.cols/2) LR=0;
			else LR=1;
			jStart=ODCenter[1]-iminfoR[1];
			jEnd = ODCenter[1]+iminfoR[1];
			if (jStart<0) jStart=0;
			if (jEnd >=imtemp1.rows) jEnd=imtemp1.rows-1;
			for(int j=jStart; j<jEnd; j++){
				meanV=0,nbP=0;
				for (int n=-8; n<=8; n++){
					yy = j+n;
					if (yy<0 || yy>=imtemp1.rows) continue;
					if(LR==0){
						for (int i=0; i<imtemp1.cols/2; i++){
							if (imtemp1.at<uchar>(yy,i)>0){
								meanV += imtemp1.at<uchar>(yy,i);
								nbP++;
								//if (j==179){
								//	cout<<i<<" "<<yy<<" "<<(int)imtemp1.at<uchar>(yy,i)<<"  ";
								//}
							}
						}
					}
					else{
						for (int i=imtemp1.cols/2; i<imtemp1.cols; i++){
							if (imtemp1.at<uchar>(yy,i)>0){
								meanV += imtemp1.at<uchar>(yy,i);
								nbP++;
							}
						}
					}
				}
				vMean.push(meanV/nbP);
				posit.push(j);
			}
			int minV(255),minP(0);
			meanV=0; nbP=posit.size();
			while(!vMean.empty()){
				//cout<<posit.front()<<" "<<vMean.front()<<"  ";
				if (vMean.front()<=minV){
					minV = vMean.front();
					minP = posit.front();
				}
				meanV += vMean.front();
				posit.pop();
				vMean.pop();
			}
			meanV/=nbP;
		/*	cout<<minP<<" "<<minV<<" "<<meanV<<endl;
			cout<<endl;*/
			if ((meanV-minV)>10){// enough low
				if(LR==0){
					for (int i=0; i<imtemp1.cols/2; i++){
						if (imROIR.at<uchar>(minP,i)>0){
							MCCenter[0]=i;
							break;
						}
					}
				}
				else{
					for (int i=imtemp1.cols-1;i>imtemp1.cols/2; i--){
						if (imROIR.at<uchar>(minP,i)>0){
							MCCenter[0]=i;
							break;
						}
					}
				}
				MCCenter[1] = minP;
			}
		}

		ODCenter[2]=MCCenter[0]; ODCenter[3]=MCCenter[1];
		//cout<<"Macular center: "<<MCCenter[0]<<" "<<MCCenter[1]<<endl;
	}


	delete[] vWidthLinePrj;

	////-----------------------------------
	Mat imOD = Mat::zeros(imRedR.rows, imRedR.cols, CV_8U);
	if (ODCenter[0]!=-1){
		Point center(ODCenter[0],ODCenter[1]);
		circle(imOD,center,iminfoR[1]/2,255,-1);
	}
	//imwrite("imOD.png",imOD);
	////-----------------------------------
	return ODCenter;

}

#endif


//Mat imResiduRed = imRedR.clone();
//Mat imout1 = Mat::zeros(imRedR.rows,imRedR.cols,CV_8U);
//Mat imout2 = Mat::zeros(imRedR.rows,imRedR.cols,CV_8U);
//Mat imout3 = Mat::zeros(imRedR.rows,imRedR.cols,CV_8U);
//vector<Mat> im3out;
//im3out.push_back(imout1);
//im3out.push_back(imout2);
//im3out.push_back(imout3);


////int C_area = sizeOD*sizeOD/4/2;
//int C_area = int(3.14*iminfoR[1]*iminfoR[1]/4/5);
//
//for (int i=0; i<3; i++){
//	cout<<i<<endl;
//	int meanV = meanValue(imCR[i],1);
//	imCompare(imROIR,0,0,meanV,imCR[i],imtemp1);

//	// mean filter
//	fastMeanFilter(imtemp1,50,imtemp2);

//	// remove background variation
//	subtract(imCR[i],imtemp2,imtemp1);
//	imInf(imROIR,imtemp1,imtemp2);
//	if(i==0) imtemp2.copyTo(imResiduRed);

//	// remove bright border
//	imCompare(imBorderReflectionR,0,1,0,imtemp2,imtemp1);

//	// remove vessel like bright structure
//	lengthOpening(imtemp1,imtemp2,10,50*50,5,2);
//	imwrite("imtemp1.png",imtemp1);
//	imwrite("imtemp2.png",imtemp2);

//	// mean filter II
//	fastMeanFilter(imtemp2,50,imtemp1);
//	
//	// area selection from the top of the histogram
//	int *hist = histogram(imtemp1);
//	int s(0);
//	for (int h=255; h>0; h--){
//		if (s>C_area){
//			threshold(imtemp1,im3out[i],h+1,255,0);
//			break;
//		}
//		s+=hist[h];
//	}

//}
//imwrite("od_r.png",im3out[0]);
//imwrite("od_g.png",im3out[1]);
//imwrite("od_b.png",im3out[2]);