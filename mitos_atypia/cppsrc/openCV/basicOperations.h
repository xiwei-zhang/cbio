#ifndef __basicOperations_h
#define __basicOperations_h

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <stdint.h>

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

int **nl(int se, bool even=true){
	if (se==4){
		int** nbList = new int*[4];
		for (int i=0; i<4; i++) nbList[i] = new int[2];
		nbList[0][0] = 0; nbList[0][1] = 1;
		nbList[1][0] = -1; nbList[1][1] = 0;
		nbList[2][0] = 0; nbList[2][1] = -1;
		nbList[3][0] = 1; nbList[3][1] = 0;
		return nbList;
	}
	if (se==8){
		int** nbList = new int*[8];
		for (int i=0; i<8; i++) nbList[i] = new int[2];
		nbList[0][0] = -1; nbList[0][1] = -1;
		nbList[1][0] = -1; nbList[1][1] = 0;
		nbList[2][0] = -1; nbList[2][1] = 1;
		nbList[3][0] = 0; nbList[3][1] = -1;
		nbList[4][0] = 0; nbList[4][1] = 1;
		nbList[5][0] = 1; nbList[5][1] = -1;
		nbList[6][0] = 1; nbList[6][1] = 0;
		nbList[7][0] = 1; nbList[7][1] = 1;
		return nbList;
	}
	if (se==6){
		if (even){
			int **nbList = new int*[6];
			for (int i=0; i<6; i++) nbList[i] = new int[2];
			nbList[0][0] = 0; nbList[0][1] = 1;
			nbList[1][0] = -1; nbList[1][1] = 0;
			nbList[2][0] = -1; nbList[2][1] = -1;
			nbList[3][0] = 0; nbList[3][1] = -1;
			nbList[4][0] = 1; nbList[4][1] = -1;
			nbList[5][0] = 1; nbList[5][1] = 0;
			//nbList[0][0] = -1; nbList[0][1] = -1;
			//nbList[1][0] = -1; nbList[1][1] = 0;
			//nbList[2][0] = 0; nbList[2][1] = -1;
			//nbList[3][0] = 0; nbList[3][1] = 1;
			//nbList[4][0] = 1; nbList[4][1] = -1;
			//nbList[5][0] = 1; nbList[5][1] = 0;
			return nbList;
		}
		if (!even){
			int **nbList = new int*[6];
			for (int i=0; i<6; i++) nbList[i] = new int[2];
			nbList[0][0] = 0; nbList[0][1] = 1;
			nbList[1][0] = -1; nbList[1][1] = 1;
			nbList[2][0] = -1; nbList[2][1] = 0;
			nbList[3][0] = 0; nbList[3][1] = -1;
			nbList[4][0] = 1; nbList[4][1] = 0;
			nbList[5][0] = 1; nbList[5][1] = 1;
			return nbList;
		}
	}
}

void myCopy(Mat imin, Mat imout){
	int tp = imin.depth();
	if (tp == 2){
		for( int y = 0; y < imin.rows; y++ ){
			for( int x = 0; x < imin.cols; x++ ){
				imout.at<uchar>(y,x) = imin.at<int16_t>(y,x)%256;
			}
		}
	}
	if (tp == 4){
		for( int y = 0; y < imin.rows; y++ ){
			for( int x = 0; x < imin.cols; x++ ){
				imout.at<uchar>(y,x) = imin.at<int>(y,x)%256;
			}
		}
	}
}

int *histogram( Mat imin){
	int size[2] = {imin.cols, imin.rows };
	int *hist = new int[258];
	for (int j = 0; j<258; ++j){
		hist[j]=0;
	}
	for (int j = 0; j<size[1]; ++j){
		for (int i =0; i<size[0]; ++i){
			++hist[int(imin.at<uchar>(j,i))];
		}
	}
	int min(0),max(255),fmin(0),fmax(0);
	for (int j = 0; j<255; ++j){
		if (hist[j]==0 && fmin==0) ++min;
		else fmin=1;

		if (hist[255-j]==0 && fmax==0) --max;
		else fmax=1;
	}
	hist[256]=min;
	hist[257]=max;
	return hist;
}

double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
	return diffms;
} 

template <class T1> int round(T1 number){
	double a;
	if (number>=0)
		return (float)modf(number,&a)>0.5?ceil(number):floor(number);
	else
		return (float)abs(modf(number,&a))>0.5?floor(number):ceil(number);
}

void imCompare(Mat imin, int v, int op, Mat imYes, Mat imNo, Mat imout){
	// op: 0->equal 1->greater 2->smaller 3-> greater equal 4-> smaller equal
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_in = imin.ptr<uchar>(y);
		uchar* line_yes = imYes.ptr<uchar>(y);
		uchar* line_no = imNo.ptr<uchar>(y);
		uchar* line_out = imout.ptr<uchar>(y);
		for( int x = 0; x < imin.cols; x++ ){
			switch(op){
			case 0:
				if (line_in[x]==v)
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 1:
				if (line_in[x]>v)
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 2:
				if (line_in[x]<v)
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 3:
				if (line_in[x]>=v)
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 4:
				if (line_in[x]<=v)
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			}
		}
	}
}

void imCompare(Mat imin, int v, int op, int vYes, Mat imNo, Mat imout){
	// op: 0->equal 1->greater 2->smaller 3-> greater equal 4-> smaller equal
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_in = imin.ptr<uchar>(y);
		uchar* line_no = imNo.ptr<uchar>(y);
		uchar* line_out = imout.ptr<uchar>(y);
		for( int x = 0; x < imin.cols; x++ ){
			switch(op){
			case 0:
				if (line_in[x]==v)
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 1:
				if (line_in[x]>v)
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 2:
				if (line_in[x]<v)
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 3:
				if (line_in[x]>=v)
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 4:
				if (line_in[x]<=v)
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			}
		}
	}
}

void imCompare(Mat imin, int v, int op, int vYes, int vNo, Mat imout){
	// op: 0->equal 1->greater 2->smaller 3-> greater equal 4-> smaller equal
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_in = imin.ptr<uchar>(y);
		uchar* line_out = imout.ptr<uchar>(y);
		for( int x = 0; x < imin.cols; x++ ){
			switch(op){
			case 0:
				if (line_in[x]==v)
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 1:
				if (line_in[x]>v)
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 2:
				if (line_in[x]<v)
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 3:
				if (line_in[x]>=v)
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 4:
				if (line_in[x]<=v)
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			}
		}
	}
}

void imCompare(Mat imin, Mat imComp, int op, Mat imYes, Mat imNo, Mat imout){
	// op: 0->equal 1->greater 2->smaller 3-> greater equal 4-> smaller equal
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_in = imin.ptr<uchar>(y);
		uchar* line_yes = imYes.ptr<uchar>(y);
		uchar* line_no = imNo.ptr<uchar>(y);
		uchar* line_out = imout.ptr<uchar>(y);
		uchar* line_comp = imComp.ptr<uchar>(y);
		for( int x = 0; x < imin.cols; x++ ){
			switch(op){
			case 0:
				if (line_in[x]==line_comp[x])
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 1:
				if (line_in[x]>line_comp[x])
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 2:
				if (line_in[x]<line_comp[x])
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 3:
				if (line_in[x]>=line_comp[x])
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			case 4:
				if (line_in[x]<=line_comp[x])
					line_out[x] = line_yes[x];
				else
					line_out[x] = line_no[x];
				break;
			}
		}
	}
}

void imCompare(Mat imin, Mat imComp, int op, int vYes, Mat imNo, Mat imout){
	// op: 0->equal 1->greater 2->smaller 3-> greater equal 4-> smaller equal
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_in = imin.ptr<uchar>(y);
		uchar* line_no = imNo.ptr<uchar>(y);
		uchar* line_out = imout.ptr<uchar>(y);
		uchar* line_comp = imComp.ptr<uchar>(y);
		for( int x = 0; x < imin.cols; x++ ){
			switch(op){
			case 0:
				if (line_in[x]==line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 1:
				if (line_in[x]>line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 2:
				if (line_in[x]<line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 3:
				if (line_in[x]>=line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			case 4:
				if (line_in[x]<=line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = line_no[x];
				break;
			}
		}
	}
}

void imCompare(Mat imin, Mat imComp, int op, int vYes, int vNo, Mat imout){
	// op: 0->equal 1->greater 2->smaller 3-> greater equal 4-> smaller equal
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_in = imin.ptr<uchar>(y);
		uchar* line_out = imout.ptr<uchar>(y);
		uchar* line_comp = imComp.ptr<uchar>(y);
		for( int x = 0; x < imin.cols; x++ ){
			switch(op){
			case 0:
				if (line_in[x]==line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 1:
				if (line_in[x]>line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 2:
				if (line_in[x]<line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 3:
				if (line_in[x]>=line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			case 4:
				if (line_in[x]<=line_comp[x])
					line_out[x] = vYes;
				else
					line_out[x] = vNo;
				break;
			}
		}
	}
}

void imSup(Mat imin1, Mat imin2, Mat imout){
	for (int j=0; j<imin1.rows; j++){
		for (int i=0; i<imin1.cols; i++){
			imout.at<uchar>(j,i) = imin1.at<uchar>(j,i)>imin2.at<uchar>(j,i)?imin1.at<uchar>(j,i):imin2.at<uchar>(j,i);
		}
	}
}

void imInf(Mat imin1, Mat imin2, Mat imout){
	for (int j=0; j<imin1.rows; j++){
		for (int i=0; i<imin1.cols; i++){
			imout.at<uchar>(j,i) = imin1.at<uchar>(j,i)>imin2.at<uchar>(j,i)?imin2.at<uchar>(j,i):imin1.at<uchar>(j,i);
		}
	}
}

int autoThreshold(Mat imin, int onset){

	int th,th_;
	int *hist = histogram(imin);
	th = (hist[256]+hist[257])/2;
	if (onset){
		if (hist[256]==0)  // calculate on set
			hist[256]=1;
	}
	for (int k=0; k<100; k++){
		int sumV(0), sumP(0);
		double th1, th2;
		for (int i=hist[256]; i<th; i++){
			sumV += hist[i]*i;
			sumP += hist[i];
		}
		th1 = double(sumV)/sumP;
		sumV=0; sumP=0;
		for (int i=th; i<=hist[257]; i++){
			sumV += hist[i]*i;
			sumP += hist[i];
		}
		th2 = double(sumV)/sumP;
		th_ = th;
		th = round((th1+th2)/2);
		if (th_==th) break;
		//cout<<th<<" "<<th1<<" "<<th2<<endl;
	}
	return th;
}

int autoThresholdMaxVar(Mat imin, int onset){

	int th;
	// initialization
	int *hist = histogram(imin);
	if (hist[257]==0) return 0; // if a black image
	if (onset){
		if (hist[256]==0)  // calculate on set
			hist[256]=1;
	}
	double sumP(0);
	for (int i=hist[256]; i<=hist[257]; i++){
		sumP+=hist[i];
	}
	double *p = new double[hist[257]+1];
	if (onset){
		p[0]=0;
		for (int i=1; i<=hist[257]; i++){
			p[i]=hist[i]/sumP;
		}
	}
	else {
		for (int i=0; i<=hist[257]; i++){
			p[i]=hist[i]/sumP;
		}
	}

	double maxVar(0.0);
	int maxT;
	for (th=hist[256];th<=hist[257]; th++){
		double omiga0(0),omiga1(0),miu0(0),miu1(0),miuT(0),sigmaP2(0);
		for (int i=0; i<th; i++){
			omiga0+=p[i];
			miu0+=i*p[i];
			miuT+=i*p[i];
		}
		if (omiga0==0) miu0=0;
		else miu0/=omiga0;

		for (int i=th; i<hist[257]; i++){
			omiga1+=p[i];
			miu1+=i*p[i];
			miuT+=i*p[i];
		}
		if (omiga1==0) miu1=0;
		else miu1/=omiga1;

		sigmaP2 = omiga0*(miu0-miuT)*(miu0-miuT) + omiga1*(miu1-miuT)*(miu1-miuT);
		if (sigmaP2>maxVar){ maxVar = sigmaP2; maxT = th;}
	}
	return maxT;
}

int meanValue(Mat imin, int onset=0){
	int v(0),np(0);
	if (onset==0){
		for (int j=0; j<imin.rows; j++){
			for (int i=0; i<imin.cols; i++){
				v+=imin.at<uchar>(j,i);
			}
		}
		np = imin.cols * imin.rows;
	}
	else{
		for (int j=0; j<imin.rows; j++){
			for (int i=0; i<imin.cols; i++){
				if (imin.at<uchar>(j,i)>0){
					v+=imin.at<uchar>(j,i);
					np++;
				}
			}
		}
	}
	if (np==0) return 0;
	return v/np;
}

inline double getHV(int r, int g, int b){
	return atan2((sqrt(3.0f)*(g-b)),(2*r-g-b));
}

void getHue(vector <Mat> planes, Mat imout){
	Mat imgreen = planes[1];
	Mat imred = planes[2];
	Mat imblue = planes[0];
	int r,g,b;
	imout = Mat::zeros(imout.rows,imout.cols,CV_8U);
	for (int j=0; j<imout.rows; j++){
		for (int i=0; i<imout.cols; i++){
			r = imred.at<uchar>(j,i);
			g = imgreen.at<uchar>(j,i);
			b = imblue.at<uchar>(j,i);
			if (r==0) continue;
			imout.at<uchar>(j,i) = round<double>((getHV(r,g,b)+3.14)/2/3.14*255);
		}
	}
	imwrite("H1.png",imout);
}

#endif
