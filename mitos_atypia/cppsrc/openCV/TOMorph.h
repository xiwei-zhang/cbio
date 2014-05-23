#ifndef __TOMORPH_H
#define __TOMORPH_H

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <queue>
#include <stack>
#include <math.h>
#include <stdint.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "basicOperations.h"
#include "maxTree.h"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;


void Dilate(Mat imin, Mat imout, int se, int size){
	/*	imin: imput image
		imout: output image
		se: 6->hexagonal   8->square
		size: repeat how many times
	*/
	// se == 8 means a square structuring element
	if (se==8){
		Mat imtemp1 = imin.clone();
		// decompose on horizontal and vertical dilation
		for( int n=0; n<size; n++){
			for( int y = 1; y < imtemp1.rows-1; y++ ){
				uchar* line_y = imtemp1.ptr<uchar>(y);
				uchar* line_yu = imtemp1.ptr<uchar>(y-1);
				uchar* line_yd = imtemp1.ptr<uchar>(y+1);
				for( int x = 0; x < imtemp1.cols; x++ ){
					uchar m = max(line_y[x],line_yu[x]);
					imout.at<uchar>(y,x) = max(m,line_yd[x]);
				}
			}
			imout.copyTo(imtemp1);
			for( int y = 0; y < imtemp1.rows; y++ ){
				uchar* line_y = imtemp1.ptr<uchar>(y);
				for( int x = 1; x < imtemp1.cols-1; x++ ){
					uchar m = max(line_y[x],line_y[x-1]);
					imout.at<uchar>(y,x) = max(m,line_y[x+1]);
				}
			}
			imout.copyTo(imtemp1);
		}
	}
	
	// se == 6 means a hexagonal structuring element
	if (se == 6){
		// decompose in three direction
		//   5   0
		// 4   x   1
		//   3   2
		// even:
		//   o  o
		//   o  o  o
		//	 o  o
		// odd :
		//	    x  x
		//	 x  x  x
		//		x  x
		Mat imtemp1 = imin.clone();
		for (int n=0; n<size; n++){
			// in direction 5:
			for( int y = 1; y < imtemp1.rows; y++ ){
				uchar* line_yu = imtemp1.ptr<uchar>(y-1);
				uchar* line_y = imtemp1.ptr<uchar>(y);
				for( int x = 0; x < imtemp1.cols; x++ ){
					if (y%2==0)
						imout.at<uchar>(y,x) = max(line_y[x], line_yu[x-1]);
					else
						imout.at<uchar>(y,x) = max(line_y[x], line_yu[x]);
				}
			}
			// in direction 3:
			for( int y = 0; y < imout.rows-1; y++ ){
				uchar* line_yd = imout.ptr<uchar>(y+1);
				uchar* line_y = imout.ptr<uchar>(y);
				for( int x = 0; x < imout.cols; x++ ){
					if (y%2==0)
						imtemp1.at<uchar>(y,x) = max(line_y[x], line_yd[x-1]);
					else
						imtemp1.at<uchar>(y,x) = max(line_y[x], line_yd[x]);
				}
			}

			// in direction 1:
			for( int y = 0; y < imtemp1.rows; y++ ){
				uchar* line_y = imtemp1.ptr<uchar>(y);
				for( int x = 0; x < imtemp1.cols; x++ ){
					if (y%2==0)
						imout.at<uchar>(y,x) = max(line_y[x], line_y[x+1]);
					else
						imout.at<uchar>(y,x) = max(line_y[x], line_y[x+1]);
				}
			}
			imout.copyTo(imtemp1);
		}
	}
}

void Erode(Mat imin, Mat imout, int se, int size){
	/*	imin: imput image
		imout: output image
		se: 6->hexagonal   8->square
		size: repeat how many times
	*/
	// se == 8 means a square structuring element
	if (se==8){
		Mat imtemp1 = imin.clone();
		// decompose on horizontal and vertical dilation
		for( int n=0; n<size; n++){
			for( int y = 1; y < imtemp1.rows-1; y++ ){
				uchar* line_y = imtemp1.ptr<uchar>(y);
				uchar* line_yu = imtemp1.ptr<uchar>(y-1);
				uchar* line_yd = imtemp1.ptr<uchar>(y+1);
				for( int x = 0; x < imtemp1.cols; x++ ){
					uchar m = min(line_y[x],line_yu[x]);
					imout.at<uchar>(y,x) = min(m,line_yd[x]);
				}
			}
			imout.copyTo(imtemp1);
			for( int y = 0; y < imtemp1.rows; y++ ){
				uchar* line_y = imtemp1.ptr<uchar>(y);
				for( int x = 1; x < imtemp1.cols-1; x++ ){
					uchar m = min(line_y[x],line_y[x-1]);
					imout.at<uchar>(y,x) = min(m,line_y[x+1]);
				}
			}
			imout.copyTo(imtemp1);
		}
	}
	
	// se == 6 means a hexagonal structuring element
	if (se == 6){
		Mat imtemp1 = imin.clone();
		for (int n=0; n<size; n++){
			// in direction 5:
			for( int y = 1; y < imtemp1.rows; y++ ){
				uchar* line_yu = imtemp1.ptr<uchar>(y-1);
				uchar* line_y = imtemp1.ptr<uchar>(y);
				for( int x = 0; x < imtemp1.cols; x++ ){
					if (y%2==0)
						imout.at<uchar>(y,x) = min(line_y[x], line_yu[x-1]);
					else
						imout.at<uchar>(y,x) = min(line_y[x], line_yu[x]);
				}
			}
			// in direction 3:
			for( int y = 0; y < imout.rows-1; y++ ){
				uchar* line_yd = imout.ptr<uchar>(y+1);
				uchar* line_y = imout.ptr<uchar>(y);
				for( int x = 0; x < imout.cols; x++ ){
					if (y%2==0)
						imtemp1.at<uchar>(y,x) = min(line_y[x], line_yd[x-1]);
					else
						imtemp1.at<uchar>(y,x) = min(line_y[x], line_yd[x]);
				}
			}

			// in direction 1:
			for( int y = 0; y < imtemp1.rows; y++ ){
				uchar* line_y = imtemp1.ptr<uchar>(y);
				for( int x = 0; x < imtemp1.cols; x++ ){
					if (y%2==0)
						imout.at<uchar>(y,x) = min(line_y[x], line_y[x+1]);
					else
						imout.at<uchar>(y,x) = min(line_y[x], line_y[x+1]);
				}
			}
			imout.copyTo(imtemp1);
		}
	}
}

void Close(Mat imin, Mat imout, int se, int size){
	Mat imtemp = imout.clone();
	Dilate(imin,imtemp,se,size);
	Erode(imtemp,imout,se,size);
}
void Open(Mat imin, Mat imout, int se, int size){
	Mat imtemp = imout.clone();
	Erode(imin,imtemp,se,size);
	Dilate(imtemp,imout,se,size);
}


Mat Label(Mat imin, Mat imout, int se){
	/* Vincent's queue algo. scan once */
	// initialization
	// threshold(imin,imout,0,0,0);
	imout = Mat::zeros(imin.rows, imin.cols, CV_32S );
	Mat imflag = Mat::zeros(imin.rows, imin.cols, CV_8U );
	Mat imreturn = imin.clone();
	int size[2] = {imin.cols, imin.rows };

	queue<int> Qx,Qy;
	int label(0),x,y,s,t;
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);

	for(int j=0;j<size[1];++j){
		for(int i=0;i<size[0];++i){
			if (imflag.at<uchar>(j,i)==1)
				continue;
			if (imin.at<uchar>(j,i)==0 && imflag.at<uchar>(j,i)==0){
				imflag.at<uchar>(j,i) = 1;
				imout.at<int>(j,i)= 0;
			}
			else {
				imout.at<int>(j,i) = ++label;
				imflag.at<uchar>(j,i)=1;
				Qx.push(i);
				Qy.push(j);
			}

			while (!Qx.empty()){
				s = Qx.front();
				t = Qy.front();
				Qx.pop();
				Qy.pop();

				for (int k=0; k<se; ++k){
					if (t%2==0){
						x = s + se_even[k][1];
						y = t + se_even[k][0];
					}
					else{
						x = s + se_odd[k][1];
						y = t + se_odd[k][0];
					}
					if (x<0 || x>=size[0] || y<0 || y>=size[1]) continue;
					if (imflag.at<uchar>(y,x)==0){
						imflag.at<uchar>(y,x)=1;
						if (imin.at<uchar>(y,x)!=0){
							imout.at<int>(y,x) = label;
							Qx.push(x);
							Qy.push(y);
						}
					}
				}
			}
		}
	}
	
	myCopy(imout,imreturn);

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;

	imwrite("imLabel.png",imreturn);
	return imreturn;
}

int labelCount(Mat imlabel){
	int n(0);
	for(int j=0;j<imlabel.rows;++j){
		for(int i=0;i<imlabel.cols;++i){
			if (imlabel.at<int>(j,i)>n)
				n = imlabel.at<int>(j,i);
		}
	}
	return n;
}


void Distance( Mat imin, Mat imout, int se ){
	/* Vincent's queue algo. 
	First scan, set all region >0: 1, and border of CC: 2 and put into a queue
	search neighbors of each queue, to update distance map
	result - 1 in the end
	*/
	int size[2] = {imin.cols, imin.rows };
	threshold(imin,imout,0,1,0);

	queue<int> Qx,Qy;
	int x,y,m,n;
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);


	for (int j=0;j<size[1];j++){
		for (int i=0;i<size[0];i++){
			if (imout.at<uchar>(j,i)==1){
				if (i==0 || i==size[0]-1 || j==0 || j==size[1]-1){
					imout.at<uchar>(j,i)=2;
					Qx.push(i);
					Qy.push(j);
					continue;
				}
				for (int k=0; k<se; ++k){
					if (j%2==0){
						x = i + se_even[k][0];
						y = j + se_even[k][1];
					}
					else{
						x = i + se_odd[k][0];
						y = j + se_odd[k][1];
					}
					if (x<0 || x>=size[0] || y<0 || y>=size[1]) continue;

					if (imout.at<uchar>(y,x) ==0){
						Qx.push(i);
						Qy.push(j);
						imout.at<uchar>(j,i) = 2;
						break;
					}
				}
			}
		}
	}

	while (!Qx.empty()){
		m = Qx.front();
		n = Qy.front();
		Qx.pop();
		Qy.pop();
		
		for (int k=0; k<se; ++k){
			if ( n%2==0){
				x = m + se_even[k][0];
				y = n + se_even[k][1];
			}
			else{
				x = m + se_odd[k][0];
				y = n + se_odd[k][1];
			}
			if (x<0 || x>=size[0] || y<0 || y>=size[1]) continue;
			if (imout.at<uchar>(y,x) ==1){
				imout.at<uchar>(y,x) = imout.at<uchar>(n,m)+1;
				Qx.push(x);
				Qy.push(y);
			}
		}
	}

	subtract(imout,1,imout);

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}




void RecOverBuild(Mat immask, Mat immark, Mat imout, int se){
	// Initialize HQ and state image
	int size[2] = {immask.cols, immask.rows };
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	uint16_t x,y,xx,yy;
	int prio;
	cv::max(immask,immark,imout);
	Mat imstate = Mat::zeros(immask.rows, immask.cols, CV_8U);
	queue<uint16_t> HQ[256][2];
	for( y = 0; y < imout.rows; y++ ){
		uchar* line_y = imout.ptr<uchar>(y);
		for( x = 0; x < imout.cols; x++ ){
			HQ[(int)line_y[x]][0].push(x);
			HQ[(int)line_y[x]][1].push(y);
		}
	}
	
	// Start treat HQ
	for (int i=0; i<256; i++){
		while (!HQ[i][0].empty()){
			x = HQ[i][0].front();
			y = HQ[i][1].front();
			HQ[i][0].pop();
			HQ[i][1].pop();
			imstate.at<uchar>(y,x)=2;
			// find neighbor
			for (int k=0; k<se; ++k){
				if (y%2==0){
					xx = x + se_even[k][0];
					yy = y + se_even[k][1];
				}
				else{
					xx = x + se_odd[k][0];
					yy = y + se_odd[k][1];
				}
				if (xx<0 || xx>=size[0] || yy<0 || yy>=size[1]) continue;
				if (imstate.at<uchar>(yy,xx)==2 ) continue;
				if (imstate.at<uchar>(yy,xx)==0){
					imstate.at<uchar>(yy,xx) = 1;  // update state image
					prio = max(imout.at<uchar>(y,x),immask.at<uchar>(yy,xx));
					HQ[prio][0].push(xx);  // put neighbors in the HQ
					HQ[prio][1].push(yy);
					imout.at<uchar>(yy,xx) = prio;
				}
			}
		}
	}
	
	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}

void RecUnderBuild(Mat immask, Mat immark, Mat imout, int se){
	// Initialize HQ and state image
	int size[2] = {immask.cols, immask.rows };
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	uint16_t x,y,xx,yy;
	int prio;
	cv::min(immask,immark,imout);
	Mat imstate = Mat::zeros(immask.rows, immask.cols, CV_8U);
	queue<uint16_t> HQ[256][2];
	for( y = 0; y < imout.rows; y++ ){
		uchar* line_y = imout.ptr<uchar>(y);
		for( x = 0; x < imout.cols; x++ ){
			HQ[(int)line_y[x]][0].push(x);
			HQ[(int)line_y[x]][1].push(y);
		}
	}

	// Start treat HQ
	for (int i=255; i>=0; i--){
		while (!HQ[i][0].empty()){
			x = HQ[i][0].front();
			y = HQ[i][1].front();
			HQ[i][0].pop();
			HQ[i][1].pop();
			imstate.at<uchar>(y,x)=2;
			// find neighbor
			for (int k=0; k<se; ++k){
				if (y%2==0){
					xx = x + se_even[k][0];
					yy = y + se_even[k][1];
				}
				else{
					xx = x + se_odd[k][0];
					yy = y + se_odd[k][1];
				}
				if (xx<0 || xx>=size[0] || yy<0 || yy>=size[1]) continue;
				if (imstate.at<uchar>(yy,xx)==2 ) continue;
				if (imstate.at<uchar>(yy,xx)==0){
					imstate.at<uchar>(yy,xx) = 1;  // update state image
					prio = min(imout.at<uchar>(y,x),immask.at<uchar>(yy,xx));
					HQ[prio][0].push(xx);  // put neighbors in the HQ
					HQ[prio][1].push(yy);
					imout.at<uchar>(yy,xx) = prio;
				}
			}
		}
	}
	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}

void Maxima(Mat imin, Mat imout, int se){
	Mat imtemp1 = imin.clone();
	Mat imtemp2 = imin.clone();
	subtract(imin,1,imtemp1);
	RecUnderBuild(imin,imtemp1,imtemp2,6);
	subtract(imin,imtemp2,imout);
}


void Minima(Mat imin, Mat imout, int se){
	Mat imtemp1 = imin.clone();
	Mat imtemp2 = imin.clone();
	add(imin,1,imtemp1);
	RecOverBuild(imin,imtemp1,imtemp2,6);
	subtract(imtemp2,imin, imout);
}




void FillHoles(Mat imin, Mat imout, int se){

	Mat imtemp = Mat::zeros(imin.rows, imin.cols,CV_8U);
	Point p1,p2;
	p1.x=0; p1.y=0;
	p2.x=imin.cols-1; p2.y=0;
	line(imtemp,p1,p2,255);
	p2.x=0; p2.y=imin.rows-1;
	line(imtemp,p1,p2,255);
	p1.x=imin.cols-1;
	p2.x=imin.cols-1;
	line(imtemp,p1,p2,255);
	p1.x=0; p1.y=imin.rows-1;
	line(imtemp,p1,p2,255);
	
	Mat imInvert = imin.clone();
	subtract(255,imin,imInvert);

	RecUnderBuild(imInvert,imtemp,imout,se);
	subtract(255,imout,imout);
}

void SkeletonWithAnchor( Mat imin, Mat immark, Mat imout, int se){
	// now it uses Hexgonal SE
	// mySE is reserved for future
	int imsize[2] = {imin.cols, imin.rows};

	imout = Mat::zeros(imin.rows, imin.cols, CV_8U);
	Mat imtemp = Mat::zeros(imin.rows, imin.cols, CV_8U);

	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);

	/* #######################################
	 * initialize tab_homo
	   #######################################*/
	int tab_homo[64] = {0};
	tab_homo[0] = 1; 	//tab_homo[0] = 111111;
	tab_homo[5] = 1;	//tab_homo[5] = 212111;
	tab_homo[9] = 1; 	//tab_homo[9] = 211211;
	tab_homo[10] = 1;	//tab_homo[10] = 121211;
	tab_homo[11] = 1;	//tab_homo[11] = 221211;
	tab_homo[13] = 1;	//tab_homo[13] = 212211;
	tab_homo[17] = 1;	//tab_homo[17] = 211121;
	tab_homo[18] = 1;	//tab_homo[18] = 121121;
	tab_homo[19] = 1;	//tab_homo[19] = 221121; 
	tab_homo[20] = 1;	//tab_homo[20] = 112121; 
	tab_homo[21] = 1;	//tab_homo[21] = 212121;
	tab_homo[22] = 1;	//tab_homo[22] = 122121;
	tab_homo[23] = 1;	//tab_homo[23] = 222121; 
	tab_homo[25] = 1;	//tab_homo[25] = 211221; 
	tab_homo[26] = 1;	//tab_homo[26] = 121221; 
	tab_homo[27] = 1;	//tab_homo[27] = 221221; 
	tab_homo[29] = 1;	//tab_homo[29] = 212221; 
	tab_homo[34] = 1;	//tab_homo[34] = 121112; 
	tab_homo[36] = 1;	//tab_homo[36] = 112112; 
	tab_homo[37] = 1;	//tab_homo[37] = 212112; 
	tab_homo[38] = 1;	//tab_homo[38] = 122112; 
	tab_homo[40] = 1;	//tab_homo[40] = 111212; 
	tab_homo[41] = 1;	//tab_homo[41] = 211212; 
	tab_homo[42] = 1;	//tab_homo[42] = 121212; 
	tab_homo[43] = 1;	//tab_homo[43] = 221212; 
	tab_homo[44] = 1;	//tab_homo[44] = 112212; 
	tab_homo[45] = 1;	//tab_homo[45] = 212212; 
	tab_homo[46] = 1;	//tab_homo[46] = 122212; 
	tab_homo[50] = 1;	//tab_homo[50] = 121122;
	tab_homo[52] = 1;	//tab_homo[52] = 112122;
	tab_homo[53] = 1;	//tab_homo[53] = 212122;
	tab_homo[54] = 1;	//tab_homo[54] = 122122;
	tab_homo[58] = 1;	//tab_homo[58] = 121222; 
	tab_homo[63] = 1;	//tab_homo[63] = 222222; 

	// table of extremities
	int tab_prune[64] = {0};
	tab_prune[0] = 1;
	tab_prune[1] = 1;
	tab_prune[2] = 1;
	tab_prune[3] = 1;
	tab_prune[4] = 1;
	tab_prune[6] = 1;
	tab_prune[8] = 1;
	tab_prune[12] = 1;
	tab_prune[16] = 1;
	tab_prune[24] = 1;
	tab_prune[32] = 1;
	tab_prune[33] = 1;
	tab_prune[48] = 1;
	

	// initialization of imout
	// max -> 1, others -> 2.
	//   and
	// initialize queue
	queue<int> Qx,Qy,tabX,tabY;
	stack<int> tempX,tempY;
	int mod,x,y,m,n,index;
	for (int j=0;j<imsize[1];j++){
		mod = j%2;
		for (int i=0;i<imsize[0];i++){
			if (imin.at<uchar>(j,i)>0){
			    if (immark.at<uchar>(j,i)>0)
					imout.at<uchar>(j,i) = 1;
				else 
					imout.at<uchar>(j,i) = 2;
				if (i==0 || j==0 || i==imsize[0]-1 || j==imsize[1]-1){
					Qx.push(i);
					Qy.push(j);
					imout.at<uchar>(j,i)=3;
					continue;
				}
				for (int k=0; k<se; ++k){
					if (mod==0){
						x = i + se_even[k][1];
						y = j + se_even[k][0];
					}
					else{
						x = i + se_odd[k][1];
						y = j + se_odd[k][0];
					}
					if (x<0 || x>=imsize[0] || y<0 || y>=imsize[1]) continue;
					if (imin.at<uchar>(y,x)==0 && imout.at<uchar>(j,i)==2){
						Qx.push(i);
						Qy.push(j);
						imout.at<uchar>(j,i)=3;
						break;
					}
				}
			}
		}
	}

	// push fictif
	Qx.push(9999);
	Qy.push(9999);

	// propagation
	while (true){
		m = Qx.front();
		n = Qy.front();
		Qx.pop();
		Qy.pop();
		
		if (m==9999 && n==9999){
			if (Qx.empty()){
				int count(0);
				while (count<tabX.size()){
					m = tabX.front();
					n = tabY.front();
					tabX.pop();
					tabY.pop();
					tabX.push(m);
					tabY.push(n);
					index = 0;

					mod = n%2;
					for (int k=0; k<se; ++k){
						if (mod==0){
							x = m + se_even[k][1];
							y = n + se_even[k][0];
						}
						else{
							x = m + se_odd[k][1];
							y = n + se_odd[k][0];
						}
						if (x<0 || x>=imsize[0] || y<0 || y>=imsize[1]) continue;
						if (imout.at<uchar>(y,x)!=0)
							index = index + (int)pow(2.0f,k);
					}
					if (tab_prune[index]==1){
						imout.at<uchar>(n,m)=2;
						Qx.push(m);
						Qy.push(n);
					}
					count++;
				}

				while (!Qx.empty()){
					m = Qx.front();
					n = Qy.front();
					Qx.pop();
					Qy.pop();
					mod = n%2;
					index = 0;

					for (int k=0; k<se; ++k){
						if (mod==0){
							x = m + se_even[k][1];
							y = n + se_even[k][0];
						}
						else{
							x = m + se_odd[k][1];
							y = n + se_odd[k][0];
						}
						if (x<0 || x>=imsize[0] || y<0 || y>=imsize[1]) continue;
						if (imout.at<uchar>(y,x)!=0)
							index = index + (int)pow(2.0f,k);
					}
					if (tab_prune[index]==1){
						imout.at<uchar>(n,m)=0;
						mod = n%2;

						for (int k=0; k<se; ++k){
							if (mod==0){
								x = m + se_even[k][1];
								y = n + se_even[k][0];
							}
							else{
								x = m + se_odd[k][1];
								y = n + se_odd[k][0];
							}
							if (x<0 || x>=imsize[0] || y<0 || y>=imsize[1]) continue;
							if (imout.at<uchar>(y,x)==3){
								Qx.push(x);
								Qy.push(y);
								imout.at<uchar>(y,x) = 2;
							}
						}
					}
					else imout.at<uchar>(n,m) = 3;
				}
	
				while(!tabX.empty()){
					if (imout.at<uchar>(tabY.front(),tabX.front())==3)
						imout.at<uchar>(tabY.front(),tabX.front())=1;
					tabX.pop();
					tabY.pop();
				}

				break;
			}

			else{
				while(!Qx.empty()){
					tempX.push(Qx.front());
					tempY.push(Qy.front());
					Qx.pop();
					Qy.pop();
				}
				while(!tempX.empty()){
					Qx.push(tempX.top());
					Qy.push(tempY.top());
					tempX.pop();
					tempY.pop();
				}
				Qx.push(9999);
				Qy.push(9999);
			}
		}

		else {
			index = 0;
			mod = n%2;

			for (int k=0; k<se; ++k){
				if (mod==0){
					x = m + se_even[k][1];
					y = n + se_even[k][0];
				}
				else{
					x = m + se_odd[k][1];
					y = n + se_odd[k][0];
				}
				if (x<0 || x>=imsize[0] || y<0 || y>=imsize[1]) continue;
				if (imout.at<uchar>(y,x)!=0)
					index = index + (int)pow(2.0f,k);
				if (imout.at<uchar>(y,x) ==2){
					Qx.push(x);
					Qy.push(y);
					imout.at<uchar>(y,x)=3;
				}
			}

			if (tab_homo[index]==1){
				tabX.push(m);
				tabY.push(n);
			}
			else 
				imout.at<uchar>(n,m) = 0;

		}
	}

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}

void lengthOpening(Mat imin, Mat imout, int C_length, int C_area, int C_circ,int op){
	// op: 1-keep elongated structure ; 2-keep round things
	int *hist = histogram(imin);
	int h=hist[256];
	int lenH = hist[257]+1;
	Mat imstate = Mat::zeros( imin.rows, imin.cols, CV_32S);
	Mat imtemp = Mat::zeros( imin.rows, imin.cols, CV_8U);
	subtract(imstate,2,imstate); // initiate to -2
	mxt maxTree(imin,imstate);
	maxTree.flood_h(h,imin, imstate, 6);

	layer **node = new layer* [lenH];
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [maxTree.Nnodes[i]];
	}
	getRelations(maxTree,node,imin,imstate,lenH,C_area);
	lengthSelection(node,imin,imstate,imout,C_length,C_area,C_circ,op);


	delete[] hist;
	maxTree.DeMT();
	for (int i=0; i<lenH; i++)
		delete[] node[i];
	delete[] node;

}


void binAreaSelection(Mat imin, Mat imout, int se, int C_area){
	/************************************************************************/
	/* Keep binary structures who's area is smaller than C_area                                                                     */
	/************************************************************************/
	imout = Mat::zeros(imin.rows, imin.cols, CV_8U );
	Mat imflag = Mat::zeros(imin.rows, imin.cols, CV_8U );
	int size[2] = {imin.cols, imin.rows };

	queue<int> Qx,Qy;
	queue<int> Qx2,Qy2;
	int label(0),x,y,s,t,area;
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	bool keep(false);

	for(int j=0;j<size[1];++j){
		for(int i=0;i<size[0];++i){
			if (imflag.at<uchar>(j,i)==1)
				continue;
			if (imin.at<uchar>(j,i)==0 && imflag.at<uchar>(j,i)==0){
				imflag.at<uchar>(j,i) = 1;
			}
			else {
				//imout.at<int>(j,i) = ++label;
				area = 1;
				keep = false;
				imflag.at<uchar>(j,i)=1;
				Qx.push(i);
				Qy.push(j);
				Qx2.push(i);
				Qy2.push(j);
			}

			while (!Qx.empty()){
				s = Qx.front();
				t = Qy.front();
				Qx.pop();
				Qy.pop();

				for (int k=0; k<se; ++k){
					if (t%2==0){
						x = s + se_even[k][1];
						y = t + se_even[k][0];
					}
					else{
						x = s + se_odd[k][1];
						y = t + se_odd[k][0];
					}
					if (x<0 || x>=size[0] || y<0 || y>=size[1]) continue;
					if (imflag.at<uchar>(y,x)==0){
						imflag.at<uchar>(y,x)=1;
						if (imin.at<uchar>(y,x)!=0){
							//imout.at<int>(y,x) = label;
							area++;
							Qx.push(x);
							Qy.push(y);
							Qx2.push(x);
							Qy2.push(y);
						}
					}
				}
			}
			if(area<C_area){
				while(!Qx2.empty()){
					s = Qx2.front();
					t = Qy2.front();
					Qx2.pop();
					Qy2.pop();
					imout.at<uchar>(t,s)=255;
				}
			}
			else{
				while(!Qx2.empty()){
					Qx2.pop();
					Qy2.pop();
				}
			}
		}
	}

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}

void UltimateOpening(Mat imin, Mat imout, int se, int C_length, int delta=0){

	int *hist = histogram(imin);
	int h=hist[256];
	int lenH = hist[257]+1;
	Mat imstate = Mat::zeros( imin.rows, imin.cols, CV_32S);
	Mat imtemp = Mat::zeros( imin.rows, imin.cols, CV_8U);
	imout = Mat::zeros( imin.rows, imin.cols, CV_8U);
	subtract(imstate,2,imstate); // initiate to -2
	mxt maxTree(imin,imstate);
	maxTree.flood_h(h,imin, imstate, 6);

	layer **node = new layer* [lenH];
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [maxTree.Nnodes[i]];
	}

	getRelations(maxTree,node,imin,imstate,lenH,0);

	//cout<<"GD "<<imstate.at<int>(269,275)<<" "<<(int)imin.at<uchar>(269,275)<<endl;
	int hh,ii,fh,fi,diff,xmin,xmax,ymin,ymax,w,l,lf,h1;
	list<int>::iterator it1;
	list<int>::iterator it2;
	for (int i=hist[257]; i>0; --i){
		for (int j=0; j<maxTree.Nnodes[i]; ++j){
			if (!node[i][j].children[0].empty()) continue;
			hh = i; ii = j;
			diff=0;
			xmax = node[i][j].xmax;
			xmin = node[i][j].xmin;
			ymax = node[i][j].ymax;
			ymin = node[i][j].ymin;
			w = xmax-xmin+1;
			h = ymax-ymin+1;
			l = max(h,w);
			h1 = hh;

			while(l<=C_length){
				lf = l;
				if (node[hh][ii].mark != 0) break;
				fh = node[hh][ii].parent[0];
				fi = node[hh][ii].parent[1];
				
				if(fh==-1){
					diff = h1;
				}
				else{
					xmax = max(xmax,node[fh][fi].xmax);
					xmin = min(xmin,node[fh][fi].xmin);
					ymax = max(ymax,node[fh][fi].ymax);
					ymin = min(ymin,node[fh][fi].ymin);
					w = xmax-xmin+1;
					h = ymax-ymin+1;
					l = max(h,w);

					if(l>(lf+delta)){
						diff = h1 - fh;
						h1 = fh;
					}
				}
				if (diff==255) cout<<"FFF "<<h1<<" "<<fh<<endl;

				//if(i==204 && j==60) cout<<hh<<" "<<diff<<endl;
				// update output image
				list<int> p[2];
				node[hh][ii].getPixels(node,p);
				it1 = p[0].begin();
				it2 = p[1].begin();
				while(it1!=p[0].end()){
					if(imout.at<uchar>(*it2,*it1) < diff){
						imout.at<uchar>(*it2,*it1) = diff;
					}
					it1++;
					it2++;
				}

				node[hh][ii].mark=1;
				if (fh ==-1) break;
				hh = fh; ii = fi;
			}
		}
	}


	imwrite("UOimin.png",imin);
	imwrite("UOimout.png",imout);


	delete[] hist;
	maxTree.DeMT();
	for (int i=0; i<lenH; i++)
		delete[] node[i];
	delete[] node;
}


void grayAreaSelection(Mat imin, Mat imout, int se, int C_area){
	int *hist = histogram(imin);
	int h=hist[256];
	int lenH = hist[257]+1;
	Mat imstate = Mat::zeros( imin.rows, imin.cols, CV_32S);
	subtract(imstate,2,imstate); // initiate to -2
	mxt maxTree(imin,imstate);
	maxTree.flood_h(h,imin, imstate, 6);

	layer **node = new layer* [lenH];
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [maxTree.Nnodes[i]];
	}

	getRelations(maxTree, node, imin, imstate, lenH, C_area);

	areaSelection(node, imin, imstate, imout, C_area);

	delete[] hist;
	maxTree.DeMT();
	for (int i=0; i<lenH; i++)
		delete[] node[i];
	delete[] node;
}


#endif
