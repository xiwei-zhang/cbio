#ifndef __filter_h
#define __filter_h

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



void AmeobaFilterSimple(Mat imin, Mat imout, int se, int size, float lambda, float th){
	lambda = 0.5;
	th = 10;
	imout = Mat::zeros(imin.rows, imin.cols, CV_8U);
	Mat kernal = Mat::zeros((size*2+1),(size*2+1),CV_8U);
	Mat kernal2 = Mat::zeros((size*2+1),(size*2+1),CV_8U);
	Mat kernal3 = kernal.clone();
	Mat kernal4 = Mat::zeros((size*2+1),(size*2+1),CV_8U);
	Mat kernal5 = Mat::zeros((size*2+1),(size*2+1),CV_8U);

	int x0(12),y0(19);
	x0 = 11;
	y0 = 19;

	for (int i=-size; i<=size; i++){
		for (int j=-size; j<=size; j++){
			if (y0+j < 0 || y0+j>=imin.rows || x0+i<0 || x0+i>=imin.cols) continue;
			kernal2.at<uchar>(size+j,size+i) = imin.at<uchar>(y0+j,x0+i);
		}
	}
	//kernal3.copyTo(kernal2,kernal3);


	//// max tree///////////////////////
	int *hist = histogram(kernal2);
	int h=hist[256];
	int lenH = hist[257]+1;
	Mat imstate = Mat::zeros( size*2+1, size*2+1, CV_32S);
	subtract(imstate,2,imstate); // initiate to -2
	mxt maxTree(kernal2,imstate);
	maxTree.flood_h(h,kernal2, imstate, 6);

	layer **node = new layer* [lenH];  // node is a Class layer object
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [maxTree.Nnodes[i]];
	}
	getRelations(maxTree,node,kernal2,imstate,lenH,(size*2+1)*(size*2+1));

	int hh,ii,fh,fi,h0;
	hh = kernal2.at<uchar>(size,size);
	ii = imstate.at<int>(size,size);
	fh = hh; fi=ii; h0=hh;
	while(fh != -1){
		if ((h0-fh)>2) break;
		hh = fh; ii = fi;
		fh = node[hh][ii].parent[0];
		fi = node[hh][ii].parent[1];
	}
	cout<<hh<<" "<<ii<<endl;
	while (!node[hh][ii].p[0].empty()){
		kernal4.at<uchar>(node[hh][ii].p[1].front(),node[hh][ii].p[0].front()) = 255;
		node[hh][ii].p[0].pop_front();
		node[hh][ii].p[1].pop_front();
	}
	
	for (int i=0; i<lenH; ++i){
		delete[] node[i];
	}
	delete[] node;
	delete[] hist;
	maxTree.DeMT();
	///////////////////////////////


	//// min tree///////////////////////
	subtract(255,kernal2,kernal3);
	hist = histogram(kernal3);
	h=hist[256];
	lenH = hist[257]+1;
	imstate = Mat::zeros( size*2+1, size*2+1, CV_32S);
	subtract(imstate,2,imstate); // initiate to -2
	mxt minTree(kernal3,imstate);
	minTree.flood_h(h,kernal3, imstate, 6);
	node = new layer* [lenH];  // node is a Class layer object
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [minTree.Nnodes[i]];
	}
	getRelations(minTree,node,kernal3,imstate,lenH,(size*2+1)*(size*2+1));

	hh = kernal3.at<uchar>(size,size);
	ii = imstate.at<int>(size,size);
	fh = hh; fi=ii; h0=hh;
	while(fh != -1){
		if ((h0-fh)>2) break;
		hh = fh; ii = fi;
		fh = node[hh][ii].parent[0];
		fi = node[hh][ii].parent[1];
	}
	cout<<hh<<" "<<ii<<endl;
	while (!node[hh][ii].p[0].empty()){
		kernal5.at<uchar>(node[hh][ii].p[1].front(),node[hh][ii].p[0].front()) = 255;
		node[hh][ii].p[0].pop_front();
		node[hh][ii].p[1].pop_front();
	}
	for (int i=0; i<lenH; ++i){
		delete[] node[i];
	}
	delete[] node;
	delete[] hist;
	minTree.DeMT();
	///////////////////////////////


	imInf(kernal5,kernal4,kernal3);


	//kernal.at<uchar>(size,size) = 1;
	//kernal.copyTo(kernal2,kernal);
	//for (int i=0; i<size; i++){
	//	fastDilate(kernal2,kernal3,6,i+1);
	//	add(kernal3,kernal,kernal);
	//}
	//subtract(size+1,kernal,kernal);

	//kernal2 = Mat::zeros((size*2+1),(size*2+1),CV_8U);

	//unsigned __int8 curV = imin.at<uchar>(y0,x0);
	//for (int i=-size; i<=size; i++){
	//	for (int j=-size; j<=size; j++){
	//		kernal2.at<uchar>(size+j,size+i) = abs(imin.at<uchar>(y0+j,x0+i) - curV);
	//		kernal3.at<uchar>(size+j,size+i) = round(kernal2.at<uchar>(size+j,size+i)*lambda + kernal.at<uchar>(size+j,size+i));
	//		kernal4.at<uchar>(size+j,size+i) = kernal3.at<uchar>(size+j,size+i)>th?0:255;
	//	}
	//}
	imwrite("temp5.png",kernal5);
	imwrite("temp4.png",kernal4);
	imwrite("temp3.png",kernal3);
	imwrite("temp2.png",kernal2);
	imwrite("temp.png",kernal);
}

#endif