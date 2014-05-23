#ifndef __fastMorph_h
#define __fastMorph_h

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdint.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;


inline void lineSup(const uint8_t *linein1, const uint8_t *linein2, uint8_t *lineout, const int size ){
	const __m128i* lin1 = (__m128i*) linein1;
	const __m128i* lin2 = (__m128i*) linein2;
	__m128i* lout = (__m128i*) lineout;
	__m128i r0,r1,r2;
	for(int i=0 ; i<size ; i+=16){
		r0 = _mm_load_si128(lin1);
		r1 = _mm_load_si128(lin2);
		r2 = _mm_max_epu8(r0,r1);
		_mm_store_si128( lout,r2);
		lin1++;
		lin2++;
		lout++;
	}
}

inline void lineInf(const uint8_t *linein1, const uint8_t *linein2, uint8_t *lineout, const int size ){
	const __m128i* lin1 = (__m128i*) linein1;
	const __m128i* lin2 = (__m128i*) linein2;
	__m128i* lout = (__m128i*) lineout;
	__m128i r0,r1,r2;
	for(int i=0 ; i<size ; i+=16){
		r0 = _mm_load_si128(lin1);
		r1 = _mm_load_si128(lin2);
		r2 = _mm_min_epu8(r0,r1);
		_mm_store_si128( lout,r2);
		lin1++;
		lin2++;
		lout++;
	}
}


void fastDilate(Mat imin, Mat imout, int se, int size){
	/*	imin: imput image
		imout: output image
		se: 6->hexagonal   8->square
		size: repeat how many times
	*/
	// se == 8 means a square structuring element

	int imsize[2] = {imin.rows, (int)ceil(imin.cols/16.0)*16};

	uint8_t** imin8 = new uint8_t*[imsize[0]];
	uint8_t** imtemp8 = new uint8_t*[imsize[0]];
	for (int i=0;i<imsize[0];i++){
		imin8[i] = new uint8_t[imsize[1]];
		imtemp8[i] = new uint8_t[imsize[1]];
	}
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_y = imin.ptr<uchar>(y);
		for( int x = 0; x <imsize[1]; x++ ){
			if (x>=imin.cols){
				imin8[y][x] = (uint8_t)line_y[imin.cols-1];
				imtemp8[y][x] = (uint8_t)line_y[imin.cols-1];
			}
			else{
				imin8[y][x] = (uint8_t)line_y[x];
				imtemp8[y][x] = (uint8_t)line_y[x];
			}
		}
	}

	if (se==8){
		uint8_t* line_src;
		uint8_t* line_temp1 = new uint8_t[imsize[1]];
		uint8_t* line_temp2 = new uint8_t[imsize[1]];
		for( int n=0; n<size; n++){
			// up and down
			for (int y=1; y<imin.rows-1; y++){
				line_src = imin8[y];
				memcpy(line_temp1, imin8[y-1], imsize[1]);
				lineSup(line_src,line_temp1,line_temp2,imsize[1]);
				line_src = imtemp8[y];
				memcpy(line_temp1, imin8[y+1], imsize[1]);
				lineSup(line_temp1,line_temp2,line_src,imsize[1]);
			}

			// left and right
			for (int y=0; y<imin.rows; y++){
				line_src = imtemp8[y];
				memcpy(line_temp1, line_src+1, imsize[1]-1);
				line_temp1[imsize[1]-1] = line_src[imsize[1]-1];
				lineSup(line_src,line_temp1,line_temp2,imsize[1]);

				line_src = imin8[y];
				line_temp1[0] = imtemp8[y][0];
				memcpy(line_temp1+1, imtemp8[y], imsize[1]-1);
				lineSup(line_temp2,line_temp1,line_src,imsize[1]);
			}
		}
		delete[] line_temp1;
		delete[] line_temp2; 
	}

	if (se==6){
		uint8_t* line_src;
		uint8_t* line_temp1 = new uint8_t[imsize[1]];
		uint8_t* line_temp2 = new uint8_t[imsize[1]];
		for( int n=0; n<size; n++){
			// 5
			for (int y=1; y<imin.rows; y++){
				line_src = imin8[y];
				if (y%2==0){
					memcpy(line_temp1+1, imin8[y-1], imsize[1]-1);
					line_temp1[0] = imin8[y-1][0];
				}
				else{
					memcpy(line_temp1,imin8[y-1], imsize[1] );
				}
				
				lineSup(line_src,line_temp1,line_temp2,imsize[1]);
				memcpy(imtemp8[y], line_temp2, imsize[1]);
			}


			// 1
			for (int y=0; y<imin.rows; y++){
				line_src = imtemp8[y];
				memcpy(line_temp1, line_src+1, imsize[1]-1);
				line_temp1[imsize[1]-1] =line_src[imsize[1]-1];
				lineSup(line_src,line_temp1,line_temp2,imsize[1]);
				memcpy(imin8[y], line_temp2, imsize[1]);
			}

			// 3
			for (int y=0; y<imin.rows-1; y++){
				line_src = imin8[y];
				if (y%2==0){
					memcpy(line_temp1+1, imin8[y+1], imsize[1]-1);
					line_temp1[0] = imin8[y+1][0];
				}
				else{
					memcpy(line_temp1, imin8[y+1], imsize[1]);
				}
				
				lineSup(line_src,line_temp1,line_temp2,imsize[1]);
				memcpy(imtemp8[y], line_temp2, imsize[1]);
			}

			for (int y=0; y<imin.rows; y++)
				memcpy(imin8[y], imtemp8[y], imsize[1]);
		}
		delete[] line_temp1;
		delete[] line_temp2;
	}

	

	for( int y = 0; y < imin.rows; y++ ){
		for( int x = 0; x < imin.cols; x++ ){
			imout.at<uchar>(y,x) = (uchar)imin8[y][x];
		}
	}
	for (int y=0; y<imin.rows; y++){
		delete[] imin8[y];
		delete[] imtemp8[y];
	}
	delete[] imin8;
	delete[] imtemp8;

}



void fastErode(Mat imin, Mat imout, int se, int size){
	int imsize[2] = {imin.rows, (int)ceil(imin.cols/16.0)*16};

	uint8_t** imin8 = new uint8_t*[imsize[0]];
	uint8_t** imtemp8 = new uint8_t*[imsize[0]];
	for (int i=0;i<imsize[0];i++){
		imin8[i] = new uint8_t[imsize[1]];
		imtemp8[i] = new uint8_t[imsize[1]];
	}
	for( int y = 0; y < imin.rows; y++ ){
		uchar* line_y = imin.ptr<uchar>(y);
		for( int x = 0; x <imsize[1]; x++ ){
			if (x>=imin.cols){
				imin8[y][x] = (uint8_t)line_y[imin.cols-1];
				imtemp8[y][x] = (uint8_t)line_y[imin.cols-1];
			}
			else{
				imin8[y][x] = (uint8_t)line_y[x];
				imtemp8[y][x] = (uint8_t)line_y[x];
			}
		}
	}

	if (se==8){
		uint8_t* line_src;
		uint8_t* line_temp1 = new uint8_t[imsize[1]];
		uint8_t* line_temp2 = new uint8_t[imsize[1]];
		for( int n=0; n<size; n++){
			// up and down
			for (int y=1; y<imin.rows-1; y++){
				line_src = imin8[y];
				memcpy(line_temp1, imin8[y-1], imsize[1]);
				lineInf(line_src,line_temp1,line_temp2,imsize[1]);
				line_src = imtemp8[y];
				memcpy(line_temp1, imin8[y+1], imsize[1]);
				lineInf(line_temp1,line_temp2,line_src,imsize[1]);
			}

			// left and right
			for (int y=0; y<imin.rows; y++){
				line_src = imtemp8[y];
				memcpy(line_temp1, line_src+1, imsize[1]-1);
				line_temp1[imsize[1]-1] = line_src[imsize[1]-1];
				lineInf(line_src,line_temp1,line_temp2,imsize[1]);

				line_src = imin8[y];
				line_temp1[0] = imtemp8[y][0];
				memcpy(line_temp1+1, imtemp8[y], imsize[1]-1);
				lineInf(line_temp2,line_temp1,line_src,imsize[1]);
			}
		}
		delete[] line_temp1;
		delete[] line_temp2; 
	}

	if (se==6){
		uint8_t* line_src;
		uint8_t* line_temp1 = new uint8_t[imsize[1]];
		uint8_t* line_temp2 = new uint8_t[imsize[1]];
		for( int n=0; n<size; n++){
			// 5
			for (int y=1; y<imin.rows; y++){
				line_src = imin8[y];
				if (y%2==0){
					memcpy(line_temp1+1, imin8[y-1], imsize[1]-1);
					line_temp1[0] = imin8[y-1][0];
				}
				else{
					memcpy(line_temp1,imin8[y-1], imsize[1] );
				}

				lineInf(line_src,line_temp1,line_temp2,imsize[1]);
				memcpy(imtemp8[y], line_temp2, imsize[1]);
			}


			// 1
			for (int y=0; y<imin.rows; y++){
				line_src = imtemp8[y];
				memcpy(line_temp1, line_src+1, imsize[1]-1);
				line_temp1[imsize[1]-1] =line_src[imsize[1]-1];
				lineInf(line_src,line_temp1,line_temp2,imsize[1]);
				memcpy(imin8[y], line_temp2, imsize[1]);
			}

			// 3
			for (int y=0; y<imin.rows-1; y++){
				line_src = imin8[y];
				if (y%2==0){
					memcpy(line_temp1+1, imin8[y+1], imsize[1]-1);
					line_temp1[0] = imin8[y+1][0];
				}
				else{
					memcpy(line_temp1, imin8[y+1], imsize[1]);
				}

				lineInf(line_src,line_temp1,line_temp2,imsize[1]);
				memcpy(imtemp8[y], line_temp2, imsize[1]);
			}

			for (int y=0; y<imin.rows; y++)
				memcpy(imin8[y], imtemp8[y], imsize[1]);
		}
		delete[] line_temp1;
		delete[] line_temp2;
	}



	for( int y = 0; y < imin.rows; y++ ){
		for( int x = 0; x < imin.cols; x++ ){
			imout.at<uchar>(y,x) = (uchar)imin8[y][x];
		}
	}
	for (int y=0; y<imin.rows; y++){
		delete[] imin8[y];
		delete[] imtemp8[y];
	}
	delete[] imin8;
	delete[] imtemp8;
}

void fastClose(Mat imin, Mat imout, int se, int size){
	Mat imtemp = imout.clone();
	fastDilate(imin,imtemp,se,size);
	fastErode(imtemp,imout,se,size);
}
void fastOpen(Mat imin, Mat imout, int se, int size){
	Mat imtemp = imout.clone();
	fastErode(imin,imtemp,se,size);
	fastDilate(imtemp,imout,se,size);
}

#endif
