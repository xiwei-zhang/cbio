/* CLAHE modified version from Karel Zuiderveld, Computer Vision Research Group,
 *	     Utrecht, The Netherlands (karel@cv.ruu.nl)
 * Coded by Xiwei ZHANG
 */

#ifndef __clahe_z
#define __clahe_z


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

void MakeLut (int *lookUpTable, int minV, int maxV, int NrBins)
/* To speed up histogram clipping, the input image [Min,Max] is scaled down to
 * [0,uiNrBins-1]. This function calculates the LUT.
 */
{
	cout<<NrBins<<" "<<minV<<" "<<maxV<<endl;
	float intv = (float)(NrBins-1)/(maxV-minV);

	for (int i=0; i<256; i++){
		if (i<minV || i>maxV) lookUpTable[i]=0;
		else {
			lookUpTable[i] = round(intv * (i-minV));
		}
	}

	
}


Mat PadImage(Mat imin, int XRes, int YRes, int XResNew, int YResNew, int XSize, int YSize){
	// padding image/////////////////////////////////////////////////
	
	cout<<XRes<<" "<<YRes<<" "<<XResNew<< " "<<YResNew<<endl;
	Mat imout = Mat::zeros(YResNew,XResNew,CV_8U);
	int meanV = meanValue(imin,1);
	imout.setTo(meanV);
	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			imout.at<uchar>(j+YSize,i+XSize) = imin.at<uchar>(j,i);
		}
	}
	imwrite("impadding.png",imout);
	return imout;
}

void UnPadImage(Mat imin, Mat imout, int XSizeOrig, int YSizeOrig){
	for (int j=0; j<imout.rows; j++){
		for (int i=0; i<imout.cols; i++){
			imout.at<uchar>(j,i) = imin.at<uchar>(j+YSizeOrig,i+XSizeOrig);
		}
	}
}

void MakeHistogram(Mat imin, unsigned long ***pulMapArray, int X, int Y, int XRes, int YRes, int XSize, int YSize, int NrBins,int *lookUpTable){
	memset(pulMapArray[X][Y],0,sizeof(unsigned long)*NrBins);
	if ((Y+1)*YSize<=YRes || (X+1)*XSize<=XRes){
		for (int j = Y*YSize; j<(Y+1)*YSize; j++){
			for (int i=X*XSize; i<(X+1)*XSize; i++){
				pulMapArray[X][Y][lookUpTable[imin.at<uchar>(j,i)]]++;
			}
		}
	}
}



void ClipHistogram(unsigned long ***pulMapArray, int X, int Y, int XRes, int YRes, int XSize, int YSize, int NrBins, unsigned long ClipLimitL){
	/* This function performs clipping of the histogram and redistribution of bins.
 * The histogram is clipped and the number of excess pixels is counted. Afterwards
 * the excess pixels are equally redistributed across the whole histogram (providing
 * the bin count is smaller than the cliplimit).
 */
	if ((Y+1)*YSize<=YRes || (X+1)*XSize<=XRes){
		unsigned long NrExcess = 0;
		long BinExcess;
		for (int i=0; i < NrBins; i++) { /* calculate total number of excess pixels */
			BinExcess = pulMapArray[X][Y][i] - ClipLimitL;
			if (BinExcess > 0) NrExcess += BinExcess;	  /* excess in current bin */
		}
		
		/* Second part: clip histogram and redistribute excess pixels in each bin */
		unsigned long BinIncr = NrExcess / NrBins;		  /* average binincrement */
		unsigned long BinRes = NrExcess % NrBins;
		for (int i=0; i<NrBins; i++){
			if (pulMapArray[X][Y][i]>ClipLimitL) pulMapArray[X][Y][i] = ClipLimitL+BinIncr;
			else pulMapArray[X][Y][i] += BinIncr;
		}
		
		if (BinRes>0){
			unsigned long stepSize = NrBins / BinRes;
			if (stepSize < 1) stepSize = 1;	 /* stepsize at least 1 */
			int count = 0;
			while (count<BinRes){
				for ( int i=0; i<NrBins && count<BinRes; i += stepSize){
					pulMapArray[X][Y][i]++;
					count ++;
				}
			}
		}
	}

}


void MapHistogram(unsigned long ***pulMapArray, int minV, int maxV, int X, int Y, int XSize, int YSize, int NrBins){
	unsigned long sumV(0);
	const float fScale = ((float)(maxV - minV)) / (XSize*YSize);
	for (int i = 0; i < NrBins; i++) {
		sumV += pulMapArray[X][Y][i]; 
		pulMapArray[X][Y][i] =(unsigned long)(minV+sumV*fScale);
		if (pulMapArray[X][Y][i] > maxV) pulMapArray[X][Y][i] = maxV;
	}
}



void Interpolate(Mat imin, Mat imout, unsigned long ***pulMapArray, int NrX, int NrY, int XRes, int YRes, int XSize, int YSize, int *lookUpTable){
	
	cout<<NrX<<" "<<NrY<<endl;
	// center part of the image.
	for (int Y = 1; Y < NrY-1; Y++) {
		for (int X = 1; X < NrX-1; X++) {
			int XL, XR, YU, YB;
			int coordXL, coordXR, coordYU, coordYB;
			int VLU,VRU,VLB,VRB,VU,VB,VI;
			int coefY=0, coefX;
			for (int j = Y*YSize; j<(Y+1)*YSize; j++){
				coefX=0;
				for (int i=X*XSize; i<(X+1)*XSize; i++){
					int curruntPixel = imin.at<uchar>(j,i);
					int mapPixel = lookUpTable[curruntPixel];
					if (coefX<XSize/2){
						if (coefY<YSize/2){   // LU
							XL=X-1; XR=X;
							YU=Y-1; YB=Y;
						}
						else{	// LB
							XL=X-1; XR=X;
							YU=Y; YB=Y+1;
						}
					}
					else{
						if (coefY<YSize/2){   //RU
							XL=X; XR=X+1;
							YU=Y-1; YB=Y;
						}
						else{		// RB
							XL=X; XR=X+1;
							YU=Y; YB=Y+1;
						}
					}
					coordXL = (XL+0.5)*XSize;
					coordXR = (XR+0.5)*XSize;
					coordYU = (YU+0.5)*YSize;
					coordYB = (YB+0.5)*YSize;

					VLU = pulMapArray[XL][YU][mapPixel];
					VRU = pulMapArray[XR][YU][mapPixel];
					VLB = pulMapArray[XL][YB][mapPixel];
					VRB = pulMapArray[XR][YB][mapPixel];
					
					VU = VLU*(coordXR-i)/XSize + VRU*(i-coordXL)/XSize;
					VB = VLB*(coordXR-i)/XSize + VRB*(i-coordXL)/XSize;
					VI = VU*(coordYB-j)/YSize + VB*(j-coordYU)/YSize;
					imout.at<uchar>(j,i) = VI;
					/*if (i==1007 && j==506) cout<<i<<" "<<j<<" "<<curruntPixel<<" "<<mapPixel<<" "<<VLU<<" "<<VRU<<" "<<VLB<<" "<<VRB<<" "<<VU<<" "<<VB<<" "<<VI<<endl;
					if (i==1008 && j==506) cout<<i<<" "<<j<<" "<<curruntPixel<<" "<<mapPixel<<" "<<VLU<<" "<<VRU<<" "<<VLB<<" "<<VRB<<" "<<VU<<" "<<VB<<" "<<VI<<endl;*/
					coefX++;
				}
				coefY++;
			}
		}
	}



	// border part of image
	for (int Y = 0; Y < NrY; Y++) {
		for (int X = 0; X < NrX; X++) {
			int XL, XR, YU, YB;
			int coordXL, coordXR, coordYU, coordYB;
			int VLU,VRU,VLB,VRB,VU,VB,VI,VL,VR;
			int coefY=0, coefX;

			
			if ((X==0 && Y==0) || (X==0 && Y==NrY-1) || (X==NrX-1 && Y==0) || (X==NrX-1 && Y==NrY-1)){ // 4 corners
				for (int j = Y*YSize; j<(Y+1)*YSize; j++){
					coefX=0;
					for (int i=X*XSize; i<(X+1)*XSize; i++){
						if (i>=XRes || j>=YRes) continue;
						int curruntPixel = imin.at<uchar>(j,i);
						int mapPixel = lookUpTable[curruntPixel];
						if ((X==0 && Y==0 && coefX>=XSize/2 && coefY>=YSize/2) || (X==NrX-1 && Y==0 && coefX<XSize/2 && coefY>=YSize/2)
							|| (X==0 && Y==NrY-1 && coefX>=XSize/2 && coefY<YSize/2) || (X==NrX-1 && Y==NrY-1 && coefX<XSize/2 && coefY<YSize/2)){  // 4 corner inner
							if (coefX<XSize/2){
								if (coefY<YSize/2){   // LU
									XL=X-1; XR=X;
									YU=Y-1; YB=Y;
								}
								else{	// LB
									XL=X-1; XR=X;
									YU=Y; YB=Y+1;
								}
							}
							else{
								if (coefY<YSize/2){   //RU
									XL=X; XR=X+1;
									YU=Y-1; YB=Y;
								}
								else{		// RB
									XL=X; XR=X+1;
									YU=Y; YB=Y+1;
								}
							}

							coordXL = (XL+0.5)*XSize;
							coordXR = (XR+0.5)*XSize;
							coordYU = (YU+0.5)*YSize;
							coordYB = (YB+0.5)*YSize;

							VLU = pulMapArray[XL][YU][mapPixel];
							VRU = pulMapArray[XR][YU][mapPixel];
							VLB = pulMapArray[XL][YB][mapPixel];
							VRB = pulMapArray[XR][YB][mapPixel];

							VU = VLU*(coordXR-i)/XSize + VRU*(i-coordXL)/XSize;
							VB = VLB*(coordXR-i)/XSize + VRB*(i-coordXL)/XSize;
							VI = VU*(coordYB-j)/YSize + VB*(j-coordYU)/YSize;
							imout.at<uchar>(j,i) = VI;
						}

						else{  // 4 corners outter
							imout.at<uchar>(j,i) = mapPixel;
						}
						coefX++;
					}
					coefY++;
				}
			}
			
			
			else {
				coefY = 0;
				if (X==0 || X==NrX-1){ // left and right border
					for (int j = Y*YSize; j<(Y+1)*YSize; j++){
						coefX=0;
						for (int i=X*XSize; i<(X+1)*XSize; i++){
							int curruntPixel = imin.at<uchar>(j,i);
							int mapPixel = lookUpTable[curruntPixel];

							if ((X==0 && coefX>=XSize/2) || (X==NrX-1 && coefX<XSize/2)){  // Left inner and right inner
								if (coefX<XSize/2){
									if (coefY<YSize/2){   // LU
										XL=X-1; XR=X;
										YU=Y-1; YB=Y;
									}
									else{	// LB
										XL=X-1; XR=X;
										YU=Y; YB=Y+1;
									}
								}
								else{
									if (coefY<YSize/2){   //RU
										XL=X; XR=X+1;
										YU=Y-1; YB=Y;
									}
									else{		// RB
										XL=X; XR=X+1;
										YU=Y; YB=Y+1;
									}
								}
								coordXL = (XL+0.5)*XSize;
								coordXR = (XR+0.5)*XSize;
								coordYU = (YU+0.5)*YSize;
								coordYB = (YB+0.5)*YSize;

								VLU = pulMapArray[XL][YU][mapPixel];
								VRU = pulMapArray[XR][YU][mapPixel];
								VLB = pulMapArray[XL][YB][mapPixel];
								VRB = pulMapArray[XR][YB][mapPixel];

								VU = VLU*(coordXR-i)/XSize + VRU*(i-coordXL)/XSize;
								VB = VLB*(coordXR-i)/XSize + VRB*(i-coordXL)/XSize;
								VI = VU*(coordYB-j)/YSize + VB*(j-coordYU)/YSize;
								imout.at<uchar>(j,i) = VI;
							}

							else{  // left and right outter
								if (coefY<YSize/2){
									YU=Y-1; YB=Y;
								}
								else{
									YU=Y; YB=Y+1;
								}
								coordYU = (YU+0.5)*YSize;
								coordYB = (YB+0.5)*YSize;
								VU = pulMapArray[X][YU][mapPixel];
								VB = pulMapArray[X][YB][mapPixel];
								VI = VU*(coordYB-j)/YSize + VB*(j-coordYU)/YSize;

								imout.at<uchar>(j,i) = VI;
							}

						
							coefX++;
						}
						coefY++;
					}
				}

				coefY=0;
				if (Y==0 || Y==NrY-1){ // up and bottom border
					for (int j = Y*YSize; j<(Y+1)*YSize; j++){
						coefX=0;
						for (int i=X*XSize; i<(X+1)*XSize; i++){
							int curruntPixel = imin.at<uchar>(j,i);
							int mapPixel = lookUpTable[curruntPixel];

							if ((Y==0 && coefY>=YSize/2) || (Y==NrY-1 && coefY<YSize/2) ){  // Upper and bottom inner
								if (coefX<XSize/2){
									if (coefY<YSize/2){   // LU
										XL=X-1; XR=X;
										YU=Y-1; YB=Y;
									}
									else{	// LB
										XL=X-1; XR=X;
										YU=Y; YB=Y+1;
									}
								}
								else{
									if (coefY<YSize/2){   //RU
										XL=X; XR=X+1;
										YU=Y-1; YB=Y;
									}
									else{		// RB
										XL=X; XR=X+1;
										YU=Y; YB=Y+1;
									}
								}
								coordXL = (XL+0.5)*XSize;
								coordXR = (XR+0.5)*XSize;
								coordYU = (YU+0.5)*YSize;
								coordYB = (YB+0.5)*YSize;

								VLU = pulMapArray[XL][YU][mapPixel];
								VRU = pulMapArray[XR][YU][mapPixel];
								VLB = pulMapArray[XL][YB][mapPixel];
								VRB = pulMapArray[XR][YB][mapPixel];

								VU = VLU*(coordXR-i)/XSize + VRU*(i-coordXL)/XSize;
								VB = VLB*(coordXR-i)/XSize + VRB*(i-coordXL)/XSize;
								VI = VU*(coordYB-j)/YSize + VB*(j-coordYU)/YSize;
								imout.at<uchar>(j,i) = VI;
							}
						
							else {   // up and bottom outter
								if (coefX<XSize/2){
									XL=X-1; XR=X;
								}
								else{
									XL=X; XR=X+1;
								}
								coordXL = (XL+0.5)*XSize;
								coordXR = (XR+0.5)*XSize;
								VL = pulMapArray[XL][Y][mapPixel];
								VR = pulMapArray[XR][Y][mapPixel];
								VI = VL*(coordXR-i)/XSize + VR*(i-coordXL)/XSize;
								imout.at<uchar>(j,i) = VI;
							}
							coefX++;
						}
						coefY++;
					}
				}
			}
		}
	}
	



}



void CLAHE (Mat imin, Mat imoutOrig, int NrX, int NrY, int NrBins, float Cliplimit, int minVOut=-1, int maxVOut=-1){


	// Pad image first!! 
	// add blocks to the border
	int XResOrig = imin.cols, YResOrig = imin.rows;
	int XSizeOrig = XResOrig/NrX, YSizeOrig = YResOrig/NrY;
	int XRes = XResOrig+2*(XResOrig/NrX), YRes = YResOrig+2*(YResOrig/NrY);

	Mat imPad = PadImage(imin,XResOrig,YResOrig,XRes,YRes,XSizeOrig,YSizeOrig);

	NrX+=2; NrY+=2;
	int XSize = XRes/NrX;
	int YSize = YRes/NrY;

	Mat imout=imPad.clone();
	imPad.copyTo(imout);

	// 0. Preparation
	

	int minV(255),maxV(0);
	for (int j=0; j<imPad.rows; j++){
		for (int i=0; i<imPad.cols; i++){
			if (minV > imPad.at<uchar>(j,i)) minV = imPad.at<uchar>(j,i);
			if (maxV < imPad.at<uchar>(j,i)) maxV = imPad.at<uchar>(j,i);
		}
	}
	if (minVOut==-1){minVOut=minV; maxVOut=maxV;}

	unsigned long ***pulMapArray = new unsigned long**[NrX];
	for (int i=0; i<NrX; i++){
		pulMapArray[i] = new unsigned long*[NrY];
		for (int j=0; j<NrY; j++){
			pulMapArray[i][j] = new unsigned long[NrBins];
		}
	}

	unsigned long NrPixels = XSize*YSize;

	unsigned long ClipLimitL;
	if(Cliplimit > 0.0) {		  /* Calculate actual cliplimit	 */
		ClipLimitL = (unsigned long) (Cliplimit * (XSize * YSize) / NrBins);
		ClipLimitL = (ClipLimitL < 1) ? 1 : ClipLimitL;
	}
	cout<<"clip limmit: "<<Cliplimit<<" "<<ClipLimitL<<endl;
	cout<<"window size: "<<XSize<<" "<<YSize<<" "<<XRes<<" "<<YRes<<endl;

	int *lookUpTable = new int[256];
	MakeLut(lookUpTable,minV,maxV,NrBins); /* Make lookup table for mapping of greyvalues */
	
	/* Calculate greylevel mappings for each contextual region */
	for (int Y = 0; Y < NrY; Y++) {
		for (int X = 0; X < NrX; X++) {
			MakeHistogram(imPad,pulMapArray,X,Y,XRes,YRes,XSize,YSize,NrBins,lookUpTable);
			ClipHistogram(pulMapArray,X,Y,XRes,YRes,XSize,YSize,NrBins,ClipLimitL);
			MapHistogram(pulMapArray, minVOut, maxVOut,X,Y,XSize,YSize,NrBins);
		}
	}

	

	/* Interpolate greylevel mappings to get CLAHE image */
	Interpolate(imPad,imout,pulMapArray,NrX,NrY,XRes,YRes,XSize,YSize,lookUpTable);
	
	UnPadImage(imout,imoutOrig,XSizeOrig,YSizeOrig);

	imwrite("imout.png",imoutOrig);


	for (int i=0; i<NrX; i++){
		for (int j=0; j<NrY; j++){
			delete[] pulMapArray[i][j];
		}
		delete[] pulMapArray[i];
	}

	delete[] pulMapArray;
	delete[] lookUpTable;
}

#endif