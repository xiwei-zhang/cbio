#ifndef __Gabor_h
#define __Gabor_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "detectROI.h"
#include "TOfft.h"
#include "ccAnalyse.h"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;


#define VM_MAX(x,y) ((x>y)?(x):(y))


class CC{
public:
	int center[2];
	list<int> p[2];
	int startP[2], endP[2];
	int area;
	int geoLen, eucLen, bboxWidth, meanWidth,maxWidth;
	float orient,diffOrient, widthVar;
	int meanV;
	int mark;
	
	CC();
	void geoLength(Mat imin);
	void lengthOrtho();
};

CC::CC(){
	mark = 1; area = 0; 
}


void CC::geoLength(Mat imin){
		/*
	For steps:
	1. Get all border pixels
	2. Get the most further pixel (to the geometric center)
	3. First propagation from the most further pixel, get the last pixel(another most further, to avoid concave case)
	4. Second propagation from the new most further pixel, get the geo-length
	*/
	int se = 6;
	int size[2] = {imin.cols,imin.rows};
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	Mat imstate = Mat::zeros(imin.rows, imin.cols, CV_8U);

	queue<int> Q[2];
	list<int>::iterator itx;
	list<int>::iterator ity;
	itx = p[0].begin();
	ity = p[1].begin();
	float dist, maxDist(0);
	int px,py,mx,my,f(0), len(0),sumX(0),sumY(0);
	list<int> temppp[2];

	// 1. Get border pixels. put into Q
	while (itx != p[0].end()){
		f = 0;

		for (int k=0; k<se; ++k){
			if (*ity%2==0){
				px = *itx + se_even[k][1];
				py = *ity + se_even[k][0];
			}
			else{
				px = *itx + se_odd[k][1];
				py = *ity + se_odd[k][0];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)==0){  // see if it's on the edge;
				f = 1;
				break;
			}
		}

		if (f==1){
			Q[0].push(*itx);
			Q[1].push(*ity);
			sumX += *itx;
			sumY += *ity;
		}

		imstate.at<uchar>(*ity,*itx) = 0;

		++itx;
		++ity;
	}

	// for perimeter
	float perimeter = (float)Q[0].size();
	// for center
	center[0] = round(sumX / float(perimeter));
	center[1] = round(sumY / float(perimeter));

	// 2. Get the pixel most far 
	//cout<<h<<" "<<i<<" "<<size[0]<<" "<<size[1]<<endl;
	while(!Q[0].empty()){
		px = Q[0].front();
		py = Q[1].front();
		dist = sqrt((float)(center[0]-px)*(center[0]-px) + (center[1]-py)*(center[1]-py));
		if (dist>=maxDist){
			maxDist = dist;
			mx = px;
			my = py;
		}
		Q[0].pop();
		Q[1].pop();
	}

	Q[0].push(mx);
	Q[1].push(my);

	while(!Q[0].empty()){
		mx = Q[0].front();
		my = Q[1].front();
		
		if (imstate.at<uchar>(my,mx)!=0){
			Q[0].pop();
			Q[1].pop();
			continue;
		}

		for (int k=0; k<se; ++k){
			if (mx==867 && my==106) cout<<" "<<k<<endl;
			if ((my%2)==0){
				px = mx + se_even[k][1];
				py = my + se_even[k][0];
			}
			else{
				px = mx+ se_odd[k][1];
				py = my + se_odd[k][0];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)>0 && imstate.at<uchar>(py,px)==0){  // see if it's on the edge;
				Q[0].push(px);
				Q[1].push(py);
			}
		}
		imstate.at<uchar>(my,mx) = 1;
		Q[0].pop();
		Q[1].pop();
	}

	Q[0].push(mx);
	Q[1].push(my);

	Q[0].push(-1); // -1 is a mark point
	Q[1].push(-1);
	imstate.at<uchar>(my,mx) = 2;
	startP[0] = mx;
	startP[1] = my;
	// cout<<p3[0]<<" "<<p3[1]<<endl;

	// 4. Second propagation
	while(!Q[0].empty()){
		mx = Q[0].front();
		my = Q[1].front();

		if (mx == -1) {  // if the mark point pop out, one iteration is done, len ++
			++len;
			Q[0].pop();
			Q[1].pop();
			if (Q[0].empty()) break;
			Q[0].push(-1);
			Q[1].push(-1);
			mx = Q[0].front();
			my = Q[1].front();
		}
		endP[0] = mx;
		endP[1] = my;

		f = 0;
		for (int k=0; k<se; ++k){
			if (my%2==0){
				px = mx + se_even[k][1];
				py = my + se_even[k][0];
			}
			else{
				px = mx + se_odd[k][1];
				py = my + se_odd[k][0];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)>0 && imstate.at<uchar>(py,px)==1){
				Q[0].push(px);	
				Q[1].push(py);
			
				imstate.at<uchar>(py,px) = 2;
				f = 1;
			}
		}

	//	imstate[my][mx] = 2;
		Q[0].pop();
		Q[1].pop();
	}

	geoLen = len;
	eucLen = sqrt(pow(float(startP[0]-endP[0]),2) + pow(float(startP[1]-endP[1]),2));

	float dx = startP[0] - endP[0];
	float dy = -(startP[1] - endP[1]);
	if (dx<0) {dx = -dx; dy = -dy;} //  -pi/2 ~ pi/2
	orient = atan2(dy,dx);

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;

}


void CC::lengthOrtho(){
	// cout<<" "<<startP[0]<<" "<<startP[1]<<" "<<endP[0]<<" "<<endP[1]<<" "<<pp[0].size()<<endl;

	/* Axis rotation:
		(x') = (cosA -sinA)   (x)
		(y')   (sinA cosA )   (y)
		x' = x cosA - y sinA
		y' = x sinA + y cosA
	*/
	float A,minVx(9999),maxVx(-9999),minVy(9999),maxVy(-9999),x_,y_;
	list<int> temp[2];

	if ((startP[0] - endP[0])==0) A=3.1415f/2;
	else{
		A = float(3.1415 - atan2(float(endP[1]-startP[1]),float(endP[0]-startP[0])));
	}

	list<int>::iterator itx;
	list<int>::iterator ity;
	itx = p[0].begin();
	ity = p[1].begin();				
	while(itx!=p[0].end()){
		x_ = *itx * cos(A) - *ity * sin(A);
		y_ = *itx * sin(A) + *ity * cos(A);
		if (y_<=minVy) {minVy = y_;}
		if (y_>=maxVy) {maxVy = y_;}
		if (x_<=minVx) {minVx = x_;}
		if (x_>=maxVx) {maxVx = x_;}
		temp[0].push_back(round(x_));
		temp[1].push_back(round(y_));
		itx++;
		ity++;
	}
	bboxWidth = round(maxVy - minVy);

	
	//#########################
	// for the mean width
	float w(0),meanNorm;
	int n(0),x,y,x0,minVY(9999),maxVY(-9999);
	list<int> wd; //width
	temp[0].push_back(99999999);
	temp[1].push_back(99999999);
	for (int i=round(minVx); i<=round(maxVx); i++){
		while(temp[0].front() != 99999999){
			x = temp[0].front();
			y = temp[1].front();
			temp[0].pop_front();
			temp[1].pop_front();
			if (x==i){
				if (y<=minVY) minVY=y;
				if (y>=maxVY) maxVY=y;
			}
			else{
				temp[0].push_back(x);
				temp[1].push_back(y);
			}
		}
		if (maxVY!=-9999)
			wd.push_back(maxVY - minVY+1);
		temp[0].pop_front();
		temp[1].pop_front();
		if (temp[0].empty()) break;
		x0 = temp[0].front();
		minVY = temp[1].front();
		maxVY = temp[1].front();
		temp[0].pop_front();
		temp[1].pop_front();
		temp[0].push_back(99999999);
		temp[1].push_back(99999999);
	}
	list<int>::iterator it = wd.begin();
	while(it != wd.end()){
		w+=*it;
		it++;
	}
	
	if (wd.size()==0) {
		meanWidth = 0;
	}
	else {
		meanWidth = w/wd.size();
	}
	// VAR
	wd.sort();
	meanNorm = float(meanWidth)/wd.back();
	it = wd.begin();
	w = 0;
	while(it != wd.end()){
		w += pow((float(*it)/wd.back() - meanNorm),2);
		it++;
	}
	widthVar = w/(wd.size());
	maxWidth = wd.back();
}



Mat creatKernel(double sigma, double theta, double lambda, double psi, double gamma){
	int Xmax,Ymax,dx,dy,nstds=3;
	double dXmax,dYmax;
		
		
	//**********************
	//Generation of the kernel
	//**********************
	double sigma_x = sigma;
	double sigma_y = sigma*gamma;

	//Bounding box
	dXmax = VM_MAX(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
	dYmax = VM_MAX(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
	Xmax = (int) VM_MAX(1, ceil(dXmax));
	Ymax = (int) VM_MAX(1, ceil(dYmax));
	dx = 2*Xmax + 1;
	dy = 2*Ymax + 1;

	double **x_theta = new double* [dy];
	double **y_theta = new double* [dy];
	for (int i=0; i<dy; i++){
		x_theta[i] = new double[dx];
		y_theta[i] = new double[dx];
	}

	//2D Rotation
	for(int i=0;i<dx;i++){
		for(int j=0;j<dy;j++){
			x_theta[j][i] = (i-dx/2)*cos(theta) + (j-dy/2)*sin(theta);
			y_theta[j][i] = -(i-dx/2)*sin(theta) + (j-dy/2)*cos(theta);
		}
	}
	
	Mat imKernel = Mat::zeros(dy,dx,CV_64F);
	//T1 *gabor = imOut.rawPointer();
	for(int j=0;j<dy;j++){
		for(int i=0;i<dx;i++){
			imKernel.at<double>(j,i) = exp(-0.5 * ((x_theta[j][i]*x_theta[j][i])/(sigma_x*sigma_x) + (y_theta[j][i]*y_theta[j][i])/(sigma_y*sigma_y)))*
			cos(2*3.14159/lambda*x_theta[j][i]+psi);
		}
	}

	for (int i=0; i<dy; i++){
		delete[] x_theta[i];
		delete[] y_theta[i];
	}

	return imKernel;
}

void displayKernel(Mat imKernel, char* filename){
	double maxV,minV;
	maxV = imKernel.at<double>(0,0);
	minV = imKernel.at<double>(0,0);
	Mat imout = Mat::zeros(imKernel.rows, imKernel.cols, CV_8U);
	for (int j=0; j<imKernel.rows; j++){
		for (int i=0; i<imKernel.cols; i++){
			if (minV>imKernel.at<double>(j,i)) minV = imKernel.at<double>(j,i);
			if (maxV<imKernel.at<double>(j,i)) maxV = imKernel.at<double>(j,i);
		}
	}

	if((maxV - minV) == 0)
		imwrite(filename,imout);
	else{
		for (int j=0; j<imKernel.rows; j++){
			for (int i=0; i<imKernel.cols; i++){
				imout.at<uchar>(j,i) = int((imKernel.at<double>(j,i) - minV) / (maxV - minV) * 255);
			}
		}
		imwrite(filename,imout);
	}
}

void filterGabor(Mat imin, Mat imout,double sigma, double theta, double lambda, double psi, double gamma){

	int Xmax,Ymax,dx,dy,nstds=3;
	double dXmax,dYmax;	
		
	//**********************
	//Generation of the kernel
	//**********************
	double sigma_x = sigma;
	double sigma_y = sigma*gamma;

	//Bounding box
	dXmax = VM_MAX(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
	dYmax = VM_MAX(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
	Xmax = (int) VM_MAX(1, ceil(dXmax));
	Ymax = (int) VM_MAX(1, ceil(dYmax));
	dx = 2*Xmax + 1;
	dy = 2*Ymax + 1;

	cout<<dx<<" "<<dy<<endl;

	double **x_theta = new double* [dy];
	double **y_theta = new double* [dy];
	for (int i=0; i<dy; i++){
		x_theta[i] = new double[dx];
		y_theta[i] = new double[dx];
	}

	//2D Rotation
	for(int i=0;i<dx;i++){
		for(int j=0;j<dy;j++){
			x_theta[j][i] = (i-dx/2)*cos(theta) + (j-dy/2)*sin(theta);
			y_theta[j][i] = -(i-dx/2)*sin(theta) + (j-dy/2)*cos(theta);
		}
	}
	
	Mat imKernel = Mat::zeros(dy,dx,CV_64F);
	//T1 *gabor = imOut.rawPointer();
	for(int j=0;j<dy;j++){
		for(int i=0;i<dx;i++){
			imKernel.at<double>(j,i) = exp(-0.5 * ((x_theta[j][i]*x_theta[j][i])/(sigma_x*sigma_x) + (y_theta[j][i]*y_theta[j][i])/(sigma_y*sigma_y)))*
			cos(2*3.14159/lambda*x_theta[j][i]+psi);
		}
	}

	for (int i=0; i<dy; i++){
		delete[] x_theta[i];
		delete[] y_theta[i];
	}


	//////////////////////////////////////////////////////////////////////////
	// start convolution
	//////////////////////////////////////////////////////////////////////////


	int H = imin.rows;
	int W = imin.cols;
	int I,J;
	double D;

	for(int j=0;j<H;j++){
		for(int i=0;i<W;i++){
			D=0;
			for(int k=-dx/2;k<=dx/2;k++){
				for(int l=-dy/2;l<=dy/2;l++){
					I=(i+k)%W;
					J=(j+l)%H;
					if(I<0)I+=W;				//Mirror
					if(J<0)J+=H;

					D += imKernel.at<double>((l+dy/2), (k+dx/2)) * imin.at<uchar>(J,I);
				}
			}
			imout.at<double>(j,i) = D;
		}
	}
}

void gaborNorm(Mat imGabor, Mat imout){
	double maxV = imGabor.at<double>(0,0), minV = imGabor.at<double>(0,0);
	for (int j=0; j<imGabor.rows; j++){
		for (int i=0; i<imGabor.cols; i++){
			if (maxV<imGabor.at<double>(j,i)) maxV = imGabor.at<double>(j,i);
			if (minV>imGabor.at<double>(j,i)) minV = imGabor.at<double>(j,i);
		}
	}
	if ((maxV - minV)==0) imwrite("imGabor.png",imout);
	else{
		for (int j=0; j<imGabor.rows; j++){
			for (int i=0; i<imGabor.cols; i++){
				imout.at<uchar>(j,i) = int((imGabor.at<double>(j,i) - minV)/(maxV - minV)*255);
			}
		}
		imwrite("imGabor.png",imout);
	}
}

float circVar(queue<float> orientList){
	float sCos(0),sSin(0), R;
	int len = 0;
	while (!orientList.empty()){
		sCos += cos(2*orientList.front());
		sSin += sin(2*orientList.front());
		len ++;
		orientList.pop();
	}

	R = sqrt(sCos*sCos + sSin*sSin) / len;

	return 1-R;
}

void extractMainVessel(Mat imASF, Mat imGaborSup, Mat imOrient, Mat imMainVessel, int* imInfo, int C_area, int C_inten, int searchAngle){

	float interv = 3.1415 / 12;
	float angleList[12] = {interv*6, interv*5, interv*4, interv*3, interv*2, interv*1, 0, -interv*1, -interv*2, -interv*3, -interv*4, -interv*5};

	Mat imtemp1 = imGaborSup.clone();
	Mat imtemp2 = imGaborSup.clone();
	Mat imtemp3 = imGaborSup.clone();
	Mat imVessel = imGaborSup.clone();
	Mat imtemp32_1 = Mat::zeros(imGaborSup.rows, imGaborSup.cols, CV_32S);

	for (int k=0; k<12; k++){

		int i0,i1;
		i1 = (k+13)%12+1;
		i0 = (k-1+12)%12+1;

		imCompare(imOrient,i0,0,255,0,imtemp1);
		imCompare(imOrient,i1,0,255,0,imtemp2);
		imSup(imtemp2,imtemp1,imtemp2);
		imCompare(imOrient,k+1,0,255,0,imtemp1);
		imSup(imtemp1,imtemp2,imtemp1);
		binAreaSelection(imtemp1,imtemp2,6,C_area);
		subtract(imtemp1,imtemp2,imtemp1);

		// fill reflection gap in the vessels
		Dilate(imtemp1,imtemp2,6,imInfo[2]/5);
		Erode(imtemp2,imtemp3,6,imInfo[2]/5 +1);
		imSup(imtemp3,imtemp1,imtemp1);

		Label(imtemp1,imtemp32_1,6);
		int N = labelCount(imtemp32_1);
		double *sumV = new double[N];
		int *np = new int[N];
		memset(sumV,0,sizeof(double)*N);
		memset(np,0,sizeof(int)*N);

		CC *candidate = new CC[N];

		for (int j=0; j<imGaborSup.rows; j++){
			for (int i=0; i<imGaborSup.cols; i++){
				if (imtemp32_1.at<int>(j,i)==0) continue;
				np[imtemp32_1.at<int>(j,i)-1]++;
				sumV[imtemp32_1.at<int>(j,i)-1] += double(imASF.at<uchar>(j,i));
				candidate[imtemp32_1.at<int>(j,i)-1].p[0].push_back(i);
				candidate[imtemp32_1.at<int>(j,i)-1].p[1].push_back(j);
			}
		}

		for (int i=0; i<N; i++){
			candidate[i].area = np[i];
			candidate[i].meanV = sumV[i]/np[i];
			candidate[i].geoLength(imtemp1);
			candidate[i].lengthOrtho();
			candidate[i].diffOrient = abs(candidate[i].orient - angleList[k]);
			if ( candidate[i].diffOrient > 3.1415/2) candidate[i].diffOrient = 3.1415 - candidate[i].diffOrient;

			//selection
			float C_orient = 0.4;
			C_inten = 3;

			if (candidate[i].geoLen > imInfo[0]/2){
				candidate[i].mark = 1;
			}
			else{
				if (candidate[i].diffOrient <= C_orient) {
					if (candidate[i].meanV >= C_inten){
						if (candidate[i].maxWidth < 2*imInfo[2]){
							candidate[i].mark = 1;
						}
						else{
							if ((float(candidate[i].maxWidth) / candidate[i].meanWidth)<2.5 && candidate[i].widthVar<0.06){
								candidate[i].mark=1;
							}
							else {
								candidate[i].mark = 0;
							}
						}
					}
					else if (candidate[i].meanV > 1){
						if (candidate[i].maxWidth<imInfo[2]/2 && (float(candidate[i].eucLen)/candidate[i].geoLen)>0.85 && candidate[i].geoLen>imInfo[2]*2){
							candidate[i].mark = 1;
						}
						else{
							candidate[i].mark = 0;
						}
					}
					else{
						candidate[i].mark = 0;
					}
				}
				else{
					candidate[i].mark=0;
				}
			}
		}

		for (int j=0; j<imGaborSup.rows; j++){
			for (int i=0; i<imGaborSup.cols; i++){
				if (imtemp32_1.at<int>(j,i)==0) continue;
				if (candidate[imtemp32_1.at<int>(j,i)-1].mark == 0) imtemp1.at<uchar>(j,i)=100;
			}
		}

		//imtemp1.copyTo(imMainVessel);
		imCompare(imtemp1,200,1,k+1,imMainVessel,imMainVessel);
	
		int nn = 0;

		//cout<<candidate[nn].center[0]<<" "<<candidate[nn].center[1]<<" "<<candidate[nn].geoLen<<" "<<candidate[nn].eucLen<<endl;
		//cout<<candidate[nn].meanV<<endl;
		//cout<<candidate[nn].bboxWidth<<" "<<candidate[nn].meanWidth<<" "<<candidate[nn].widthVar<<" "<<candidate[nn].maxWidth<<endl;
		//cout<<candidate[nn].orient<<" "<<candidate[nn].startP[0]<<" "<<candidate[nn].startP[1]<<" "<<candidate[nn].endP[0]<<" "<<candidate[nn].endP[1]<<endl;
		//cout<<angleList[k]<<" "<<candidate[nn].diffOrient<<endl;

		delete[] candidate;
		delete[] sumV;
		delete[] np;
	}

	
	imCompare(imMainVessel,0,0,0,imOrient,imMainVessel);
	imwrite("imMainVesselOrient.png",imMainVessel);
	imCompare(imMainVessel,0,0,0,imGaborSup,imMainVessel);
	imwrite("imMainVesselGb.png",imMainVessel);
	imCompare(imMainVessel,0,0,0,imASF,imMainVessel);
	imwrite("imMainVesselASF.png",imMainVessel);

	// Circular variance 
	// ref: http://www.ebi.ac.uk/thornton-srv/software/PROCHECK/nmr_manual/man_cv.html
	int wsize = imInfo[2]; // half
	int x,y;
	Mat imCircVar = Mat::zeros(imGaborSup.rows, imGaborSup.cols, CV_8U);
	for (int j=0; j<imGaborSup.rows; j++){
		for (int i=0; i<imGaborSup.cols; i++){
			if (imMainVessel.at<uchar>(j,i)==0) continue;
			queue <float> orientList;
			for (int m = -wsize; m<=wsize; m++){
				for (int n = -wsize; n<wsize; n++){
					x = i+m;
					y = j+n;
					if (x<0 || x>=imGaborSup.cols || y<0 || y>=imGaborSup.rows) continue;
					if (imMainVessel.at<uchar>(y,x) > 0)
						orientList.push(angleList[imMainVessel.at<uchar>(y,x)-1]);
				}
			}
			imCircVar.at<uchar>(j,i) = circVar(orientList)*255;
		}
	}
	imwrite("imCircVar.png",imCircVar);
}


void smallVessel(Mat imMainVessel, Mat imOrient, int *imInfo){
	Mat imtemp1 = imMainVessel.clone();
	Mat imtemp2 = imMainVessel.clone();
	Mat imtemp32_1 = Mat::zeros(imMainVessel.rows, imMainVessel.cols, CV_32S);

	threshold(imMainVessel,imtemp1,0,255,0);
	//Label(imtemp1,imtemp32_1,6);
	//int *hist = histogram(imtemp32_1);
	binAreaSelection(imtemp1,imtemp2,6,imInfo[2]*imInfo[1]/2);
	imwrite("z1.png",imtemp2);
	subtract(imtemp1,imtemp2,imtemp2);
	imwrite("z2.png",imtemp2);
}

void fastGaborFilter(Mat imin, Mat imout, double sigma, double theta, double lambda, double psi, double gamma){
	// depth 8->uchar    33->float
	// initialization /////////////////////////////////////////////////
	long lWidth = imin.cols;
	long lHeight = imin.rows;
	long w,h; //fft width and hight. should be pow of 2
	int wp,hp;

	w=1; h=1; wp=0; hp=0;
	// calculate the width and height of FFT trans
	while (w <= lWidth){
		w *=2;
		wp ++;
	}
	while (h <= lHeight){
		h *=2;
		hp ++;
	}
	/////////////////////////////////////////////////




	// padding image/////////////////////////////////////////////////
	Mat imPad = Mat::zeros(h,w,CV_64F);
	Mat imKernalPad = Mat::zeros(h,w,CV_64F);
	int meanV = meanValue(imin,1);
	imPad.setTo(meanV);
	int xx0 = imin.cols/2, yy0 = imin.rows/2, x0_=w/2, y0_=h/2;
	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			imPad.at<double>(j-yy0+y0_,i-xx0+x0_) = imin.at<uchar>(j,i);
		}
	}



	// padding kernel////////////////////////
	int Xmax,Ymax,dx,dy,nstds=3;
	double dXmax,dYmax;	
		
	//**********************
	//Generation of the kernel
	//**********************
	double sigma_x = sigma;
	double sigma_y = sigma*gamma;

	//Bounding box
	dXmax = VM_MAX(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
	dYmax = VM_MAX(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
	Xmax = (int) VM_MAX(1, ceil(dXmax));
	Ymax = (int) VM_MAX(1, ceil(dYmax));
	dx = 2*Xmax + 1;
	dy = 2*Ymax + 1;


	double **x_theta = new double* [dy];
	double **y_theta = new double* [dy];
	for (int i=0; i<dy; i++){
		x_theta[i] = new double[dx];
		y_theta[i] = new double[dx];
	}
	

	//2D Rotation
	for(int i=0;i<dx;i++){
		for(int j=0;j<dy;j++){
			x_theta[j][i] = (i-dx/2)*cos(theta) + (j-dy/2)*sin(theta);
			y_theta[j][i] = -(i-dx/2)*sin(theta) + (j-dy/2)*cos(theta);
		}
	}

	Mat imKernel = Mat::zeros(dy,dx,CV_64F);
	//T1 *gabor = imOut.rawPointer();
	for(int j=0;j<dy;j++){
		for(int i=0;i<dx;i++){
			imKernel.at<double>(j,i) = exp(-0.5 * ((x_theta[j][i]*x_theta[j][i])/(sigma_x*sigma_x) + (y_theta[j][i]*y_theta[j][i])/(sigma_y*sigma_y)))*
			cos(2*3.14159/lambda*x_theta[j][i]+psi);
		}
	}

	for (int i=0; i<dy; i++){
		delete[] x_theta[i];
		delete[] y_theta[i];
	}


	xx0 = imKernel.cols/2; yy0 = imKernel.rows/2; x0_=w/2; y0_=h/2;
	for (int j=0; j<imKernel.rows; j++){
		for (int i=0; i<imKernel.cols; i++){
			//cout<<j<<" "<<i<<" "<<j-yy0+y0_<<" "<<i-xx0+x0_<<endl;
			imKernalPad.at<double>(j-yy0+y0_,i-xx0+x0_) = imKernel.at<double>(j,i);
		}
	}

	


	//int dwKernal = (size-1)/2;
	//for (int j=-dwKernal; j<=dwKernal; j++){
	//	for (int i=-dwKernal; i<=dwKernal; i++){
	//		imKernalPad.at<float>(h/2+j,w/2+i) = 1.0f/(size*size);
	//	}
	//}
	/////////////////////////////////////////////////


	// FFT/////////////////////////////////////////////////
	complex<double> *TD = new complex<double>[w*h];
	complex<double> *FD = new complex<double>[w*h];
	complex<double> *kTD = new complex<double>[w*h];
	complex<double> *kFD = new complex<double>[w*h];

	for (int j=0; j<h; j++){
		for (int i=0; i<w; i++){
			TD[i+j*w]= complex<double>(imPad.at<double>(j,i),0);
			kTD[i+j*w]= complex<double>(imKernalPad.at<double>(j,i),0);
		}
	}
	// y direction FFT
	for(int j=0; j<h; j++) FFT(&TD[w*j], &FD[w*j], wp);
	for(int j=0; j<h; j++) FFT(&kTD[w*j], &kFD[w*j], wp);
	// transform
	for(int j=0; j<h; j++){
		for(int i=0; i<w; i++){
			TD[j+h*i] = FD[i+w*j];
			kTD[j+h*i] = kFD[i+w*j];
		}
	}
	// x direction FFT
	for (int i=0; i<w; i++) FFT(&TD[h*i], &FD[h*i], hp);
	for (int i=0; i<w; i++) FFT(&kTD[h*i], &kFD[h*i], hp);
	/////////////////////////////////////////////////


	// Product in frequency domain//////////////////
	for (int i=0; i<h*w; i++)
		FD[i]= FD[i]*kFD[i];
	/////////////////////////////////////////////////

	// Amplitude
	//for (int j=0; j<h; j++){
	//	for (int i=0; i<w; i++){
	//		dTemp = sqrt(FD[i*h +j].real() * FD[i*h+j].real() + FD[i*h +j].imag()*FD[i*h +j].imag())/100; // /100;
	//		if (dTemp>255) dTemp=255;
	//		imPad.at<uchar>(j,i) = dTemp;
	//	}
	//}
	//imwrite("fft3.png",imPad);

	///IFFT ////////////////////////////////////////
	// IFFT x
	for (int i=0; i<w; i++) IFFT(&FD[h*i], &TD[h*i], hp);
	// transform
	for(int j=0; j<h; j++)
		for(int i=0; i<w; i++)
			FD[i+w*j] = TD[j+h*i];
	// IFFT y
	for(int j=0; j<h; j++) IFFT(&FD[w*j], &TD[w*j], wp);

	//for (int j=0; j<imin.rows; j++){
	//	for (int i=0; i<imin.cols; i++){
	//		imout.at<uchar>(j,i) = round(TD[(i<w/2?w/2+i:i-w/2)+(j<h/2?h/2+j:j-h/2)*w].real());
	//	}
	//}
	for (int j=0; j<h; j++){
		for (int i=0; i<w; i++){
			imPad.at<double>(j,i) = round(TD[(i<w/2?w/2+i:i-w/2)+(j<h/2?h/2+j:j-h/2)*w].real());
		}
	}
	xx0 = imin.cols/2; yy0 = imin.rows/2; x0_=w/2; y0_=h/2;
	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			imout.at<double>(j,i) = imPad.at<double>(j-yy0+y0_,i-xx0+x0_);
		}
	}
	/////////////////////////////////////////////////


	delete[] TD;
	delete[] FD;
	delete[] kTD;
	delete[] kFD;
}



vector<Mat> fastGaborFilterVessel(Mat imASF, Mat imGaborSup, double sigma, double lambda, double psi, double gamma){
	
	Mat imGabor = Mat::zeros(imASF.rows, imASF.cols, CV_64F);
	Mat imout = Mat::zeros(imASF.rows, imASF.cols, CV_8U);
	vector<Mat> gaborArray;


	// initialization /////////////////////////////////////////////////
	long lWidth = imASF.cols;
	long lHeight = imASF.rows;
	long w,h; //fft width and hight. should be pow of 2
	int wp,hp;

	w=1; h=1; wp=0; hp=0;
	// calculate the width and height of FFT trans
	while (w <= lWidth){
		w *=2;
		wp ++;
	}
	while (h <= lHeight){
		h *=2;
		hp ++;
	}
	/////////////////////////////////////////////////


	// padding image/////////////////////////////////////////////////
	Mat imPad = Mat::zeros(h,w,CV_64F);
	Mat imKernalPad = Mat::zeros(h,w,CV_64F);
	int meanV = meanValue(imASF,1);
	imPad.setTo(meanV);
	int xx0 = imASF.cols/2, yy0 = imASF.rows/2, x0_=w/2, y0_=h/2;
	for (int j=0; j<imASF.rows; j++){
		for (int i=0; i<imASF.cols; i++){
			imPad.at<double>(j-yy0+y0_,i-xx0+x0_) = imASF.at<uchar>(j,i);
		}
	}

	// FFT for original padded image
	complex<double> *TD = new complex<double>[w*h];
	complex<double> *FD = new complex<double>[w*h];
	for (int j=0; j<h; j++){
		for (int i=0; i<w; i++){
			TD[i+j*w]= complex<double>(imPad.at<double>(j,i),0);
		}
	}
	// y direction FFT
	for(int j=0; j<h; j++) FFT(&TD[w*j], &FD[w*j], wp);
	// transform
	for(int j=0; j<h; j++){
		for(int i=0; i<w; i++){
			TD[j+h*i] = FD[i+w*j];
		}
	}
	// x direction FFT
	for (int i=0; i<w; i++) FFT(&TD[h*i], &FD[h*i], hp);

	Mat imKernal;

	for (int i=0; i<12; i++){
		cout<<"Gabor Orient: "<<i<<endl;
		
		double theta = 3.14/12*i;
		
		//**********************
		//Generation of the kernel
		//**********************
		// padding kernel////////////////////////
		int Xmax,Ymax,dx,dy,nstds=3;
		double dXmax,dYmax;	

		double sigma_x = sigma;
		double sigma_y = sigma*gamma;

		//Bounding box
		dXmax = VM_MAX(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
		dYmax = VM_MAX(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
		Xmax = (int) VM_MAX(1, ceil(dXmax));
		Ymax = (int) VM_MAX(1, ceil(dYmax));
		dx = 2*Xmax + 1;
		dy = 2*Ymax + 1;


		double **x_theta = new double* [dy];
		double **y_theta = new double* [dy];
		for (int i=0; i<dy; i++){
			x_theta[i] = new double[dx];
			y_theta[i] = new double[dx];
		}
	

		//2D Rotation
		for(int i=0;i<dx;i++){
			for(int j=0;j<dy;j++){
				x_theta[j][i] = (i-dx/2)*cos(theta) + (j-dy/2)*sin(theta);
				y_theta[j][i] = -(i-dx/2)*sin(theta) + (j-dy/2)*cos(theta);
			}
		}

		Mat imKernel = Mat::zeros(dy,dx,CV_64F);
		//T1 *gabor = imOut.rawPointer();
		for(int j=0;j<dy;j++){
			for(int i=0;i<dx;i++){
				imKernel.at<double>(j,i) = exp(-0.5 * ((x_theta[j][i]*x_theta[j][i])/(sigma_x*sigma_x) + (y_theta[j][i]*y_theta[j][i])/(sigma_y*sigma_y)))*
				cos(2*3.14159/lambda*x_theta[j][i]+psi);
			}
		}

		for (int i=0; i<dy; i++){
			delete[] x_theta[i];
			delete[] y_theta[i];
		}


		xx0 = imKernel.cols/2; yy0 = imKernel.rows/2; x0_=w/2; y0_=h/2;
		for (int j=0; j<imKernel.rows; j++){
			for (int i=0; i<imKernel.cols; i++){
				imKernalPad.at<double>(j-yy0+y0_,i-xx0+x0_) = imKernel.at<double>(j,i);
			}
		}

		complex<double> *kTD = new complex<double>[w*h];
		complex<double> *kFD = new complex<double>[w*h];
		for (int j=0; j<h; j++){
			for (int i=0; i<w; i++){
				kTD[i+j*w]= complex<double>(imKernalPad.at<double>(j,i),0);
			}
		}
		// y direction FFT
		for(int j=0; j<h; j++) FFT(&kTD[w*j], &kFD[w*j], wp);
		// transform
		for(int j=0; j<h; j++){
			for(int i=0; i<w; i++){
				kTD[j+h*i] = kFD[i+w*j];
			}
		}
		// x direction FFT
		for (int i=0; i<w; i++) FFT(&kTD[h*i], &kFD[h*i], hp);

		// Product in frequency domain//////////////////
		complex<double> *rFD = new complex<double>[w*h];
		complex<double> *rTD = new complex<double>[w*h];
		for (int i=0; i<h*w; i++)
			rFD[i]= FD[i]*kFD[i];
		

		///IFFT ////////////////////////////////////////
		// IFFT x
		for (int i=0; i<w; i++) IFFT(&rFD[h*i], &rTD[h*i], hp);
		// transform
		for(int j=0; j<h; j++)
			for(int i=0; i<w; i++)
				rFD[i+w*j] = rTD[j+h*i];
		// IFFT y
		for(int j=0; j<h; j++) IFFT(&rFD[w*j], &rTD[w*j], wp);

	
		Mat imResultPad = imPad.clone();
		for (int j=0; j<h; j++){
			for (int i=0; i<w; i++){
				imResultPad.at<double>(j,i) = round(rTD[(i<w/2?w/2+i:i-w/2)+(j<h/2?h/2+j:j-h/2)*w].real());
			}
		}
		xx0 = imASF.cols/2; yy0 = imASF.rows/2; x0_=w/2; y0_=h/2;
		for (int j=0; j<imASF.rows; j++){
			for (int i=0; i<imASF.cols; i++){
				imGabor.at<double>(j,i) = imResultPad.at<double>(j-yy0+y0_,i-xx0+x0_);
			}
		}
		/////////////////////////////////////////////////

		//fastGaborFilter(imASF,imGabor,1,theta,20,0,2); //1 2 for small image

		//fastGaborFilter(imgreen,imGabor,1,theta,20,3.14,3);
		gaborNorm(imGabor,imout);

		// push into array for following precessing
		Mat imtemp;
		imGabor.copyTo(imtemp);
		gaborArray.push_back(imtemp);

		char number[2];
		char filename[20];
		memset(filename,0,sizeof(filename));
		memset(number,0,sizeof(number));
		strcat(filename,"Gabor");
		itoa(i,number,10);
		strcat(filename,number);
		strcat(filename,".png");
		imwrite(filename,imout);

		char kn[20];
		memset(kn,0,sizeof(kn));
		strcat(kn,"kernel");
		strcat(kn,number);
		strcat(kn,".png");

		imKernel = creatKernel(1,theta,20,0,2);
		displayKernel(imKernel,kn);

		imSup(imGaborSup,imout,imGaborSup);
		delete[] kTD;
		delete[] kFD;
		delete[] rFD;
		delete[] rTD;
		
	}
	imwrite("imSup.png",imGaborSup);
	delete[] TD;
	delete[] FD;
	return gaborArray;
}

#endif