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
#include "Gabor.h"

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




class MACand{
public:
	int labelref;
	int center[2];
	int maxiPos[2];
	list<int> p[2];
	int minX,maxX,minY,maxY;
	int W,H;
	float WH[20];
	int length[20];
	float circ[20];
	int area2[20];
	int area;
	float medianRes[20];
	int isGT;

	float maxRes;
	float meanRes;
	float meanVessel;
	float inVessel;
	float ratioMeanVessel;
	float ratioInVessel;
	

	int n_winS_thL_ccS,n_winS_thL_ccL,n_winL_thL_ccS,n_winL_thL_ccL,
		n_winS_thH_ccS,n_winS_thH_ccL,n_winL_thH_ccS,n_winL_thH_ccL;

	int p1[2];
	int p2[2];

	MACand();
	void vesselAnalyse(Mat imASF, Mat maCandiRaw, Mat imMainVesselASF, Mat imMainVesselOrient, Mat imlabel, int *imInfo );
	void envAnalyse(Mat imASF, Mat maCandiRaw, int* imInfo);
};

MACand::MACand(){
	center[0] = 0; center[1] = 0; maxiPos[0]=0; maxiPos[1]=0;
	meanRes = 0; meanVessel = 0; maxRes = 0;
	inVessel = 0;
	area = 0;
	W=0; H=0;
	minX=99999; minY=99999; maxX=-1; maxY=-1;
	isGT=0;
	n_winS_thL_ccS=0; n_winS_thL_ccL=0; n_winL_thL_ccS=0; n_winL_thL_ccL=0;
	n_winS_thH_ccS=0; n_winS_thH_ccL=0; n_winL_thH_ccS=0; n_winL_thH_ccL=0;
	for (int i=0; i<20; i++){
		WH[i] = 0.0;
		length[i] = 10.0;
		circ[i] = 0.0;
		area2[i] = 100.0;
		medianRes[i] = 0.0;
	}
}

void MACand::vesselAnalyse(Mat imASF, Mat maCandiRaw, Mat imMainVesselASF, Mat imMainVesselOrient, Mat imlabel, int *imInfo){
	// parameters:
	int windowSize = imInfo[2]*3; // This is half size
	//=================
	// 1. nearest vessel mean value 
	int startX = center[0]-windowSize;
	int startY = center[1]-windowSize;
	int endX = center[0]+windowSize;
	int endY = center[1]+windowSize;
	if (startX<0) startX=0;
	if (startY<0) startY=0;
	if (endX>=imASF.cols) endX=imASF.cols-1;
	if (endY>=imASF.rows) endY=imASF.rows-1;

	/*cout<<startX<<" "<<endX<<" "<<startY<<" "<<endY<<" ";*/
	int np(0);
	for (int j=startY; j<=endY; j++){
		for (int i=startX; i<=endX; i++){
			if (imlabel.at<int>(j,i) == (labelref+1)) continue;
			if (imMainVesselASF.at<uchar>(j,i)>0){
				meanVessel += imASF.at<uchar>(j,i);
				np++;
			}
		}
	}
	if (np==0) meanVessel = 0;
	else meanVessel /= np;
	
	// 2. is in the vessel
	windowSize = imInfo[2];
	startX = center[0]-windowSize;
	startY = center[1]-windowSize;
	endX = center[0]+windowSize;
	endY = center[1]+windowSize;
	if (startX<0) startX=0;
	if (startY<0) startY=0;
	if (endX>=imASF.cols) endX=imASF.cols-1;
	if (endY>=imASF.rows) endY=imASF.rows-1;

	list<int>::iterator it1;
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	int orient[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
	while (it1 != p[0].end()){
		orient[imMainVesselOrient.at<uchar>(*it2, *it1)-1]++;
		it1++;
		it2++;
	}
	int maxOriV(0), maxOriP(0);
	for (int i=0; i<12; i++){
		if (orient[i] > maxOriV){
			maxOriV = orient[i];
			maxOriP = i;
		}
	}
	
	np =0;
	double intv = PI/12;
	double mapAng[12] = {-intv*6,-intv*5,-intv*4,-intv*3,-intv*2,-intv,0,intv,intv*2,intv*3,intv*4,intv*5};
	if (maxOriV!=0){
		int nb[4];
		nb[0] = (maxOriP+1+12)%12+1;
		nb[1] = (maxOriP+2+12)%12+1;
		nb[2] = (maxOriP-1+12)%12+1;
		nb[3] = (maxOriP-2+12)%12+1;
		double angCenter = mapAng[maxOriP];

		for (int j=startY; j<=endY; j++){
			for (int i=startX; i<=endX; i++){
				if (imlabel.at<int>(j,i) == (labelref+1)) continue;
				if (imMainVesselOrient.at<uchar>(j,i)==(maxOriP+1) || imMainVesselOrient.at<uchar>(j,i)==nb[0]
				|| imMainVesselOrient.at<uchar>(j,i)==nb[1] || imMainVesselOrient.at<uchar>(j,i)==nb[2]
				|| imMainVesselOrient.at<uchar>(j,i)==nb[3] ){
					// if the point is in the same direction
					double angNow = getAngle(center[0],center[1],i,j);
					double angDiff = abs(angNow-angCenter);
					if (angDiff>PI/2) angDiff = PI - angDiff;
					if (angDiff>PI/6) continue;

					if (imASF.at<uchar>(j,i)>0){
						inVessel += imASF.at<uchar>(j,i);
						np++;
					}
				}
			}
		}

		if (np==0) inVessel = 0;
		else inVessel /= np;
	}
	
	else inVessel = 0;

	if (meanVessel==0) ratioMeanVessel = 255;
	else ratioMeanVessel = meanRes/meanVessel;
	if (inVessel==0) ratioInVessel = 255;
	else ratioInVessel = meanRes/inVessel;
}


	

void MACand::envAnalyse(Mat imASF, Mat maCandiRaw, int* imInfo){
	/************************************************************************/
	/* Analyse the number of the connected components around the candidate
	1. In a small window, number of CC whos area between imInfo[3]^2/3~2*imInfo[3]^2
	2. In a large window, number of CC whos area between ns3 2*imInfo[3]^2 ~ inf
	3. Thresholds are maxRes and meanRes. (H and L) */
	/************************************************************************/
	int size[2] = {imASF.cols, imASF.rows};
	int W1 = imInfo[3];
	int W2 = imInfo[3]*2;
	int W3 = imInfo[3]*4;
	int C_area1 = imInfo[3]*imInfo[3]/3;
	int C_area2 = imInfo[3]*imInfo[3]*2;

	Mat imASFCut = Mat::zeros(W3*2+1,W3*2+1,CV_8U);
	Mat imCandiCut = Mat::zeros(W3*2+1,W3*2+1,CV_8U);
	Mat imtempCut1 = imASFCut.clone();
	Mat imtempCut2 = imASFCut.clone();
	Mat imtemp32 = Mat::zeros(W3*2+1,W3*2+1,CV_32S);

	int s,t;
	for (int n=-W3; n<=W3; ++n){
		for (int m=-W3; m<=W3; ++m){
			s = center[0]+m;
			t = center[1]+n;
			if (s<0 || s>=size[0] || t<0 || t>=size[1]) continue;
			imASFCut.at<uchar>(n+W3,m+W3) = imASF.at<uchar>(t,s);
			imCandiCut.at<uchar>(n+W3,m+W3) = maCandiRaw.at<uchar>(t,s);
		}
	}

	// Higher threshold 
	threshold(imASFCut,imtempCut1,maxRes-1,255,0);
	subtract(imtempCut1,imCandiCut,imtempCut2);
	Label(imtempCut2,imtemp32,6);
	int N = labelCount(imtemp32);
	int *ccAreaCount = new int[N];
	int *ccVisited = new int[N];
	memset(ccAreaCount,0,sizeof(int)*N);
	memset(ccVisited,0,sizeof(int)*N);
	
	for(int j=0; j<W3*2+1; j++){
		for (int i=0; i<W3*2+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			ccAreaCount[imtemp32.at<int>(j,i)-1]++;
		}
	}

	for(int j=W1*3; j<W1*5+1; j++){
		for (int i=W1*3; i<W1*5+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winS_thH_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winS_thH_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}

	memset(ccVisited,0,sizeof(int)*N);
	for(int j=W1*2; j<W1*6+1; j++){
		for (int i=W1*2; i<W1*6+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winL_thH_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winL_thH_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}
	delete[] ccAreaCount;
	delete[] ccVisited;
	
	// lower threshold
	threshold(imASFCut,imtempCut1,meanRes-1,255,0);
	subtract(imtempCut1,imCandiCut,imtempCut2);
	Label(imtempCut2,imtemp32,6);
	N = labelCount(imtemp32);
	ccAreaCount = new int[N];
	ccVisited = new int[N];
	memset(ccAreaCount,0,sizeof(int)*N);
	memset(ccVisited,0,sizeof(int)*N);

	for(int j=0; j<W3*2+1; j++){
		for (int i=0; i<W3*2+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			ccAreaCount[imtemp32.at<int>(j,i)-1]++;
		}
	}

	for(int j=W1*3; j<W1*5+1; j++){
		for (int i=W1*3; i<W1*5+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winS_thL_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winS_thL_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}

	memset(ccVisited,0,sizeof(int)*N);
	for(int j=W1*2; j<W1*6+1; j++){
		for (int i=W1*2; i<W1*6+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winL_thL_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winL_thL_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}

	delete[] ccAreaCount;
	delete[] ccVisited;

}



void writeFile(MACand *maCandList, int N, int* imInfo){
	/************************************************************************/
	/*	1. maxRes
		2. meanRes
		3. meanRes/meanVessel
		4. meanRes/inVessel
		5. min(W,H)/max(W,H) X 20
		6. area X 20
		7. length X 20
		9. circularity X 20
		9. medianRes[20]
		10~17. n_winS_thL_ccS,n_winS_thL_ccL,n_winL_thL_ccS,n_winL_thL_ccL,
		n_winS_thH_ccS,n_winS_thH_ccL,n_winL_thH_ccS,n_winL_thH_ccL
		18. isGT
		*/
	/************************************************************************/
	ofstream myfile;
	myfile.open("maFeatures.txt");

	float sizeNorm1=imInfo[3];
	float sizeNorm2=imInfo[3]*imInfo[3];

	for (int i=0; i<N; i++){
		myfile << maCandList[i].maxiPos[0]<<" "<< maCandList[i].maxiPos[1]<<" "<<maCandList[i].maxRes<<" "<<maCandList[i].meanRes<<" "
			<< maCandList[i].ratioMeanVessel<<" "<<maCandList[i].ratioInVessel<<" "<<maCandList[i].meanVessel<<" "<<maCandList[i].inVessel<<" ";
		//myfile<<"# ";
		for (int k=0; k<20; k++) myfile << maCandList[i].area2[k]/sizeNorm2<<" ";
		//myfile<<"# ";
		for (int k=0; k<20; k++) myfile << maCandList[i].WH[k]<<" ";
		//myfile<<"# ";
		for (int k=0; k<20; k++) myfile << maCandList[i].length[k]/sizeNorm1<<" ";		
		//myfile<<"# ";
		for (int k=0; k<20; k++) myfile << maCandList[i].circ[k]<<" ";
		//myfile<<"# ";
		for (int k=0; k<20; k++) myfile << maCandList[i].medianRes[k] / maCandList[i].maxRes <<" ";
			
		myfile << maCandList[i].n_winS_thL_ccS<<" "<<maCandList[i].n_winS_thL_ccL<<" "<<maCandList[i].n_winL_thL_ccS<<" "<<
			maCandList[i].n_winL_thL_ccL<<" "<<maCandList[i].n_winS_thH_ccS<<" "<<maCandList[i].n_winS_thH_ccL<<" "<<
			maCandList[i].n_winL_thH_ccS<<" "<<maCandList[i].n_winL_thH_ccL<<" "<<maCandList[i].isGT<<"\n";
	}
	myfile.close();
}

int geoLength(Mat imin, Mat imstate, list<int> *p, int h){
	int se(6);
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	int size[2] = {imin.cols, imin.rows};
	int p1[2],p2[2];
	int center[2] = {0,0};


	// 1. Get border pixels. put into Q
	list<int>::iterator it1; 
	list<int>::iterator it2;
	queue<int> Q[2];
	it1 = p[0].begin();
	it2 = p[1].begin();
	int f(0),px,py;

	while (it1 != p[0].end()){
		f = 0;
		for (int k=0; k<se; ++k){
			if (*it2%2==0){
				px = *it1 + se_even[k][0];
				py = *it2 + se_even[k][1];
			}
			else{
				px = *it1 + se_odd[k][0];
				py = *it2 + se_odd[k][1];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)<h){  // see if it's on the edge;
				f = 1;
				break;
			}
		}

		if (f==1){
			Q[0].push(*it1);
			Q[1].push(*it2);
			center[0] += *it1;
			center[1] += *it2;
		}

		imstate.at<uchar>(*it2,*it1) = 0;

		++it1;
		++it2;
	}
	center[0] /= Q[0].size();
	center[1] /= Q[0].size();
	/*cout<<center[0]<<" "<<center[1]<<endl;*/

	// 2. Get the pixel most far 
	int mx,my,len(0);
	float dist, maxDist(0);
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
			if (my%2==0){
				px = mx + se_even[k][0];
				py = my + se_even[k][1];
			}
			else{
				px = mx+ se_odd[k][0];
				py = my + se_odd[k][1];
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
	p1[0] = mx;
	p1[1] = my;

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
		p2[0] = mx;
		p2[1] = my;

		for (int k=0; k<se; ++k){
			if (my%2==0){
				px = mx + se_even[k][0];
				py = my + se_even[k][1];
			}
			else{
				px = mx + se_odd[k][0];
				py = my + se_odd[k][1];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)>0 && imstate.at<uchar>(py,px)==1){
				Q[0].push(px);	
				Q[1].push(py);
				imstate.at<uchar>(py,px) = 2;
			}
		}

		//	imstate[my][mx] = 2;
		Q[0].pop();
		Q[1].pop();
	}

	/*length = len;
	length2 = sqrt(pow(float(p1[0]-p2[0]),2) + pow(float(p1[1]-p2[1]),2));*/

	//circ = (float) (4*area)/(3.1415*length*length);

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;

	return len;
}

void ccAnalyseMA(Mat maCandiRaw, Mat imgreen, Mat imASF, Mat imROI, Mat imMainVesselGB, 
	Mat imMainVesselASF, Mat imMainVesselOrient, Mat imMainVesselSup, Mat imGT, int* imInfo){

	int C_area_max = imInfo[3]*imInfo[3];
		
	int imSize[2] = {imgreen.rows,imgreen.cols};

	Mat imMaxiG = imgreen.clone();
	Mat imtemp1 = imgreen.clone();
	Mat imtemp2 = imgreen.clone();
	Mat imMaASF = imgreen.clone();
	Mat imPerimeter = imgreen.clone();
	Mat imDTLabel = Mat::zeros(imSize[0], imSize[1], CV_32S);
	Mat imGTLabel = Mat::zeros(imSize[0], imSize[1], CV_32S);
	Mat imtemp32 = Mat::zeros(imSize[0], imSize[1], CV_32S);

	imCompare(maCandiRaw,0,1,imASF,0,imMaASF);
	imwrite("imMaCandiASF.png",imMaASF);

	Erode(maCandiRaw,imtemp1,6,1);
	subtract(maCandiRaw,imtemp1,imPerimeter);

	// maximas value in residu : maxRes
	Maxima(imASF,imtemp1,6);
	imCompare(imtemp1,0,1,imASF,0,imMaxiG);
	
	Label(maCandiRaw,imDTLabel,6);
	int N = labelCount(imDTLabel);
	cout<<N<<endl;

	MACand *maCandList = new MACand[N];

	for (int j=0; j<imgreen.rows; j++){
		for (int i=0; i<imgreen.cols; i++){
			if (imDTLabel.at<int>(j,i) == 0) continue;
			maCandList[imDTLabel.at<int>(j,i)-1].p[0].push_back(i);
			maCandList[imDTLabel.at<int>(j,i)-1].p[1].push_back(j);
			maCandList[imDTLabel.at<int>(j,i)-1].center[0] += i;
			maCandList[imDTLabel.at<int>(j,i)-1].center[1] += j;
			maCandList[imDTLabel.at<int>(j,i)-1].meanRes += imASF.at<uchar>(j,i);
			maCandList[imDTLabel.at<int>(j,i)-1].area ++;

			if (maCandList[imDTLabel.at<int>(j,i)-1].maxRes < imMaxiG.at<uchar>(j,i)){ 
				maCandList[imDTLabel.at<int>(j,i)-1].maxRes = imMaxiG.at<uchar>(j,i);
				maCandList[imDTLabel.at<int>(j,i)-1].maxiPos[0]=i;
				maCandList[imDTLabel.at<int>(j,i)-1].maxiPos[1]=j;
			}
		}
	}
	cout<<maCandList[149].maxiPos[0]<<" "<<maCandList[149].maxiPos[1]<<" "<<maCandList[149].maxRes<<" "<<maCandList[149].meanRes<<endl;

	for (int i=0; i<N; i++){
		maCandList[i].labelref = i;
		maCandList[i].meanRes /= maCandList[i].area;
		maCandList[i].center[0] /= maCandList[i].area;
		maCandList[i].center[1] /= maCandList[i].area;

		maCandList[i].vesselAnalyse(imASF,maCandiRaw,imMainVesselASF,imMainVesselOrient,imDTLabel,imInfo);
		maCandList[i].envAnalyse(imASF,maCandiRaw,imInfo);
	}



	// for candidate class
	/************************************************************************/
	/*	1. dilate GT mask, note how many MA markers (nMA) contained in one CC after dilation
		2. for each CC of GT mask after dilation, choose nMA most possible candidates*/
	/************************************************************************/
	fastDilate(imGT,imtemp2,6,imInfo[3]/3);
	Label(imtemp2,imGTLabel,6);
	Label(imGT,imtemp32,6);

	int NGTD = labelCount(imGTLabel);
	int NGT = labelCount(imtemp32);
	int *nMA = new int[NGTD];
	int *temp = new int[NGT];
	memset(temp,0,sizeof(int)*NGT);
	memset(nMA,0,sizeof(int)*NGTD);

	// 1.
	for (int j=0; j<imgreen.rows; j++){
		for (int i=0; i<imgreen.cols; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (temp[imtemp32.at<int>(j,i)-1]==0){
				temp[imtemp32.at<int>(j,i)-1]=1; 
				nMA[imGTLabel.at<int>(j,i)-1]++;
			}
		}
	}

	// 2.
	delete[] temp;
	temp = new int[N];
	memset(temp,0,sizeof(int)*N);
	list<int> *ccMean = new list<int>[NGTD];
	list<int> *ccNum = new list<int>[NGTD];

	for (int j=0; j<imgreen.rows; j++){
		for (int i=0; i<imgreen.cols; i++){
			if (imGTLabel.at<int>(j,i)==0 || imDTLabel.at<int>(j,i)==0) continue;
			if (temp[imDTLabel.at<int>(j,i)-1] > 0) continue;

			ccMean[imGTLabel.at<int>(j,i)-1].push_back(maCandList[imDTLabel.at<int>(j,i)-1].meanRes);
			ccNum[imGTLabel.at<int>(j,i)-1].push_back(imDTLabel.at<int>(j,i));
			temp[imDTLabel.at<int>(j,i)-1] = 1;
		}
	}

	for (int i=0; i<NGTD; i++){
		if (ccMean[i].size() == 0) continue;
		if (nMA[i] == ccMean[i].size()){
			maCandList[ccNum[i].front()-1].isGT=1;
		}
		else {
			for (int k=0; k<nMA[i]; k++){
				float maxMean(0),  tempMean;
				int maxNum(0),tempNum;
				for (int l=0; l<ccMean[i].size(); l++){
					tempMean = ccMean[i].front();
					tempNum = ccNum[i].front();
					ccMean[i].pop_front();
					ccNum[i].pop_front();
					if (maxMean<tempMean){
						maxMean = tempMean;
						maxNum = tempNum;
					}
					ccMean[i].push_back(tempMean);
					ccNum[i].push_back(tempNum);
				}
				maCandList[maxNum-1].isGT=1;

				// remove the max one
				for (int l=0; l<ccMean[i].size(); l++){
					if (ccNum[i].front() == maxNum){
						ccNum[i].pop_front();
						ccMean[i].pop_front();
					}
				}
			}
		}
	}

	delete[] nMA;
	delete[] temp;
	delete[] ccMean;
	delete[] ccNum;



	// Maxtree
	int *hist = histogram(imASF);
	int h=hist[256];
	int lenH = hist[257]+1;
	Mat imstate = Mat::zeros(imSize[0], imSize[1], CV_32S);
	subtract(imstate,2,imstate); // initiate to -2
	mxt maxTree(imASF,imstate);
	maxTree.flood_h(h,imASF, imstate, 6);

	layer **node = new layer* [lenH];  // node is a Class layer object
	for (int i=0; i<lenH; ++i){
		node[i] = new layer [maxTree.Nnodes[i]];
	}
	getRelations(maxTree,node,imASF,imstate,lenH,imInfo[3]*imInfo[3]);

	int iN,jN,mhN,miN,hh,ii,hh2,ii2;
	list<int>::iterator it1; 
	list<int>::iterator it2;
	for (int i=0; i<N; i++){
		iN = maCandList[i].maxiPos[0];
		jN = maCandList[i].maxiPos[1];
		mhN = maCandList[i].maxRes;
		miN = imstate.at<int>(jN,iN);

		int count = 0;
		hh = mhN; ii = miN;
		imtemp1.setTo(10);
		while(count<20 && node[hh][ii].area<C_area_max && hh>0){

			// WH
			maCandList[i].W = node[hh][ii].xmax - node[hh][ii].xmin + 1;
			maCandList[i].H = node[hh][ii].ymax - node[hh][ii].ymin + 1;
			if (maCandList[i].H > maCandList[i].W) maCandList[i].WH[count] = float(maCandList[i].W)/maCandList[i].H;
			else maCandList[i].WH[count] = float(maCandList[i].H)/maCandList[i].W;

			// area2
			maCandList[i].area2[count] = node[hh][ii].area;
			
			// length
			maCandList[i].length[count] = geoLength(imASF,imtemp1,node[hh][ii].p,hh);

			// circ
			maCandList[i].circ[count] = (float) (4*maCandList[i].area2[count])/
				(3.1415*maCandList[i].length[count]*maCandList[i].length[count]);

			// medianRes
			list<int> templ;
			templ.clear();
			it1 = node[hh][ii].p[0].begin();
			it2 = node[hh][ii].p[1].begin();
			while(it1!=node[hh][ii].p[0].end()){
				templ.push_back(imASF.at<uchar>(*it2,*it1));
				++it1;
				++it2;
			}
			templ.sort();
			templ.reverse();

			for (int w=0; w<maCandList[i].area2[count]; ++w){
				if ((maCandList[i].area2[count]/2==0 && w==0) || ((w+1)==maCandList[i].area2[count]/2)){
					maCandList[i].medianRes[count] = templ.front(); 
				}
				templ.pop_front();
			}


			hh2 = hh;
			ii2 = ii;
			hh = node[hh2][ii2].parent[0];
			ii = node[hh2][ii2].parent[1];

			count++;
			if (count>=20) break;

			for (int w = hh2; w>(hh+1); --w){
				maCandList[i].area2[count] = maCandList[i].area2[count-1];
				maCandList[i].WH[count] = maCandList[i].WH[count-1];
				maCandList[i].length[count] = maCandList[i].length[count-1];
				maCandList[i].circ[count] = maCandList[i].circ[count-1];
				maCandList[i].medianRes[count] = maCandList[i].medianRes[count-1];
				++count;
				if (count>=20) break ;
			}

			if (hh==-1){
				break;
			}
		}
	}


	/*for (int j=0; j<imgreen.rows; j++){
		for (int i=0; i<imgreen.cols; i++){
			if (imDTLabel.at<int>(j,i) == 0) continue;
			maCandList[imDTLabel.at<int>(j,i)-1].p[0].push_back(i);
			maCandList[imDTLabel.at<int>(j,i)-1].p[1].push_back(j);
			maCandList[imDTLabel.at<int>(j,i)-1].center[0] += i;
			maCandList[imDTLabel.at<int>(j,i)-1].center[1] += j;
			maCandList[imDTLabel.at<int>(j,i)-1].meanRes += imASF.at<uchar>(j,i);
			maCandList[imDTLabel.at<int>(j,i)-1].area ++;
			if (maCandList[imDTLabel.at<int>(j,i)-1].minX>=i) maCandList[imDTLabel.at<int>(j,i)-1].minX=i;
			if (maCandList[imDTLabel.at<int>(j,i)-1].minY>=j) maCandList[imDTLabel.at<int>(j,i)-1].minY=j;
			if (maCandList[imDTLabel.at<int>(j,i)-1].maxX<=i) maCandList[imDTLabel.at<int>(j,i)-1].maxX=i;
			if (maCandList[imDTLabel.at<int>(j,i)-1].maxY<=j) maCandList[imDTLabel.at<int>(j,i)-1].maxY=j;

			if (maCandList[imDTLabel.at<int>(j,i)-1].maxRes < imMaxiG.at<uchar>(j,i)) 
				maCandList[imDTLabel.at<int>(j,i)-1].maxRes = imMaxiG.at<uchar>(j,i);
		}
	}

	for (int i=0; i<N; i++){
		maCandList[i].labelref = i;
		maCandList[i].meanRes /= maCandList[i].area;
		maCandList[i].center[0] /= maCandList[i].area;
		maCandList[i].center[1] /= maCandList[i].area;
		maCandList[i].H = maCandList[i].maxY - maCandList[i].minY + 1;
		maCandList[i].W = maCandList[i].maxX - maCandList[i].minX + 1;

		if (maCandList[i].H > maCandList[i].W) maCandList[i].WH = float(maCandList[i].W)/maCandList[i].H;
		else maCandList[i].WH = float(maCandList[i].H)/maCandList[i].W;

		maCandList[i].vesselAnalyse(imASF,maCandiRaw,imMainVesselASF,imMainVesselOrient,imDTLabel,imInfo);
		maCandList[i].geoLength(imMaASF,imPerimeter,imtemp1);

		maCandList[i].envAnalyse(imASF,maCandiRaw,imInfo);
	}*/





	writeFile(maCandList,N,imInfo);
	delete[] maCandList;

}



void detectMA(Mat imgreenR, Mat imROIR, Mat imASFR,
	 int* imInfoR, Mat imGT){

	int imSizeR[2] = {imgreenR.rows,imgreenR.cols};
	Mat imtempR1 = imgreenR.clone();
	Mat imtempR2 = imgreenR.clone();
	Mat imtempR3 = imgreenR.clone();
	Mat imtempR4 = imgreenR.clone();
	Mat imtempR5 = imgreenR.clone();
	Mat imtempR6 = imgreenR.clone();
	Mat imGTD = imgreenR.clone();
	Mat imMainVesselGB = imgreenR.clone();
	Mat imMainVesselOrient = imgreenR.clone();
	Mat imMainVesselSup = imgreenR.clone();
	Mat imMainVesselASF = imgreenR.clone();
	Mat maCandiRaw = imgreenR.clone();
	Mat imtemp32int = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imLabelGT = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imLabelGTD = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imLabelDT = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imLabelArea1 = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imLabelArea2 = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imMACandi = imgreenR.clone();
	Mat imBN = imgreenR.clone();
	Mat imDarkPart = imgreenR.clone();
	Mat imHMCandi = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imROIRs = imgreenR.clone();
	Erode(imROIR,imROIRs,6,2);


	//========================================================
	// Get candidates
	/************************************************************************/
	/*	1. divided by 2, underbuild, get maximas
		2. remove large elements
		3. remove by mean value in each CC
		4. remove too long too small things.	*/
	/************************************************************************/
	int C_area_min = imInfoR[3]/2;
	if (C_area_min<4) C_area_min=4;

	divide(imASFR,2,imtempR1);
	imwrite("PP_div2.png",imtempR1);
	RecUnderBuild(imASFR,imtempR1,imtempR2,8);
	imwrite("PP_rec.png",imtempR2);
	Maxima(imtempR2,imtempR3,8);
	imwrite("PP_maxi.png",imtempR3);

	binAreaSelection(imtempR3,imtempR1,8,imInfoR[3]*imInfoR[3]);
	imwrite("PP_areaOP.png",imtempR1);
	imtempR1.copyTo(imtempR6);

	LabelByMean(imtempR1,imASFR,imtempR5,8);
	LabelByArea(imtempR1,imLabelArea2,8);
	Label(imtempR1,imLabelDT,8);
	int nDT = labelCount(imLabelDT);

	grayAreaSelection(imASFR,imtempR1,8,imInfoR[3]*imInfoR[3]);
	subtract(imASFR,imtempR1,imtempR2);
	threshold(imtempR2,imtempR3,0,255,0);
	LabelByArea(imtempR3,imLabelArea1,8);
	imwrite("PP_area1.png",imtempR3);



	// Get GT
	threshold(imGT,imtempR1,0,255,0);
	Label(imtempR1,imLabelGT,8);
	int nGT = labelCount(imLabelGT);
	binAreaSelection(imtempR1,imtempR2,8,imInfoR[3]*2);
	fastDilate(imtempR2,imtempR3,6,2);
	imSup(imtempR1,imtempR3,imGTD);
	Label(imGTD,imLabelGTD,8);
	int nGTD = labelCount(imLabelGTD);
	imwrite("PP_gtd.png",imGTD);

	int *numGTinGTD = new int[nGTD];
	int *GTVisited = new int[nGT];
	memset(numGTinGTD,0,sizeof(int)*nGTD);
	memset(GTVisited,0,sizeof(int)*nGT);
	
	for(int j=0;j<imgreenR.rows;++j){
		for(int i=0;i<imgreenR.cols;++i){
			if (imLabelGT.at<int>(j,i)==0) continue;
			if (GTVisited[imLabelGT.at<int>(j,i)-1]==0){
				GTVisited[imLabelGT.at<int>(j,i)-1] = 1;
				numGTinGTD[imLabelGTD.at<int>(j,i)-1] ++;
			}
		}
	}

	// get maxima
	Maxima(imASFR,imtempR3,8);
	imCompare(imtempR3,0,1,imASFR,0,imtempR2);
	imwrite("PP_max.png",imtempR2);



	list<int> *ccMean = new list<int>[nGTD];
	list<int> *ccNum = new list<int>[nGTD];
	list<int> *numInDT = new list<int>[nGTD];
	int *DTVisited = new int[nDT];
	memset(DTVisited,0,sizeof(int)*nDT);
	int **DTList = new int*[nDT];
	for (int i=0; i<nDT; i++){
		DTList[i] = new int[7];
		memset(DTList[i],0,sizeof(int)*7); // meanConst, maxConst, areaBot, areaMid, x, y
	}

	cout<<nGT<<" "<<nGTD<<" "<<nDT<<endl;

	for(int j=0;j<imgreenR.rows;++j){
		for(int i=0;i<imgreenR.cols;++i){
			if (imLabelDT.at<int>(j,i)==0 && imLabelGTD.at<int>(j,i)==0) continue;
			if (imLabelDT.at<int>(j,i)>0){
				DTList[imLabelDT.at<int>(j,i)-1][0] = imtempR5.at<uchar>(j,i);
				if (imtempR2.at<uchar>(j,i) > 0){
					if (DTList[imLabelDT.at<int>(j,i)-1][1] <= imtempR2.at<uchar>(j,i)){
						DTList[imLabelDT.at<int>(j,i)-1][1] = imtempR2.at<uchar>(j,i);
						DTList[imLabelDT.at<int>(j,i)-1][4] = i; 
						DTList[imLabelDT.at<int>(j,i)-1][5] = j;
					}
				}
				if (DTList[imLabelDT.at<int>(j,i)-1][2] < imLabelArea1.at<int>(j,i))
					DTList[imLabelDT.at<int>(j,i)-1][2] = imLabelArea1.at<int>(j,i);
				if (DTList[imLabelDT.at<int>(j,i)-1][3] < imLabelArea2.at<int>(j,i))
					DTList[imLabelDT.at<int>(j,i)-1][3] = imLabelArea2.at<int>(j,i);
			}
			if (DTVisited[imLabelDT.at<int>(j,i) - 1]>0) continue;
			if (imLabelDT.at<int>(j,i)>0 && imLabelGTD.at<int>(j,i)>0){
				ccMean[imLabelGTD.at<int>(j,i)-1].push_back(imtempR5.at<uchar>(j,i));
				ccNum[imLabelGTD.at<int>(j,i)-1].push_back(imLabelDT.at<int>(j,i)-1);
				DTVisited[imLabelDT.at<int>(j,i) - 1] = 1;
			}
		}
	}

	for (int i=0; i<nGTD; i++){
		if (ccMean[i].size()==0) continue;
		if (numGTinGTD[i] == ccMean[i].size()){
			while(!ccNum[i].empty()){
				numInDT[i].push_back(ccNum[i].front());
				DTList[ccNum[i].front()][6] = 1;
				ccNum[i].pop_front();
			}
		}
		else{
			for (int k=0; k<numGTinGTD[i]; k++){
				int tempMean(0), tempNum(0), maxMean(0), maxNum(0);
				for (int j=0; j<ccMean[i].size(); j++){
					tempMean = ccMean[i].front();
					tempNum = ccNum[i].front();
					ccMean[i].pop_front();
					ccNum[i].pop_front();
					if (maxMean<tempMean &&  DTList[tempNum][4]>0){
						maxMean=tempMean;
						maxNum=tempNum;
					}
					ccMean[i].push_back(tempMean);
					ccNum[i].push_back(tempNum);
				}
				
				numInDT[i].push_back(maxNum);
				DTList[maxNum][6] = 1;

				//remove the max one
				for (int j=0; j<ccMean[i].size(); j++){
					tempMean = ccMean[i].front();
					tempNum = ccNum[i].front();
					ccMean[i].pop_front();
					ccNum[i].pop_front();
					if (tempNum !=  maxNum){
						ccMean[i].push_back(tempMean);
						ccNum[i].push_back(tempNum);
					}
				}
			}
		}
	}
	

	ofstream myfile;
	myfile.open("maFeatures.txt");
	for (int i=0; i<nDT; i++){
		if (DTList[i][1]==0) continue;
		for (int j=0; j<7; j++){
			myfile << DTList[i][j] <<" ";
		}
		myfile << "\n";
	}
	
	myfile.close();

	for(int j=0;j<imgreenR.rows;++j){
		for(int i=0;i<imgreenR.cols;++i){
			if (imLabelDT.at<int>(j,i)==0) continue;
			if (DTList[imLabelDT.at<int>(j,i)-1][6]==1) imtempR6.at<uchar>(j,i) = 2;
		}
	}
	imwrite("Final.png",imtempR6);


	delete[] numGTinGTD;
	delete[] GTVisited;
	delete[] ccMean;
	delete[] ccNum;
	delete[] numInDT;
	delete[] DTVisited;
	for (int i=0; i<nDT; i++){
		delete[] DTList[i];
	}
	delete[] DTList;


	//binAreaSelection(imtempR3,imtempR1,6,imInfoR[3]*imInfoR[3]);
	//imwrite("PP_area_select.png",imtempR1);
	//LabelByMean(imtempR1,imASFR,imtempR2,6);
	//threshold(imtempR2,imtempR3,4,255,0);
	//imwrite("PP_contrast_select.png",imtempR3);
	//lengthOpening(imtempR3,imtempR1,imInfoR[3]*1.5,imInfoR[3]*imInfoR[3],0,2);
	//subtract(imtempR3,imtempR1,imtempR2);
	//imwrite("PP_length_select.png",imtempR2);
	//binAreaSelection(imtempR2,imtempR1,6,C_area_min);
	//subtract(imtempR2,imtempR1,imtempR3);
	//imwrite("PP_small_select.png",imtempR3);

	//RecUnderBuild(imtempR3,imOD,imtempR1,6);
	//subtract(imtempR3,imtempR1,maCandiRaw);
	//imwrite("imMaCandi.png",maCandiRaw);

	//ccAnalyseMA(maCandiRaw, imgreenR, imASFR, imROIR, imMainVesselGB, imMainVesselASF, imMainVesselOrient, imMainVesselSup, imGT, imInfoR);

	//Label(maCandiRaw,imtemp32int,6);

	
	//subtract(vessels[2],vessels[0],imtempR1);
	//imCompare(imROIRs,0,0,0,imtempR1,imtempR1);
	//// remove od region
	//RecUnderBuild(imtempR1,imOD,imtempR3,6);
	//subtract(imtempR1,imtempR3,imMACandi);
	//imwrite("imMACandi.png",imMACandi);
	//========================================================


	cout<<"Hello world"<<endl;
}



#endif
