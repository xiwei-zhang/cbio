#ifndef __ccAnalyse_h
#define __ccAnalyse_h

#include <iostream>
#include <fstream>

#define PI 3.1415 

class candi{
public:
	list<int> p[2];
	int center[2];
	int minTH, maxTH, meanTH, medianTH, minI, maxI, meanI,medianI;
	int area, area2, height, volume, volume2, pics, borderDepth, VAR, VAR2, UO, UO2;
	double saturation, saturation2;
	int perimeter;
	int distCenter, distMin;
	float nPic_h, nPic_l, area_h, area_l;
	int orientSimi;
	int isEXD;

	// assist variables
	int xmin,xmax,ymin,ymax;
	int length,length2; //geo length and 
	float circ1,circ2; // circ1: length/area    circ2: perimeter/area
	int p1[2];
	int p2[2];

	// functions
	candi();
	void getCenter(Mat imDist, Mat imCandidate, Mat imGreen, Mat imPerimeter, Mat imMaxi, Mat imGT, Mat imVAR, Mat imUO, Mat imSat);
	void getCenter2(Mat imDist, Mat imCandidate, Mat imGreen, Mat imPerimeter, Mat imMaxi, Mat imGT, Mat imVAR, Mat imUO, Mat imSat);
	void geoLength(Mat imin, Mat imPerimeter, Mat imstate);
	void bdpth_vd(Mat imBDpth, Mat imVesselD, Mat imPerimeter);
	void neighbor(Mat imGreen, Mat imMaxiG, Mat immark4, int *imInfo);
};

candi::candi(){
	center[0]=0; center[1]=0;
	area = 0; area2 = 0; volume = 0; volume2 = 0; perimeter = 0; pics = 0; length=0; length2=0;
	xmin = 99999; ymin = 99999; xmax = 0; ymax=0;
	minTH = 0; maxTH = 0; meanTH = 0; medianTH = 0;
	minI = 0; maxI = 0; meanI = 0; medianI = 0;
	nPic_l = 0; nPic_h = 0; area_h  = 0; area_l = 0;
	orientSimi = 0;
	isEXD = 0;
	VAR=0; UO=0; saturation=0; VAR2 = 0; UO2 = 0; saturation2 = 0;
}

void candi::getCenter(Mat imDist, Mat imCandidate, Mat imGreen, Mat imPerimeter, Mat imMaxi, Mat imGT,  Mat imVAR, Mat imUO, Mat imSat){
	list<int>::iterator it1; 
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	priority_queue<int> THQ,IQ;
	int maxV(0),v2(0);
	while (it1!=p[0].end()){
		if (imDist.at<uchar>(*it2,*it1)>maxV){
			maxV = imDist.at<uchar>(*it2,*it1);
			center[0] = *it1;
			center[1] = *it2;
		}
		if(*it1<xmin) xmin = *it1;
		if(*it1>xmax) xmax = *it1;
		if(*it2<ymin) ymin = *it2;
		if(*it2>ymax) ymax = *it2;
		volume += imCandidate.at<uchar>(*it2,*it1);
		v2 += imGreen.at<uchar>(*it2,*it1);
		VAR += imVAR.at<uchar>(*it2,*it1);
		UO += imUO.at<uchar>(*it2,*it1);
		saturation += imSat.at<double>(*it2,*it1);
		THQ.push(imCandidate.at<uchar>(*it2,*it1));
		IQ.push(imGreen.at<uchar>(*it2,*it1));
		perimeter+=imPerimeter.at<uchar>(*it2,*it1);
		pics += imMaxi.at<uchar>(*it2,*it1);
		area++;
		++it1;
		++it2;
	}
	UO/=area;
	saturation/=area;
	VAR/=area;
	maxTH = THQ.top();
	maxI = IQ.top();
	meanTH = volume/area;
	meanI = v2/area;
	int n(0);
	while(!THQ.empty()){
		if (n==area/2){
			medianTH = THQ.top();
			medianI = IQ.top();
		}
		minI = IQ.top();
		minTH = THQ.top();
		THQ.pop();
		IQ.pop();
		n++;
	}
	height = maxI - minI;

	if(imGT.at<uchar>(center[1],center[0])>0) isEXD = 1;
	else isEXD = 0;
}


void candi::getCenter2(Mat imDist, Mat imCandidate, Mat imGreen, Mat imPerimeter, Mat imMaxi, Mat imGT,  Mat imVAR, Mat imUO, Mat imSat){
	list<int>::iterator it1; 
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	int maxV(0),v2(0);
	while (it1!=p[0].end()){
		if (imCandidate.at<uchar>(*it2,*it1)==0) {
			++it1;
			++it2;
			continue;
		}
		volume2 += imCandidate.at<uchar>(*it2,*it1);
		VAR2 += imVAR.at<uchar>(*it2,*it1);
		UO2 += imUO.at<uchar>(*it2,*it1);
		saturation2 += imSat.at<double>(*it2,*it1);
		++area2;
		++it1;
		++it2;
	}
	if (area2>0){
		UO2/=area2;
		saturation2/=area2;
		VAR2/=area2;
	}
}



void candi::geoLength(Mat imin, Mat imPerimeter, Mat imstate){
	int se(6);
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	int size[2] = {imPerimeter.cols, imPerimeter.rows};

	// 1. Get border pixels
	list<int>::iterator it1; 
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	queue<int> Q[2];
	while (it1!=p[0].end()){		
		if (imPerimeter.at<uchar>(*it2,*it1)>0){
			Q[0].push(*it1);
			Q[1].push(*it2);
		}
		imstate.at<uchar>(*it2,*it1) = 0;
		++it1;
		++it2;
	}
	// 2. Get the pixel most far 
	int px,py,mx,my,len(0);
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

	length = len;
	length2 = sqrt(pow(float(p1[0]-p2[0]),2) + pow(float(p1[1]-p2[1]),2));

	circ1 = (float) (4*area)/(3.1415*length*length);
	circ2 = (float) perimeter*perimeter /(4*3.1415*area);

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}

void candi::bdpth_vd(Mat imBDpth, Mat imVesselD, Mat imPerimeter){
	list<int>::iterator it1; 
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	int dMin(255);
	while (it1!=p[0].end()){
		if (imPerimeter.at<uchar>(*it2,*it1)>0){
			if (imVesselD.at<uchar>(*it2,*it1)<dMin){
				dMin = imVesselD.at<uchar>(*it2,*it1);
			}
		}
		++it1;
		++it2;
	}
	distCenter = imVesselD.at<uchar>(center[1],center[0]);
	distMin = dMin;
	borderDepth = imBDpth.at<uchar>(center[1],center[0]);
}

void candi::neighbor(Mat imGreen, Mat imMaxiG, Mat immark4, int *imInfo){
	Mat imtemp1 = Mat::zeros(imGreen.rows, imGreen.cols, CV_8U);
	Mat imtemp2 = Mat::zeros(imGreen.rows, imGreen.cols, CV_8U);
	Mat imtemp3 = imGreen.clone();
	Mat imnPicH = imGreen.clone();
	Mat imnPicL = imGreen.clone();
	Mat	imareaH = imGreen.clone();
	Mat imareaL = imGreen.clone();
	Mat imVOS = imGreen.clone();

	// decide threshold
	// neighborRange = imInfo[2]
	int mm = meanI;
	int mi = minI;
	int ma = maxI;
	int mm_;
	if (maxTH>15){
		mm_ = ma - height/3;
		if (mm<mm_) mm = mm_;
	}

	// get ROI
	int xstart = xmin - imInfo[2];
	int xend = xmax + imInfo[2];
	int ystart = ymin - imInfo[2];
	int yend = ymax + imInfo[2];
	if (xstart<0) xstart = 0;
	if (ystart<0) ystart = 0;
	if (xend>=imGreen.cols) xend = imGreen.cols-1;
	if (yend>=imGreen.rows) yend = imGreen.rows-1;
	for (int i=xstart; i<=xend; i++){
		for (int j=ystart; j<=yend; j++){
			imtemp1.at<uchar>(j,i) = imGreen.at<uchar>(j,i);
		}
	}
	subtract(imtemp1,immark4,imtemp2);

	//nPic_h
	imCompare(imtemp1,mm,2,0,imMaxiG,imtemp3);
	subtract(imtemp3,immark4,imnPicH);
	
	//nPic_l
	imCompare(imtemp1,mi,2,0,imMaxiG,imtemp3);
	subtract(imtemp3,immark4,imnPicL);
	
	//area_h
	threshold(imtemp2,imareaH,(mm-1),255,0);
	
	//area_l
	threshold(imtemp2,imareaL,(mi-1),255,0);

	int vos[3] = {0,0,0}, maxNP,maxVO,maxCO;
	for (int i=xstart; i<=xend; i++){
		for (int j=ystart; j<=yend; j++){
			if (imnPicH.at<uchar>(j,i)>0) nPic_h++;
			if (imnPicL.at<uchar>(j,i)>0) nPic_l++;
			if (imareaH.at<uchar>(j,i)>0) area_h++;
			if (imareaL.at<uchar>(j,i)>0) area_l++;
		}
	}

	// 0326 new
	// normalize by neighbor area (large image can have a explosive neighbor area )
	float neighborArea = ((xend-xstart)*(yend-ystart) - area);
	nPic_l /= neighborArea;
	nPic_h /= neighborArea;
	area_h /= neighborArea;
	area_l /= neighborArea;

	// vessel orientation similarity
	//if (vos[0]>vos[1]){maxNP=vos[0]; maxVO=1;}
	//else {maxNP=vos[1]; maxVO=2;}
	//if (maxNP<vos[2]){maxNP=vos[2]; maxVO=3;}

	//double angle = atan2(-float(p1[1]-p2[1]), (p1[0]-p2[0]));
	//if (maxNP<imInfo[2]) orientSimi = 0;
	//else{
	//	if ((angle>0 && angle<PI/3) || (angle>=-PI && angle<-PI/3*2)) maxCO = 3;
	//	else if((angle>-PI/3 && angle<=0) || (angle>PI/3*2 && angle<=PI)) maxCO = 2;
	//	else maxCO = 1;
	//	if (maxVO==maxCO) orientSimi = 1;
	//}

}


void borderDepth(Mat imCandi, Mat imgreen, Mat imout, int *imInfo, int n){
	Mat imtemp1 = imCandi.clone();
	Mat imtemp2 = imCandi.clone();
	Mat imtemp3 = imCandi.clone();

	Dilate(imgreen,imtemp1,6,2);
	subtract(imtemp1,imgreen,imtemp1);

	imout.setTo(0);
	for (int i=0; i<n; i++){
		Minima(imtemp1,imtemp2,6);
		add(imtemp1,i+1,imtemp2);
		RecOverBuild(imtemp1,imtemp2,imtemp3,6);
		Minima(imtemp3,imtemp2,6);
		
		binAreaSelection(imtemp2,imtemp3,6,imInfo[3]*imInfo[3]);
		threshold(imtemp3,imtemp3,1,1,0);
		RecUnderBuild(imCandi,imtemp3,imtemp2,6);

		add(imtemp2,imout,imout);

	}
}

void writeFiles(candi *Exds, int N, int* imInfo, char* fileName){
	ofstream myfile;
	myfile.open(fileName);
	/************************************************************************/
	/* 
	centerX, centerY
	0~7: maxTH, meanTH, medianTH, minI, maxI, meanI, medianI, height
	8~14: area2, volume2, perimeter, circ1, circ2
	15~22: pics, borderDepth, VAR2, UO2, saturation2, 
	23~25: distCenter, distMin, orientSimi,
	26~29: nPic_h, nPic_l, area_h, area_l
	isEXD*/
	/************************************************************************/

	if(1){ // normalize by size
		float sizeNorm1=imInfo[2];
		float sizeNorm2=imInfo[2]*imInfo[2];

		for (int i=0; i<N; i++){
			/*if (1){
				Exds[i].area /= sizeNorm2;
				Exds[i].volume /= sizeNorm2;
				Exds[i].perimeter /= sizeNorm1;
				Exds[i].area_h /= sizeNorm2;
				Exds[i].area_l /= sizeNorm2;
			}*/
			myfile << Exds[i].center[0]<<" "<< Exds[i].center[1]<<" " 
				<< Exds[i].maxTH <<" "<< Exds[i].meanTH <<" "<< Exds[i].medianTH <<" "<< Exds[i].minI <<" "<< Exds[i].maxI <<" "<< Exds[i].meanI <<" "<< Exds[i].medianI <<" "<< Exds[i].height <<" "
				<< Exds[i].area/sizeNorm2 <<" "<< Exds[i].area2/sizeNorm2  <<" "<< Exds[i].volume/sizeNorm2 <<" "<<Exds[i].volume2/sizeNorm2 <<" "<< Exds[i].perimeter/sizeNorm2 <<" "<< Exds[i].circ1 <<" "<< Exds[i].circ2 <<" "
				<< Exds[i].pics <<" "<< Exds[i].borderDepth <<" "<< Exds[i].VAR <<" "<<Exds[i].VAR2<<" "<< Exds[i].UO <<" "<<Exds[i].UO2<<" "<< Exds[i].saturation <<" "<<Exds[i].saturation2<<" "
				<< Exds[i].distCenter/sizeNorm1 <<" "<< Exds[i].distMin/sizeNorm1 <<" "
				<< Exds[i].nPic_h <<" "<< Exds[i].nPic_l <<" "<< Exds[i].area_h <<" "<< Exds[i].area_l <<" "
				<< Exds[i].isEXD<<"\n";
		}
	}
	else{ // not normalized (Don't use this!!!)
		for (int i=0; i<N; i++){	
			myfile << Exds[i].center[0]<<" "<< Exds[i].center[1]<<" " 
				<< Exds[i].maxTH <<" "<< Exds[i].meanTH <<" "<< Exds[i].medianTH <<" "<< Exds[i].minI <<" "<< Exds[i].maxI <<" "<< Exds[i].meanI <<" "<< Exds[i].medianI <<" "<< Exds[i].height <<" "
				<< Exds[i].area <<" "<< Exds[i].volume <<" "<< Exds[i].perimeter <<" "<< Exds[i].circ1 <<" "<< Exds[i].circ2 <<" "
				<< Exds[i].pics <<" "<< Exds[i].borderDepth <<" "<< Exds[i].VAR <<" "<< Exds[i].UO <<" "<< Exds[i].saturation <<" "
				<< Exds[i].distCenter <<" "<< Exds[i].distMin <<" "<<Exds[i].orientSimi<<" "
				<< Exds[i].nPic_h <<" "<< Exds[i].nPic_l <<" "<< Exds[i].area_h <<" "<< Exds[i].area_l <<" "
				<< Exds[i].isEXD<<"\n";
		}
	}
	myfile.close();
}

void ccAnalyse(Mat imCandidate,  Mat imCandiSmall, Mat imGreen, Mat imINP,  Mat imROI, Mat imVAR, Mat imUO, Mat imSat, Mat imVessel, Mat imGT, char* fileName, char* imName, int t_area){

	////----------------------------------
	// Criteria
	int *imInfo = sizeEstimate(imROI);
	////----------------------------------


	Mat imtemp1 = imCandidate.clone();
	Mat imtemp2 = imCandidate.clone();
	Mat imtemp3 = imCandidate.clone();
	Mat immark1 = imCandidate.clone(); // all candidate grey level
	Mat immark2 = imCandidate.clone(); // all candidate 255 & 128
	Mat immark3 = imCandidate.clone(); // all candidate binary
	Mat immark4 = imCandidate.clone(); // main candidate binary
	Mat immark5 = imCandidate.clone(); // 2nd level TH
	Mat immark6 = imCandidate.clone(); // 2nd level binary
	Mat imtemp32_1 = Mat::zeros(imCandidate.rows, imCandidate.cols, CV_32S );
	Mat imtemp32_2 = Mat::zeros(imCandidate.rows, imCandidate.cols, CV_32S );
	Mat imDist = imCandidate.clone();
	Mat imPerimeter = imCandidate.clone();
	Mat imMaxi = imCandidate.clone();
	Mat imMaxiG = imCandidate.clone();
	Mat imBDpth = imCandidate.clone();
	Mat imVesselD = imCandidate.clone();
	Mat imVOrient = imCandidate.clone();
	Mat imASF = imCandidate.clone();
	Mat imGTAD = imCandidate.clone();
	Mat imSatNorm = Mat::zeros(imCandidate.rows, imCandidate.cols, CV_64F );

	// 0. Fusion and class main and small candidates
	int mm = meanValue(imCandidate,1);
	if (mm>5) mm=5;
	imSup(imCandidate,imCandiSmall,immark1);
	threshold(imCandidate,imtemp2,mm,255,0);
	RecUnderBuild(imCandiSmall,imtemp2,imtemp3,6);
	subtract(imCandiSmall,imtemp3,imtemp1);
	imCompare(imtemp1,2,1,128,imtemp2,imtemp2);
	binAreaSelection(imtemp2,imtemp1,6,t_area);
	subtract(imtemp2,imtemp1,immark2);
	imCompare(immark2,255,0,255,0,immark4);
	threshold(immark2,immark3,1,255,0);
	imCompare(immark2,1,4,0,immark1,immark1);
	Label(immark3,imtemp32_1,6);
	imwrite(imName,immark3);
	
	// 0.b Get second level of each candidate
	divide(immark1,2,imtemp1);
	RecUnderBuild(immark1,imtemp1,imtemp2,6);
	subtract(immark1,imtemp2,imtemp3);
	imCompare(imtemp3,1,2,0,immark1,immark5);
	threshold(immark5,immark6,0,255,0);
	imwrite("imCandi2.png",immark5);
	imwrite("imzz1.png",immark1);
	

	// get Ground truth 
	RecUnderBuild(immark3,imGT,imGTAD,6);

	int N = labelCount(imtemp32_1);
	cout<<"number of CC: "<<N<<endl;
	candi *CCs = new candi[N];

	//-----------------------------
	// 1. get pixels, get centers
	for (int j=0; j<imCandidate.rows; j++){
		for (int i=0; i<imCandidate.cols; i++){
			if (imtemp32_1.at<int>(j,i)==0) continue;
			CCs[imtemp32_1.at<int>(j,i)-1].p[0].push_back(i);
			CCs[imtemp32_1.at<int>(j,i)-1].p[1].push_back(j);
		}
	}
	
	// 2. distance
	Distance(immark3,imDist,6); 

	// 3. contour of each candidate
	Erode(immark3,imtemp2,6,1); 
	subtract(immark3,imtemp2,imtemp3);
	threshold(imtemp3,imPerimeter,1,1,0);

	// 4. vessel distance
	subtract(255,imVessel,imtemp1); 
	Distance(imtemp1,imVesselD,6);

	// 5. calculate border depth (time consuming..)
	borderDepth(immark3,imINP,imBDpth, imInfo,10); 
	
	// 6. get maximas (keep one pixel for one maxima)
	// Maxima(immark1,imMaxi,6);
	Label(immark6,imtemp32_2,6);
	int nMaxi = labelCount(imtemp32_2);
	imMaxi.setTo(0);
	if (nMaxi>0){
		int *listOfMaxi = new int[nMaxi];
		memset(listOfMaxi,0,sizeof(int)*nMaxi);
		for (int j=0; j<imCandidate.rows; j++){
			for (int i=0; i<imCandidate.cols; i++){
				if(imtemp32_2.at<int>(j,i)==0) continue;
				if (listOfMaxi[imtemp32_2.at<int>(j,i)-1]==0){
					listOfMaxi[imtemp32_2.at<int>(j,i)-1] = 1;
					imMaxi.at<uchar>(j,i) = 1;
				}
				else{
					imMaxi.at<uchar>(j,i) = 0;
				} 
			}
		}
		delete[] listOfMaxi;
	}
	imwrite("maxi.png",imMaxi);
	
	Maxima(imGreen,imMaxiG,6);
	Label(imMaxiG,imtemp32_2,6);
	nMaxi = labelCount(imtemp32_2);

	int *listOfMaxiG = new int[nMaxi];
	memset(listOfMaxiG,0,sizeof(int)*nMaxi);
	for (int j=0; j<imCandidate.rows; j++){
		for (int i=0; i<imCandidate.cols; i++){
			if(imMaxiG.at<uchar>(j,i)==0) continue;
			if (listOfMaxiG[imtemp32_2.at<int>(j,i)-1]==0){
				listOfMaxiG[imtemp32_2.at<int>(j,i)-1] = 1;
			}
			else{
				imMaxiG.at<uchar>(j,i) = 0;
			} 
		}
	}
	delete[] listOfMaxiG;

	// 7.vessel direction
	//Mat im = imread("imASF.png"); 
	//vector<Mat> planes2;
	//split(im, planes2);
	//planes2[1].copyTo(imASF);
	//imtemp1 = Mat::zeros(imtemp1.rows,imtemp1.cols,CV_8U);
	//vector<Mat> vesselProperty;
	//vesselProperty = vesselAnalyse(imVessel,imROI,imASF,imInfo);
	//vesselProperty[3].copyTo(imVOrient);


	// 8. saturation normalization
	imCompare(imROI,0,0,0,imSat,imSat);
	int satMean = meanValue(imSat,1);
	for (int j=0; j<imCandidate.rows; j++){
		for (int i=0; i<imCandidate.cols; i++){
			imSatNorm.at<double>(j,i) = double(imSat.at<uchar>(j,i))/satMean;
		}
	}

	// start extract characteristics for each candidate
	for (int i=0; i<N; i++){
		CCs[i].getCenter(imDist,immark1,imGreen,imPerimeter,imMaxi,imGTAD,imVAR,imUO,imSatNorm);
		CCs[i].getCenter2(imDist,immark5,imGreen,imPerimeter,imMaxi,imGTAD,imVAR,imUO,imSatNorm);
		CCs[i].geoLength(immark1,imPerimeter,imtemp1);
		CCs[i].bdpth_vd(imBDpth,imVesselD,imPerimeter);
		CCs[i].neighbor(imGreen,imMaxiG,immark4, imInfo);
		
	}
	
	

	writeFiles(CCs,N,imInfo,fileName);
	Label(immark3,imtemp32_2,6);



	// CCs[26].neighbor(imGreen,imMaxiG,immark4, imVOrient, imInfo);
	//
	//imtemp1 = Mat::zeros(imtemp1.rows,imtemp1.cols,CV_8U);
	//for (int i=0; i<N; i++){
	//	if (CCs[i].orientSimi==1){
	//		while(!CCs[i].p[0].empty()){
	//			imtemp1.at<uchar>(CCs[i].p[1].front(), CCs[i].p[0].front()) = 255;
	//			CCs[i].p[1].pop_front();
	//			CCs[i].p[0].pop_front();
	//		}
	//	}
	//}


	
	//int ii=167;//34;
	//cout<<CCs[ii].center[0]<<" "<<CCs[ii].center[1]<<" "<<CCs[ii].area<<" "<<CCs[ii].volume<<" "<<CCs[ii].perimeter<<" "<<CCs[ii].pics<<endl;
	//cout<<CCs[ii].length<<" "<<CCs[ii].length2<<" "<<CCs[ii].p1[0]<<" "<<CCs[ii].p1[1]<<" "<<CCs[ii].p2[0]<<" "<<CCs[ii].p2[1]<<endl;
	//cout<<CCs[ii].circ1<<" "<<CCs[ii].circ2<<endl;
	//cout<<CCs[ii].xmin<<" "<<CCs[ii].xmax<<" "<<CCs[ii].ymin<<" "<<CCs[ii].ymax<<endl;
	//cout<<CCs[ii].minTH<<" "<<CCs[ii].maxTH<<" "<<CCs[ii].meanTH<<" "<<CCs[ii].medianTH<<" "<<CCs[ii].minI<<" "
	//	<<CCs[ii].maxI<<" "<<CCs[ii].meanI<<" "<<CCs[ii].medianI<<endl;
	//cout<<CCs[ii].borderDepth<<" "<<CCs[ii].distCenter<<" "<<CCs[ii].distMin<<" "<<CCs[ii].orientSimi<<endl;
	//cout<<CCs[ii].nPic_h<<" "<<CCs[ii].nPic_l<<" "<<CCs[ii].area_h<<" "<<CCs[ii].area_l<<endl;
	//Label(immark3,imtemp32_1,6);
	//cout<<"Hello world"<<endl;
}

#endif
