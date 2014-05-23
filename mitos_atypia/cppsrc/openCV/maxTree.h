#ifndef __maxTree_h
#define __maxTree_h

#include <queue>
#include <list>
#include "basicOperations.h"
#include "TOMorph.h"



/*  Construction of a Max-tree from an image
    Code by xiwei in 2012
Method from article:
@article{salembier1998antiextensive,
title={Antiextensive connected operators for image and sequence processing},
author={Salembier, P. and Oliveras, A. and Garrido, L.},
journal={Image Processing, IEEE Transactions on},
volume={7},
number={4},
pages={555--570},
year={1998},
publisher={IEEE}
}
*/
class mxt{
public:
	queue<int> *hqueueX;
	queue<int> *hqueueY;
	queue<int> Qhi[2];
	queue<int> Qmj[2];
	int *Nnodes;
	int *hist;
	bool *nodeAtLevel;

	mxt(Mat imin, Mat imstate);  // constructor
	void DeMT();  // Deconstruction
	int flood_h(int h, Mat imin, Mat imstate, int se);  // build a maxtree
};

// Construct for mxt (max tree) class
mxt::mxt(Mat imin, Mat imstate){
	int size[2] = {imin.cols, imin.rows };
	hist = histogram(imin);
	int lenH = hist[257]+1;
 	
	hqueueX = new queue<int>[lenH];  // hist 256,257 --> min and max histogram
	hqueueY = new queue<int>[lenH]; 
	
	Nnodes = new int[lenH];  // number of node for each layer
	for (int i=0; i<lenH; ++i)
		Nnodes[i]=0;
	nodeAtLevel = new bool[lenH];
	for (int i=0; i<lenH; ++i)
		nodeAtLevel[i] = false;

	// initialize
	// put the lowest gray level pixel into queue 
	int f(0);
	for (int j=0;j<size[1];++j){
		for (int i=0;i<size[0];++i){
			if (imin.at<uchar>(j,i)==  hist[256]){
				hqueueX[hist[256]].push(i); // hqueueX[0].push(0)
				hqueueY[hist[256]].push(j); // hqueueY[0].push(0)
				imstate.at<int>(j,i) = -1;
				nodeAtLevel[hist[256]] = true;
				f = 1;
				break;
			}
		}
		if (f==1)
			break;
	}
}

// Deconstruction 
void mxt::DeMT(){
	delete[] hqueueX;
	delete[] hqueueY;
	delete[] Nnodes;
	delete[] nodeAtLevel;
	delete[] hist;
}

// Flood at h-level (Maxtree is built here by recursion) 
int mxt::flood_h(int h, Mat imin,  Mat imstate, int se){
	int size[2] = {imin.cols, imin.rows };
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	int px,py,x,y,vp,vq,m,s,t;
	while (! hqueueX[h].empty()){
		px = hqueueX[h].front();
		py = hqueueY[h].front();
		hqueueX[h].pop();
		hqueueY[h].pop();
		imstate.at<int>(py,px) = Nnodes[h];
		vp = h;
		for (int k=0; k<se; ++k){
			if (py%2==0){
				x = px + se_even[k][0];
				y = py + se_even[k][1];
			}
			else{
				x = px + se_odd[k][0];
				y = py + se_odd[k][1];
			}
			 
			if(x<0 || x>=size[0]|| y<0||y>=size[1]) continue;
			if(imstate.at<int>(y,x)==-2){
				vq = int(imin.at<uchar>(y,x));
				hqueueX[vq].push(x);
				hqueueY[vq].push(y);
				imstate.at<int>(y,x) = -1;
				nodeAtLevel[vq] = true;
				if (vq>vp){
					m = vq;
					while(m!=h){
						m = flood_h(m,imin, imstate, se);
					}
				}
			}
		}
	}
	++Nnodes[h];

	m = h-1;
	while (m>=0 && nodeAtLevel[m] == false){
		--m;
	}

	if (m>=0){
		s = Nnodes[h]-1;
		t = Nnodes[m];
		Qhi[0].push(h);
		Qhi[1].push(s);
		Qmj[0].push(m);
		Qmj[1].push(t);
	}
	else{
		Qhi[0].push(h);
		Qhi[1].push(0);
		Qmj[0].push(-1);
		Qmj[1].push(-1);
	}

	
	nodeAtLevel[h]=false;
	return m;
}


//###########################################
// Class layer (analysis each node of maxtree)
class layer{
public:
	// Management de maxtree
	int h,i; // h: height  i: number of label
	int parent[2]; // -1,-1 mean root
	list<int> children[2];  // all children
	list<int> p[2];  // all pixels within this layer
	list<int> maxChild[2];  // all maxima of this layer
		
	// Self-properties
	int W;
	int H;
	int xmin,xmax,ymin,ymax;
	int area;
	int center[2];
	int length;// geolength;
	float length2; // cartesian, distance between two end points
	int length3; // orthogonal to length2 ,width for bounding box rotated
	float length4; // mean width along geolength direction
	float widthVar;
	int maxWidth;
	list<float> meanWidthL; // interrupted orthogonal width. inter points see interP
	list<int> maxWidthL;
	list<float> widthVarL;

	int volume;
	int perimeter;
		
	int NpixelsAboveC; // given a criterier C, for each cc, number of pixels above this threshold
	int vIPic;   // Value of maxima in I (orig) image. for MA, the minima in orig 
	int vTHPic;  // Value of maxima in Top Hat image.
	int IC;  // given a number of pixels, for each cc, from the maximum to the bottom, when arrive this number of pixels, take the greylevel
	float Imean; 
	int Imedian;
	int THC;
	float THmean;
	int THmedian;

	float circ1;  // (pi*D^2)/(4*S) where D is diameter and S is area
	float circ2;  // C^2/(4*pi*S) where C is perimeter and S is area
		
	float orient; // Not used...

	float var1; // local variance green channel
	float var2; // local variance on tophat image
	float var3; // between 12 direction

	// Environment	
	int inner_en_area[8];
	int outter_en_area[8];
	int count[4];
		
	// Others		
	int mark;  // mark:0 background ,1 vessel1 2,vessel2 3,ma 4,noise 5,big
	int markFilter;
	int p1[2];
	int p2[2];
	int p3[2];
	int p4[2];
	list<int> interP[2];

	// viterbi
	double emP[5];
	double V[5],maxV;
	int maxP,maxFP;
	int maContrast;
	int hmContrast;
		
	// Functions
	layer();
	void getPixels(layer **node, list<int>* pp);
	void getBasicInfo(list<int>* pp);
	void printInfo();
	void setValue(Mat imin, int v,int ini);
	void setValueP(Mat imin, list<int>* pp, int v,int ini);
	void geoLength(Mat imin, Mat imstate,layer **node, int se, list<int> *pp, int interval, int critere,  Mat imout, bool lenOrtho);
	void lengthOrtho(list<int> *pp, int *startP, int *endP, int critere, Mat imout);
};

layer::layer(){
	W = -1;
	H = -1;
	xmin = 99999;
	xmax = -1;
	ymin = 99999;
	ymax = -1;
	area = 0;
	center[0] = -1;
	center[1] = -1;
	orient = -1;
	mark = 0;
	vTHPic = -1;
	NpixelsAboveC = 0;
	volume = 0;
	for(int i=0; i<8; ++i){
		inner_en_area[i] = 0;
		outter_en_area[i] = 0;
	}
	count[0] = -1;
	count[1] = -1;
	markFilter = -1;
	maContrast = 0;
	hmContrast = 0;
	length = 0;
	circ1 = 0.0f;
	circ2 = 0.0f;
}

void layer::printInfo(){
	cout<<"parents: "<< parent[0]<<" "<<parent[1]<<endl;
	cout<<"W: "<<W<<endl;
	cout<<"H: "<<H<<endl;
	cout<<"area: "<<area<<endl;
	cout<<"center: "<<center[0]<<" "<<center[1]<<endl;
	cout<<"orient: "<<orient;
	cout<<endl;
}

void layer::getBasicInfo(list<int>* pp){
	list<int>::iterator it1; 
	list<int>::iterator it2;
	it1 = pp[0].begin();
	it2 = pp[1].begin();
	int xmin(9999),ymin(9999),xmax(-1),ymax(-1);
	while(it1!=pp[0].end()){
		if (*it1>=xmax) xmax = *it1;
		if (*it1<=xmin) xmin = *it1;
		if (*it2>=ymax) ymax = *it2;
		if (*it2<=ymin) ymin = *it2;
		it1++;
		it2++;
	}
	W = xmax - xmin;
	H = ymax - ymin;
	area = (int)pp[0].size();
}

void layer::getPixels( layer **node, list<int>* pp){
	list<int>::iterator it1; 
	list<int>::iterator it2; 
	it1 = p[0].begin();
	it2 = p[1].begin();
	while(it1!=p[0].end()){
		pp[0].push_back(*it1);
		pp[1].push_back(*it2);
		it1++;
		it2++;
	}
	it1 = children[0].begin();
	it2 = children[1].begin();
	while (it1 != children[0].end()){
		node[*it1][*it2].getPixels(node, pp);
		it1++;
		it2++;
	}
}

void layer::setValue(Mat imin, int v,int ini=-1){  // set image value
	list<int>::iterator it1; 
	list<int>::iterator it2; 
	it1 = p[0].begin();
	it2 = p[1].begin();
	while(it1!= p[0].end()){
		if (ini!=-1){
			if (imin.at<uchar>(*it2,*it1)==ini){
				imin.at<uchar>(*it2,*it1) = v;
				//if (*it1==16 && *it2==12) cout<<v<<endl;
			}
		}
		else
			imin.at<uchar>(*it2,*it1) = v;
		++it1;
		++it2;
	}
}

void layer::setValueP(Mat imin, list<int>* pp, int v, int ini=-1){
	list<int>::iterator it1; 
	list<int>::iterator it2; 
	it1 = pp[0].begin();
	it2 = pp[1].begin();
	while(it1!= pp[0].end()){
		if (ini!=-1){
			if (imin.at<uchar>(*it2,*it1)==ini){
				imin.at<uchar>(*it2,*it1) = v;
				//if (v==3 && *it1==496 && *it2==592) {cout<< "gGGGGGGGGGGGGGGGGGGGGGOOOOOOOOOOTTTTTTTT"<<endl;}
			}
		}
		else
			imin.at<uchar>(*it2,*it1) = v;
		++it1;
		++it2;
	}
}

// get geodesic length
void layer::geoLength(Mat imin, Mat imstate, layer **node, int se, list<int> *pp, int interval, int critere, Mat imout, bool lenOrtho){
	/*
	For steps:
	1. Get all border pixels
	2. Get the most further pixel (to the geometric center)
	3. First propagation from the most further pixel, get the last pixel(another most further, to avoid concave case)
	4. Second propagation from the new most further pixel, get the geo-length
	*/
	int size[2] = {imin.cols,imin.rows};
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	int inter = interval;

	////int croix[4][2] = {{0,-1} ,{-1,0}, {1,0}, {0,1}}; // Cross SE
	////int diag[4][2] = {{-1,-1} ,{1,-1}, {-1,1}, {1,1}}; // Cross SE
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
				px = *itx + se_even[k][0];
				py = *ity + se_even[k][1];
			}
			else{
				px = *itx + se_odd[k][0];
				py = *ity + se_odd[k][1];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)<h){  // see if it's on the edge;
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
	perimeter = (int)Q[0].size();
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
		/*cout<<" "<<my<<endl;*/
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
			if (imin.at<uchar>(py,px)>=h && imstate.at<uchar>(py,px)==0){  // see if it's on the edge;
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
	temppp[0].push_back(mx);
	temppp[1].push_back(my);
	Q[0].push(-1); // -1 is a mark point
	Q[1].push(-1);
	imstate.at<uchar>(my,mx) = 2;
	p1[0] = mx;
	p1[1] = my;
	interP[0].push_back(mx);
	interP[1].push_back(my);
	p3[0] = mx;
	p3[1] = my;
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

			if(len>=inter){
				interP[0].push_back(p4[0]);
				interP[1].push_back(p4[1]);
				//cout<<"INT: "<<interval<<" "<<p3[0]<<" "<<p3[1]<<" "<<p4[0]<<" "<<p4[1]<<"  "<<p1[0]<<" "<<p1[1]<<endl;
				lengthOrtho(temppp,p3,p4,critere,imout);
				temppp[0].clear();
				temppp[1].clear();
				
				inter += interval;
				p3[0] = mx;
				p3[1] = my;
			}
		}
		p2[0] = mx;
		p2[1] = my;

		f = 0;
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
			if (imin.at<uchar>(py,px)>=h && imstate.at<uchar>(py,px)==1){
				Q[0].push(px);	
				Q[1].push(py);
				temppp[0].push_back(px);
				temppp[1].push_back(py);
				imstate.at<uchar>(py,px) = 2;
				f = 1;
				p4[0] = px;
				p4[1] = py;
			}
		}

	//	imstate[my][mx] = 2;
		Q[0].pop();
		Q[1].pop();
	}
	// cout<<"INT: "<<interval<<" "<<p3[0]<<" "<<p3[1]<<" "<<p4[0]<<" "<<p4[1]<<"  "<<p1[0]<<" "<<p1[1]<<endl;
	if (lenOrtho)
		lengthOrtho(temppp,p3,p4,critere,imout);
	interP[0].push_back(p2[0]);
	interP[1].push_back(p2[1]);

	length = len;
	length2 = sqrt(pow(float(p1[0]-p2[0]),2) + pow(float(p1[1]-p2[1]),2));

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;

}

void layer::lengthOrtho(list<int> *pp, int *startP, int *endP,int critere, Mat imout){
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
	itx = pp[0].begin();
	ity = pp[1].begin();				
	while(itx!=pp[0].end()){
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
	length3 = round(maxVy - minVy);

	//if (startP[0]==60){
	//	Mat imtemp = Mat::zeros(200,200,CV_8U);
	//	itx = pp[0].begin();
	//	ity = pp[1].begin();
	//	while(itx!=pp[0].end()){
	//		// cout<<*itx<<" "<<*ity<<endl;
	//		imtemp.at<uchar>(*ity,*itx) = 255;
	//		itx++;
	//		ity++;
	//	}
	//	imwrite("temp.png",imtemp);
	//}

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
		meanWidthL.push_back(0);
		length4 = 0;
	}
	else {
		meanWidthL.push_back(length4);
		length4 = w/wd.size();
	}
	// VAR
	wd.sort();
	meanNorm = length4/wd.back();
	it = wd.begin();
	w = 0;
	while(it != wd.end()){
		w += pow(((*it)/wd.back() - meanNorm),2);
		it++;
	}
	widthVar = w/(wd.size());
	maxWidth = wd.back();
	widthVarL.push_back(widthVar);
	maxWidthL.push_back(maxWidth);

	//////////////// classification //////////
	//if (length4<=critere && maxWidth<=2*critere){
	//	setValueP(imout,pp,1,0);
	//	mark = 1;
	//}
	//else{
	//	setValueP(imout,pp,3);
	//	mark = 3;
	//}
}


// end of class layer
//###########################################


void getRelations(mxt maxTree, layer **node, Mat imin, Mat imstate, int lenH, int C_area_max){
	int hh,ii,fh,fi;
	int size[2] = {imin.cols,imin.rows};
	// For every node in maxtree, get it's parent nodes and children nodes
	while (! maxTree.Qhi[0].empty()){  
		hh = maxTree.Qhi[0].front(); // Qhi's father is Qmj
		ii = maxTree.Qhi[1].front();
		fh = maxTree.Qmj[0].front();
		fi = maxTree.Qmj[1].front();
		maxTree.Qhi[0].pop();
		maxTree.Qhi[1].pop();
		maxTree.Qmj[0].pop();
		maxTree.Qmj[1].pop();
		node[hh][ii].parent[0] = fh;
		node[hh][ii].parent[1] = fi;
		node[hh][ii].h = hh;
		node[hh][ii].i = ii;
		if (fh==-1 || fi == -1) continue;
		node[fh][fi].children[0].push_back(hh);
		node[fh][fi].children[1].push_back(ii);
	}

	for (int j=0; j<size[1]; ++j){ // get all pixels' position of each node
		for(int i=0; i<size[0]; ++i){
			hh = (int)imin.at<uchar>(j,i);
			ii = (int)imstate.at<int>(j,i);
			node[hh][ii].p[0].push_back(i);
			node[hh][ii].p[1].push_back(j);
			if (i<node[hh][ii].xmin) node[hh][ii].xmin = i;
			if (i>node[hh][ii].xmax) node[hh][ii].xmax = i;
			if (j<node[hh][ii].ymin) node[hh][ii].ymin = j;
			if (j>node[hh][ii].ymax) node[hh][ii].ymax = j;
			// if (hh==129 && ii ==172) cout<<i<<" "<<j<<endl;
			// if (i==1018 && j==312) cout<<hh<<" "<<ii<<endl;
		}
	}

	if (C_area_max!=0){
		list<int>::iterator it1;
		list<int>::iterator it2;
		list<int>::iterator it3;
		list<int>::iterator it4;
		for (int i=lenH-1; i>0; --i){
			for (int j=0; j<maxTree.Nnodes[i]; ++j){
				if (!node[i][j].children[0].empty()){
					it1 = node[i][j].children[0].begin();
					it2 = node[i][j].children[1].begin();
					while (it1 != node[i][j].children[0].end()){
						if (node[*it1][*it2].area>C_area_max){
							node[i][j].area = C_area_max+1;
							break;
						}
						it3 = node[*it1][*it2].p[0].begin();
						it4 = node[*it1][*it2].p[1].begin();
						while(it3 != node[*it1][*it2].p[0].end()){
							node[i][j].p[0].push_back(*it3);
							node[i][j].p[1].push_back(*it4);
							if (*it3<node[i][j].xmin) node[i][j].xmin = *it3;
							if (*it3>node[i][j].xmax) node[i][j].xmax = *it3;
							if (*it4<node[i][j].ymin) node[i][j].ymin = *it4;
							if (*it4>node[i][j].ymax) node[i][j].ymax = *it4;
							it3++;
							it4++;
						}
						it1++;
						it2++;
					}
				}
				if (node[i][j].area == 0)
					node[i][j].area = (int)node[i][j].p[0].size();
			}
		}
	}
}

void areaSelection( layer **node, Mat imin, Mat imstate, Mat imout, int C_area){
	int size[2] = {imin.cols,imin.rows};
	imin.copyTo(imout);
	
	int fh,fi,hh,ii,hh_,ii_,ch;

	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			hh = imin.at<uchar>(j,i);
			ii = imstate.at<int>(j,i);
			ch = hh;
			if (node[hh][ii].children[0].empty()){
				while (node[hh][ii].area <= C_area && node[hh][ii].area!=0){
					hh_ = hh; ii_=ii;
					fh = node[hh][ii].parent[0];
					fi = node[hh][ii].parent[1];
					hh = fh;
					ii = fi;
					if (hh==-1){
						hh=0;
						break;
					}
				}
				if(hh<ch)
					node[hh_][ii_].setValue(imout,hh);
			}
		}
	}
}


void lengthSelection( layer **node, Mat imin, Mat imstate, Mat imout, int C_len, int max_area, int C_circ, int op){
	// op: 1-keep elongated structure ; 2-keep round things
	int size[2] = {imin.cols,imin.rows};
	imin.copyTo(imout);
	Mat imtemp = Mat::zeros(imin.rows, imin.cols, CV_8U);
	Mat imvisited = Mat::zeros(imin.rows, imin.cols, CV_8U);

	int fh,fi,hh,ii,hh_,ii_,ch,count(0);

	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			if (imvisited.at<uchar>(j,i)!=0 || imin.at<uchar>(j,i)==0) continue;
			hh = imin.at<uchar>(j,i);
			ii = imstate.at<int>(j,i);
			ch = hh;
			if (node[hh][ii].children[0].empty()){
				node[hh][ii].setValue(imvisited,1);
				while (node[hh][ii].area <= max_area){
					if (node[hh][ii].length == 0 && node[hh][ii].area<=max_area){
						node[hh][ii].geoLength(imin,imtemp,node,6,node[hh][ii].p,999,999,imout,false);
					}
					if (node[hh][ii].circ1 == 0 && node[hh][ii].area<=max_area &&  C_circ>0)
						node[hh][ii].circ1 = 3.1415*node[hh][ii].length*node[hh][ii].length/(4*node[hh][ii].area);
					
					if (C_circ==0){
						if (node[hh][ii].length>C_len ) break;
					}
					else{
						if (op==1){
							if (node[hh][ii].circ1>C_circ && node[hh][ii].length>C_len) break;
						}
						else if (op==2){
							if (node[hh][ii].circ1<C_circ && node[hh][ii].length>C_len) break;
						}
					}
					hh_ = hh; ii_=ii;
					fh = node[hh][ii].parent[0];
					fi = node[hh][ii].parent[1];
					hh = fh;
					ii = fi;
					if (hh==-1 || hh==0){
						hh=0;
						break;
					}
				}
				if(hh<ch){
					node[hh_][ii_].setValue(imout,hh);
				}
			}
		}
	}
}


//
//
//void envAnalyse(mxt maxTree, layer **node,int lenH, int *size, int* imInfo, int **imDark, int **imOrig, int **imtemp, int **imstate, int **imGT, int mean_global){
//	/************************************************************************/
//	/* Analyse the number of the connected components around the candidate
//	1. node[ph][pi].count[0] = ns2;  In a small window, number of CC whos area between imInfo[3]^2/2~2*imInfo[3]^2
//	2. node[ph][pi].count[1] = ns3;  In a small window, number of CC whos area between ns3 2*imInfo[3]^2 ~ inf*/
//	/************************************************************************/
//	int C_areaMax = (imInfo[3]+5)*(imInfo[3]+5);
//	int W1 = imInfo[3];
//	int W2 = imInfo[3]*2;
//	int CS = imInfo[3]*imInfo[3];
//
//	list<int> temp[2];
//	int s,t,ph,pi,ph2,pi2,pht,pit,centerCoor[2],xt,yt,s_,t_,xt_,yt_,n_,m_;
//	int ns1,ns2,ns3; //ns1: 0~imInfo[3]^2/2; ns2 imInfo[3]^2/2~2*imInfo[3]^2; ns3 2*imInfo[3]^2 ~ inf
//	list<int>::iterator it1;
//	list<int>::iterator it2;
//
//	for (int i=lenH-1; i>=mean_global; --i){
//		for (int j=0; j<maxTree.Nnodes[i]; ++j){
//			if(node[i][j].children[0].empty()){
//				if (node[i][j].area >= C_areaMax) continue;
//				centerCoor[0] = node[i][j].center[0];
//				centerCoor[1] = node[i][j].center[1];
//				ph = i;
//				pi = j;
//				if(imGT[centerCoor[1]][centerCoor[0]] == 0) continue;
//				
//				while (node[ph][pi].area<C_areaMax){
//					// Initialize a temp windows
//					ns1 = 0; ns2 = 0; ns3 = 0; //counter
//					int **tempWin = new int*[4*W1+1];
//					for (int w=0; w<2*W1+1; ++w){
//						tempWin[w] = new int[4*W1+1];
//					}
//					for (int n=-W1; n<=W1; ++n){
//						for (int m=-W1; m<=W1; ++m){
//							s = centerCoor[0]+m;
//							t = centerCoor[1]+n;
//							if (s<0 || s>=size[0] || t<0 || t>=size[1]){
//								tempWin[n+W1][m+W1] = 1;
//								continue;
//							}
//							if (imDark[t][s] < ph){
//								tempWin[n+W1][m+W1] = 1;
//								continue;
//							}
//							tempWin[n+W1][m+W1] = 0;
//						}
//					}
//					it2 = node[ph][pi].p[1].begin();
//					for (it1 = node[ph][pi].p[0].begin(); it1!=node[ph][pi].p[0].end();it1++){
//						xt_ = *it1 - centerCoor[0] + W1;
//						yt_ = *it2 - centerCoor[1] + W1;
//						if (xt_<0 || xt_>2*W1|| yt_<0 || yt_>2*W1) {it2++;continue;}
//						tempWin[yt_][xt_] = 1;
//						it2++;
//					}
//					
//					// Start propagate (spatial)
//					temp[0].clear();
//					for (int n=-W1; n<=W1; ++n){
//						for (int m=-W1; m<=W1; ++m){
//							s = centerCoor[0]+m;
//							t = centerCoor[1]+n;
//							xt = m+W1;
//							yt = n+W1;
//							if (s<0 || s>=size[0] || t<0 || t>=size[1] || xt<0 ||xt>2*W1 || yt<0 || yt>2*W1){
//								continue;
//							}
//							if (tempWin[yt][xt]==1){
//								continue;
//							}
//
//							ph2 = imDark[t][s];
//							pi2 = imstate[t][s];
//							while (ph2>=ph){ // find the lowest gray level in the window
//								pht = ph2;
//								pit = pi2;
//								ph2 = node[pht][pit].parent[0];
//								pi2 = node[pht][pit].parent[1];
//							}
//
//							// update counters
//							if (node[pht][pit].area<CS/2) ns1++;
//							else if (node[pht][pit].area<2*CS) ns2++;
//							else ns3++;
//
//							temp[0].push_back(m);
//							temp[1].push_back(n);
//							tempWin[yt][xt] = 1;
//							while (!temp[0].empty()){
//								m_ = temp[0].front();
//								n_ = temp[1].front();
//								temp[0].pop_front();
//								temp[1].pop_front();
//
//								for (int b=-1; b<=1; ++b){
//									for (int a=-1; a<=1; ++a){
//										s_ = centerCoor[0]+m_+a;
//										t_ = centerCoor[1]+n_+b;
//										xt_ = m_+W1 +a;
//										yt_ = n_+W1 +b;
//										if (s_<0 || s_>=size[0] || t_<0 || t_>=size[1] || xt_<0 ||xt_>2*W1 || yt_<0 || yt_>2*W1 ){
//											continue;
//										}
//										if( tempWin[yt_][xt_] == 0 ){
//											temp[0].push_back(m_+a);
//											temp[1].push_back(n_+b);
//											tempWin[yt_][xt_] = 1;
//										}
//									}
//								}
//							}					
//						}
//					}
//
//					node[ph][pi].count[0] = ns2;
//					node[ph][pi].count[1] = ns3;
//
//					ph2 = ph;
//					pi2 = pi;
//					ph = node[ph2][pi2].parent[0];
//					pi = node[ph2][pi2].parent[1];
//					delete[] tempWin;
//					if (ph==-1){
//						break;
//					}
//				}
//			}
//		}
//	}
//}
//
//void envAnalyseL(mxt maxTree, layer **node,int lenH, int *size, int* imInfo, int **imDark, int **imOrig, int **imtemp, int **imstate, int **imGT, int mean_global){
//	/************************************************************************/
//	/* Analyse the number of the connected components around the candidate
//	1. node[ph][pi].count[0] = ns2;  In a large window, number of CC whos area between imInfo[3]^2/2~2*imInfo[3]^2
//	2. node[ph][pi].count[1] = ns3;  In a large window, number of CC whos area between ns3 2*imInfo[3]^2 ~ inf*/
//	/************************************************************************/
//	int C_areaMax = (imInfo[3]+5)*(imInfo[3]+5);
//	int W1 = imInfo[3];
//	int W2 = imInfo[3]*2;
//	int CS = imInfo[3]*imInfo[3];
//
//	SE mySE = SquareSE(5);
//	list<int> temp[2];
//	int s,t,ph,pi,ph2,pi2,pht,pit,centerCoor[2],xt,yt,s_,t_,xt_,yt_,n_,m_;
//	int ns1,ns2,ns3; //ns1: 0~imInfo[3]^2/2; ns2 imInfo[3]^2/2~2*imInfo[3]^2; ns3 2*imInfo[3]^2 ~ inf
//	int nCandi(0);
//	int **imtemp2 = creatImage<int>(size,0);
//	int **imEro = MorphoErode(imOrig,mySE,size);
//	list<int>::iterator it1; 
//	list<int>::iterator it2;
//
//	for (int i=lenH-1; i>=mean_global; --i){
//		for (int j=0; j<maxTree.Nnodes[i]; ++j){
//			if(node[i][j].children[0].empty()){ // && node[i][j].parent[0]>=0){
//				if (node[i][j].area >= C_areaMax) continue;
//				centerCoor[0] = node[i][j].center[0];
//				centerCoor[1] = node[i][j].center[1];
//				ph = i;
//				pi = j;
//
//				if(imGT[centerCoor[1]][centerCoor[0]] == 0) continue;
//				nCandi++;
//				// Start propagation (gray level). test each layer and do the selection
//				while (node[ph][pi].area<C_areaMax){
//					
//					// Initialize a temp windows
//					ns1 = 0; ns2 = 0; ns3 = 0; //counter
//					int **tempWin = new int*[4*W1+1];
//					for (int w=0; w<4*W1+1; ++w){
//						tempWin[w] = new int[4*W1+1];
//					}
//					for (int n=-W2; n<=W2; ++n){
//						for (int m=-W2; m<=W2; ++m){
//							s = centerCoor[0]+m;
//							t = centerCoor[1]+n;
//							if (s<0 || s>=size[0] || t<0 || t>=size[1]){
//								tempWin[n+W2][m+W2] = 1;
//								continue;
//							}
//							if (imDark[t][s] < ph){
//								tempWin[n+W2][m+W2] = 1;
//								continue;
//							}
//							tempWin[n+W2][m+W2] = 0;
//						}
//					}
//					it2 = node[ph][pi].p[1].begin(); 
//					for (it1 = node[ph][pi].p[0].begin(); it1!=node[ph][pi].p[0].end();it1++){
//						xt_ = *it1 - centerCoor[0] + W2;
//						yt_ = *it2 - centerCoor[1] + W2;
//						if (xt_<0 || xt_>2*W2|| yt_<0 || yt_>2*W2) {it2++;continue;}
//						tempWin[yt_][xt_] = 1;
//						it2++;
//					}
//
//					// Start propagate (spatial)
//					temp[0].clear();
//					for (int n=-W2; n<=W2; ++n){
//						for (int m=-W2; m<=W2; ++m){
//							s = centerCoor[0]+m;
//							t = centerCoor[1]+n;
//							xt = m+W2;
//							yt = n+W2;
//							//if (ph ==20 && i==34 && j==93 && tempWin[yt][xt]==0) cout<<"  "<<s<<" "<<t<<" "<<xt<<" "<<yt<<" "<<W2<<" "<<tempWin[yt][xt]<<endl;
//							if (s<0 || s>=size[0] || t<0 || t>=size[1] || xt<0 ||xt>2*W2 || yt<0 || yt>2*W2){
//								continue;
//							}
//							if (tempWin[yt][xt]==1){
//								continue;
//							}
//
//							ph2 = imDark[t][s];
//							pi2 = imstate[t][s];
//							while (ph2>=ph){
//								pht = ph2;
//								pit = pi2;
//								ph2 = node[pht][pit].parent[0];
//								pi2 = node[pht][pit].parent[1];
//							}
//
//							// update counters
//							if (node[pht][pit].area<CS/2) ns1++;
//							else if (node[pht][pit].area<2*CS) ns2++;
//							else ns3++;
//
//							temp[0].push_back(m);
//							temp[1].push_back(n);
//							tempWin[yt][xt] = 1;
//							while (!temp[0].empty()){
//								m_ = temp[0].front();
//								n_ = temp[1].front();
//								temp[0].pop_front();
//								temp[1].pop_front();
//
//								for (int b=-1; b<=1; ++b){
//									for (int a=-1; a<=1; ++a){
//										s_ = centerCoor[0]+m_+a;
//										t_ = centerCoor[1]+n_+b;
//										xt_ = m_+W2 +a;
//										yt_ = n_+W2 +b;
//										if (s_<0 || s_>=size[0] || t_<0 || t_>=size[1] || xt_<0 ||xt_>2*W2 || yt_<0 || yt_>2*W2 ){
//											continue;
//										}
//										if( tempWin[yt_][xt_] == 0 ){
//											temp[0].push_back(m_+a);
//											temp[1].push_back(n_+b);
//											tempWin[yt_][xt_] = 1;
//										}
//									}
//								}
//							}					
//						}
//					}
//
//					node[ph][pi].count[2] = ns2;
//					node[ph][pi].count[3] = ns3;
//
//					if (node[ph][pi].vTHPic < i)  node[ph][pi].vTHPic = i;
//
//					ph2 = ph;
//					pi2 = pi;
//					ph = node[ph2][pi2].parent[0];
//					pi = node[ph2][pi2].parent[1];
//					delete[] tempWin;
//					// ++count;
//					if (ph==-1){
//						break;
//					}
//				}
//			}
//		}
//	}
//	delete[] imtemp2;
//}

//float estImageQuality(int **imOrig, SE mySE, int* size){
//	/**************************************
//	Calculate the image quality, by integrate the gradient image
//	***************************************/
//	int nbP(0);
//	float average(0);
//	SE mySE2 = SquareSE(11); //11
//	int** imDil = MorphoDilate<int>(imOrig,mySE,size);
//	int** imEro = MorphoErode<int>(imOrig,mySE,size);
//	int** imEro2 = MorphoErode<int>(imOrig,mySE2,size);
//	for (int j=0; j<size[1]; ++j){
//		for (int i=0; i<size[0]; ++i){
//			if (imEro2[j][i]!=0){
//				average += (imDil[j][i] - imEro[j][i]);
//				++nbP;
//			}
//		}
//	}
//	average /= nbP;
//
//	delete[] imDil;
//	delete[] imEro;
//	delete[] imEro2;
//
//	return average;
//}

//void localVAR(mxt maxTree, layer **node,int lenH, int *size, int* imInfo, int **imDark, int **imOrig, int **imGT, int mean_global){
//	/**************************************
//	Calculate 3 type of Variances for each candidate
//	var1: local variance within a moving window on green image
//	var2: like var1 on top-hat image
//	var3: 1.calculate mean within a elongated structring element, which is rotating
//			in 12 direction.   2. variance for those 12 mean value.
//	***************************************/
//	int C_areaMax = (imInfo[3]+5)*(imInfo[3]+5);
//	int W2 = imInfo[3]*2;
//
//	SE mySE = SquareSE(5);
//	list<int> temp[2];
//	int s,t,ph,pi,ph2,pi2,centerCoor[2],x_,y_;
//	int nCandi(0);
//	int **imtemp2 = creatImage<int>(size,0);
//	int **imEro = MorphoErode(imOrig,mySE,size);
//	list<int>::iterator it1; 
//	list<int>::iterator it2;
//	list<float>::iterator it3;
//
//	// Initialization for var3
//	int l = imInfo[3]*3/2*2+1; //length of lines in 12 direction
//	int N = 12;
//	float *val = new float[N];
//	float *v = new float[l];
//	float angle;
//	int **x = new int *[N];
//	int **y = new int *[N];
//	for (int i=0; i<N; ++i){
//		x[i] = new int[l];
//		y[i] = new int[l];
//	}
//	float *xy = new float[l];
//	for (int i=0; i<l; ++i){xy[i] = i-l/2;}
//	for (int i=0; i<N; ++i){
//		angle = 3.1415/N*i;
//		for (int j=0; j<l; j++){
//			x[i][j] = int(round(xy[j]*cos(angle)));
//			y[i][j] = int(round(xy[j]*sin(angle)));
//		}
//	}
//
//	// Start calculate
//	for (int i=lenH-1; i>=mean_global; --i){
//		for (int j=0; j<maxTree.Nnodes[i]; ++j){
//			if(node[i][j].children[0].empty()){ 
//				if (node[i][j].area >= C_areaMax) continue;
//				float avg(0),var3(0),ma(0);
//
//				centerCoor[0] = node[i][j].center[0];
//				centerCoor[1] = node[i][j].center[1];
//				ph = i;
//				pi = j;
//
//				if(imGT[centerCoor[1]][centerCoor[0]] == 0) continue;
//				nCandi++;
//
//				// Var 3 (12 direction)
//				for (int m=0; m<N; ++m){
//					for (int n=0; n<l; ++n){
//						x_ = x[m][n] + centerCoor[0];
//						y_ = y[m][n] + centerCoor[1];
//						if (x_<0 || x_>=size[0] ||y_<0 || y_>=size[1]) continue;
//						v[n] = imDark[y_][x_];
//					}
//
//					int f1(0),f2(0); // To prevent mistaking another vessel
//					for (int k=0; k<l/2; ++k){
//						if (f1==1 && v[l/2+1+k]>5) v[l/2+1+k] = v[l/2+k];
//						if (f2==1 && v[l/2-1-k]>5) v[l/2-1-k] = v[l/2-k];
//						if (v[l/2+1+k] == 0)	f1 = 1;
//						if (v[l/2-1-k] == 0)	f2 = 1;
//					}
//
//					for (int k=0; k<l; ++k) {avg+=v[k];}
//					val[m] = avg/l;
//					avg = 0;
//				}
//				
//				for (int m=0; m<N; ++m){
//					if (val[m]>=ma) ma=val[m];
//				}
//				for (int m=0; m<N; ++m){
//					val[m]/=ma;
//					avg += val[m];
//				}
//				avg /= N;
//				for (int m=0; m<N; ++m){
//					var3 += pow((val[m]-avg),2);
//				}
//				node[i][j].var3 = var3/N;
//				//Finish for var3
//
//
//				while (node[ph][pi].area<C_areaMax){ // start propagation 
//					// Get THmedian
//					temp[0].clear();
//					it1 = node[ph][pi].p[0].begin();
//					it2 = node[ph][pi].p[1].begin();
//					while(it1!=node[ph][pi].p[0].end()){
//						temp[0].push_back(imDark[*it2][*it1]);
//						++it1;
//						++it2;
//					}
//					temp[0].sort();
//					temp[0].reverse();
//
//					for (int w=0; w<node[ph][pi].area; ++w){
//						if ((node[ph][pi].area/2==0 && w==0) || ((w+1)==node[ph][pi].area/2)){
//							node[ph][pi].THmedian = temp[0].front(); 
//						}
//						temp[0].pop_front();
//					}
//
//					if (node[ph][pi].vTHPic < i)  node[ph][pi].vTHPic = i;
//
//					ph2 = ph;
//					pi2 = pi;
//					ph = node[ph2][pi2].parent[0];
//					pi = node[ph2][pi2].parent[1];
//					if (ph==-1){
//						break;
//					}
//				}
//
//				// Local VAR 1,2
//				float average1(0), average2(0), var1(0), var2(2);
//				int nbP(0);
//				it1 = node[ph2][pi2].p[0].begin();
//				it2 = node[ph2][pi2].p[1].begin();
//				while(it1!=node[ph2][pi2].p[0].end()){
//					imtemp2[*it2][*it1] = nCandi;
//					++it1;
//					++it2;
//				}
//				for (int n=-W2; n<=W2; ++n){
//					for (int m=-W2; m<=W2; ++m){
//						s = node[ph2][pi2].center[0]+m;
//						t = node[ph2][pi2].center[1]+n;
//						if (s<0 || s>=size[0] || t<0 || t>=size[1]) continue;
//						if (imtemp2[t][s]==nCandi || imEro[t][s]==0) continue;
//						average1 += imOrig[t][s];
//						average2 += imDark[t][s];
//						nbP++;
//					}
//				}
//				average1 /= nbP;
//				average2 /= nbP;
//				for (int n=-W2; n<=W2; ++n){
//					for (int m=-W2; m<=W2; ++m){
//						s = node[ph2][pi2].center[0]+m;
//						t = node[ph2][pi2].center[1]+n;
//						if (s<0 || s>=size[0] || t<0 || t>=size[1]) continue;
//						if (imtemp2[t][s]==nCandi || imEro[t][s]==0) continue;
//						var1 += pow((imOrig[t][s]-average1),2);
//						var2 += pow((imDark[t][s]-average2),2);
//					}
//				}
//				var1 /= nbP;
//				var2 /= nbP;
//				node[i][j].var1 = var1;
//				node[i][j].var2 = var2;
//			}
//		}
//	}
//	delete[] imtemp2;
//	delete[] x;
//	delete[] y;
//	delete[] xy;
//	delete val;
//	delete v;
//}




#endif