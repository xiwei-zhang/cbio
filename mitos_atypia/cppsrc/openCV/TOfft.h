#ifndef __TOfft_h
#define __TOfft_h

#include <complex>
#include <math.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define PI 3.14159

void FFT(complex<double> *TD, complex<double> *FD, int r){
	long count;
	int bfsize,p;
	double angle;
	complex<double> *W,*X1,*X2,*X;

	count = 1 << r;

	W = new complex<double>[count/2];
	X1 = new complex<double>[count];
	X2 = new complex<double>[count];

	// cal pow coef
	for (int i=0; i<count/2; i++){
		angle = -i * PI * 2 / count;
		W[i] = complex<double> (cos(angle), sin(angle));
	}

	// real
	memcpy(X1, TD, sizeof(complex<double>) * count);

	// butterfly 
	for (int k=0; k<r;k++){
		for (int j=0; j<1<<k; j++){
			bfsize = 1<<(r-k);
			for (int i=0; i<bfsize/2; i++){
				p = j * bfsize;
				X2[i+p] = X1[i+p] + X1[i+p+bfsize/2];
				X2[i+p+bfsize/2] = (X1[i+p]-X1[i+p+bfsize/2]) * W[i*(1<<k)];
			}
		}
		X = X1;
		X1 = X2;
		X2 = X;
	}

	// order
	for (int j=0; j< count; j++){
		p = 0;
		for(int i=0; i<r; i++){
			if (j&(1<<i)){
				p+=1<<(r-i-1);
			}
		}
		FD[j]=X1[p];
	}

	delete W;
	delete X1;
	delete X2;

}

void IFFT(complex<double> *FD, complex<double> *TD, int r){
	long count;
	complex<double> *X;
	count = 1<<r;

	X = new complex<double>[count];
	memcpy(X,FD,sizeof(complex<double>)*count);

	//conjugate
	for(int i=0; i<count; i++){
		X[i] = complex<double> (X[i].real(), -X[i].imag());
	}

	FFT(X, TD, r);

	// conjugate in time
	for (int i=0; i<count; i++){
		TD[i] = complex<double> (TD[i].real()/count, -TD[i].imag()/count);
	}

	delete X;
}

void FFT2D(Mat imin){
	long lWidth = imin.cols;
	long lHeight = imin.rows;

	long w,h; //fft width and hight. should be pow of 2
	int wp,hp;

	w=1; h=1; wp=0; hp=0;
	// calculate the width and height of FFT trans
	while (w*2 <= lWidth){
		w *=2;
		wp ++;
	}
	while (h*2 <= lHeight){
		h *=2;
		hp ++;
	}

	// padding image
	Mat imPad = Mat::zeros(h,w,CV_8U);
	Mat imFFT = Mat::zeros(h,w,CV_32F);
	for (int j=0; j<imin.rows; j++)
		for (int i=0; i<imin.cols; i++)
			imPad.at<uchar>(j,i) = imin.at<uchar>(j,i);

	complex<double> *TD = new complex<double>[w*h];
	complex<double> *FD = new complex<double>[w*h];

	for (int j=0; j<h; j++){
		for (int i=0; i<w; i++){
			TD[i+j*w]= complex<double>(imPad.at<uchar>(j,i),0);
		}
	}

	// y direction FFT
	for(int j=0; j<h; j++) FFT(&TD[w*j], &FD[w*j], wp);
	// transform
	for(int j=0; j<h; j++)
		for(int i=0; i<w; i++)
			TD[j+h*i] = FD[i+w*j];
	// x direction FFT
	for (int i=0; i<w; i++) FFT(&TD[h*i], &FD[h*i], hp);

	// frequence
	//for (int j=0; j<h; j++){
	//	for (int i=0; i<w; i++){
	//		dTemp = sqrt(FD[i*h +j].real() * FD[i*h+j].real() + FD[i*h +j].imag()*FD[i*h +j].imag()); // /100;
	//		// if (dTemp>255) dTemp=255;
	//		imFFT.at<int>(j,i) = dTemp;
	//	}
	//}
	//imwrite("fft.png",imPad);


	// IFFT
	for (int i=0; i<w; i++) IFFT(&FD[h*i], &TD[h*i], hp);
	// transform
	for(int j=0; j<h; j++)
		for(int i=0; i<w; i++)
			FD[i+w*j] = TD[j+h*i];
	for(int j=0; j<h; j++) IFFT(&FD[w*j], &TD[w*j], wp);

	for (int j=0; j<h; j++){
		for (int i=0; i<w; i++){
			imPad.at<uchar>(j,i) = round(TD[i+j*w].real());
		}
	}


	delete[] TD;
	delete[] FD;
}

void fastMeanFilter(Mat imin, int size, Mat imout, int depth= 8){
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
	Mat imPad = Mat::zeros(h,w,CV_32F);
	Mat imKernalPad = Mat::zeros(h,w,CV_32F);
	int meanV = meanValue(imin,1);
	imPad.setTo(meanV);
	int xx0 = imin.cols/2, yy0 = imin.rows/2, x0_=w/2, y0_=h/2;
	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			if (depth==8) imPad.at<float>(j-yy0+y0_,i-xx0+x0_) = imin.at<uchar>(j,i);
			if (depth==33) imPad.at<float>(j-yy0+y0_,i-xx0+x0_) = imin.at<float>(j,i);
		}
	}
	int dwKernal = (size-1)/2;
	for (int j=-dwKernal; j<=dwKernal; j++){
		for (int i=-dwKernal; i<=dwKernal; i++){
			imKernalPad.at<float>(h/2+j,w/2+i) = 1.0f/(size*size);
		}
	}
	/////////////////////////////////////////////////



	// FFT/////////////////////////////////////////////////
	complex<double> *TD = new complex<double>[w*h];
	complex<double> *FD = new complex<double>[w*h];
	complex<double> *kTD = new complex<double>[w*h];
	complex<double> *kFD = new complex<double>[w*h];

	for (int j=0; j<h; j++){
		for (int i=0; i<w; i++){
			TD[i+j*w]= complex<double>(imPad.at<float>(j,i),0);
			kTD[i+j*w]= complex<double>(imKernalPad.at<float>(j,i),0);
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
			imPad.at<float>(j,i) = round(TD[(i<w/2?w/2+i:i-w/2)+(j<h/2?h/2+j:j-h/2)*w].real());
		}
	}
	
	for (int j=0; j<imin.rows; j++){
		for (int i=0; i<imin.cols; i++){
			if (depth==8) imout.at<uchar>(j,i) = imPad.at<float>(j-yy0+y0_,i-xx0+x0_);
			if (depth==33) imout.at<float>(j,i) = imPad.at<float>(j-yy0+y0_,i-xx0+x0_);
		}
	}
	/////////////////////////////////////////////////


	delete[] TD;
	delete[] FD;
	delete[] kTD;
	delete[] kFD;
}


void getVAR(Mat imin, Mat imout, int l){
	clock_t var1=clock();
	imout.setTo(0);
	int w = (l-1)/2,x,y,np;
	float meanv, sumv;
	for (int j=0; j<imin.rows; j++){
		//cout<<j<<endl;
		for (int i=0; i<imin.cols; i++){
			//cout<<j<<" "<<i<<endl;
			if (imin.at<uchar>(j,i)==0) continue;
			sumv = 0; np = 0;
			for (int m=-w; m<=w; m++){
				for (int n=-w; n<=w; n++){
					x = i+m;
					y = j+n;
					if (x<0 || x>=imin.cols || y<0 || y>=imin.rows) continue;
					sumv += (float)imin.at<uchar>(y,x);
					np++;
				}
			}
			meanv = sumv/np;
			sumv = 0;
			for (int m=-w; m<=w; m++){
				for (int n=-w; n<=w; n++){
					x = i+m;
					y = j+n;
					if (x<0 || x>=imin.cols || y<0 || y>=imin.rows) continue;
					sumv += ((float)imin.at<uchar>(y,x) - meanv)*((float)imin.at<uchar>(y,x) - meanv);
				}
			}
			sumv /= np;
			if (sumv>255) sumv=255;
			imout.at<uchar>(j,i)=round(sumv);
		}
	}

	clock_t var2=clock();
	cout<<"    VVVVVAR time: "<<double(diffclock(var2,var1))<<"ms"<<endl;

	//Mat imtempf1 = Mat::zeros(imin.rows, imin.cols, CV_32F);
	//Mat imtempf2 = Mat::zeros(imin.rows, imin.cols, CV_32F);
	//Mat imtempf3 = Mat::zeros(imin.rows, imin.cols, CV_32F);
	//Mat imtempf4 = Mat::zeros(imin.rows, imin.cols, CV_32F);
	//for (int i=0; i<imin.cols; i++){
	//	for (int j=0; j<imin.rows; j++){
	//		imtempf1.at<float>(j,i) = (float)imin.at<uchar>(j,i);
	//	}
	//}
	//multiply(imtempf1,imtempf1,imtempf2);
	//fastMeanFilter(imtempf1,l,imtempf3,33);
	//fastMeanFilter(imtempf2,l,imtempf4,33);
	//multiply(imtempf3,imtempf3,imtempf1);
	//subtract(imtempf4,imtempf1,imtempf2);
	//for (int i=0; i<imin.cols; i++){
	//	for (int j=0; j<imin.rows; j++){
	//		if(imtempf2.at<float>(j,i)<0) imtempf2.at<float>(j,i)=0;
	//		if(imtempf2.at<float>(j,i)>255) imtempf2.at<float>(j,i)=255;
	//		imout.at<uchar>(j,i) = (int)imtempf2.at<float>(j,i);
	//	}
	//}
	//cout<<"VAR: "<<(float)imtempf3.at<float>(500,500)<<" "<<(float)imtempf4.at<float>(500,500)<<endl;
}

#endif