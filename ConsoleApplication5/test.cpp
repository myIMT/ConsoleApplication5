#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "iostream"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
//#include "MyEdgeDetector.h"
#include <vector>
//#include <stdlib.h>
#include <array>

#include "wavelet2d.h"
//#include "cv.h"
//#include "highgui.h"
//#include "cxcore.h"

//#include <opencv2/legacy/compat.hpp>

using namespace cv;
using namespace std;
using std::vector;

//!{variables]
Mat src, src_gray,srcCopy;
Mat dst, detected_edges, angleHist, origHist,grayHist, pixelsBin,BinaryImg;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";
int ddepth = -1;// CV_32FC1;// CV_16S;
int scale = 1;
int delta = 0;
/// Generate grad_x and grad_y
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
Mat grad;
//------------------Angle Histogram Parameters--------
int binSize = 5;
int AngleLimit = 360;
/// Establish the number of bins
int histSize = AngleLimit / binSize;
//int histSize = 72;
/// Set the ranges ( for B,G,R) )
float rangeA[] = { 0, 360 };
const float* histRange = { rangeA };
Mat angle_hist;
bool uniform = true;
bool myAccumulate = false;
int channels[1];
int binID;

ofstream PixelsInBinFile;
std::array<std::vector<int>, 2> vvv{ {} };

int threshval = 60;
//string filename = "PixelsInBinFile";
//PixelsInBinFile.open("PixelsInBinFile.txt");
int mouseX, mouseY;
Point pt1, pt2;
int clickCounter = 0,lineCounter=0, pixelCounter=0;
std::vector<int> allObjPixelsCount;
//std::vector<double> buf;
vector<Point> points1;
vector<Point> points2;

Point anchor = Point(-1, -1);
//![variables]


void* maxval(vector<vector<double> > &arr, double &max) {
	max = 0;
	for (unsigned int i = 0; i < arr.size(); i++) {
		for (unsigned int j = 0; j < arr[0].size(); j++) {
			if (max <= arr[i][j]) {
				max = arr[i][j];
			}
		}
	}
	return 0;
}

void* maxval1(vector<double> &arr, double &max) {
	max = 0;
	for (unsigned int i = 0; i < arr.size(); i++) {
		if (max <= arr[i]) {
			max = arr[i];
		}

	}
	return 0;
}

int main()
{
#pragma region Blur_Sharpening
	//src = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	//Mat tempSrc = src;

	//cv::cvtColor(tempSrc, src_gray, cv::COLOR_BGR2GRAY);
	//Mat grayImg = src_gray;

	//Mat blurImg;
	//blur(grayImg, blurImg, Size(5, 5), Point(-1, -1));

	/////Sharpening filter----
	////Mat kernel[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};
	//Mat kernel = (Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
	////Mat	im = filter2D(tempSrc, -1, kernel);
	//Mat filImg;
	//filter2D(blurImg, filImg, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
	//imshow("Sharpening", filImg);

	//waitKey(0);
	//return 0;
#pragma endregion




	//Mat img = imread("20161215 02.33_368L.jpg");
	IplImage* img = cvLoadImage("snow.jpg");
	if (!img) {
		cout << " Can't read Image. Try Different Format." << endl;
		exit(1);
	}
	int height, width;
	height = img->height;
	width = img->width;
	int nc = img->nChannels;
	//   uchar* ptr2 =(uchar*) img->imageData;
	int pix_depth = img->depth;
	CvSize size;
	size.width = width;
	size.height = height;
	cout << "depth" << pix_depth << "Channels" << nc << endl;


	cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	cvShowImage("Original Image", img);
	cvWaitKey();
	cvDestroyWindow("Original Image");
	cvSaveImage("orig.bmp", img);


	int rows = (int)height;
	int cols = (int)width;
	Mat matimg(img);

	vector<vector<double> > vec1(rows, vector<double>(cols));


	int k = 1;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			unsigned char temp;
			temp = ((uchar*)matimg.data + i * matimg.step)[j  * matimg.elemSize() + k];
			vec1[i][j] = (double)temp;
		}

	}

	string nm = "db3";
	vector<double> l1, h1, l2, h2;
	filtcoef(nm, l1, h1, l2, h2);
	// unsigned int lf=l1.size();
	//  int rows_n =(int) (rows+ J*(lf-1));
	//  int cols_n =(int)  (cols + J * ( lf -1));

	// Finding 2D DWT Transform of the image using symetric extension algorithm
	// Extension is set to 3 (eg., int e = 3)

	vector<int> length;
	vector<double> output, flag;
	int J = 3;
	dwt_2d_sym(vec1, J, nm, output, flag, length);

	double max;
	vector<int> length2;
	// This algorithm computes DWT of image of any given size. Together with convolution and
	// subsampling operations it is clear that subsampled images are of different length than
	// dyadic length images. In order to compute the "effective" size of DWT we do additional
	// calculations.
	dwt_output_dim_sym(length, length2, J);
	// length2 is gives the integer vector that contains the size of subimages that will
	// combine to form the displayed output image. The last two entries of length2 gives the
	// size of DWT ( rows_n by cols_n)

	int siz = length2.size();
	int rows_n = length2[siz - 2];
	int cols_n = length2[siz - 1];

	vector<vector< double> > dwtdisp(rows_n, vector<double>(cols_n));
	dispDWT(output, dwtdisp, length, length2, J);

	// dispDWT returns the 2D object dwtdisp which will be displayed using OPENCV's image
	// handling functions

	vector<vector<double> >  dwt_output = dwtdisp;

	maxval(dwt_output, max);// max value is needed to take care of overflow which happens because
							// of convolution operations performed on unsigned 8 bit images

							//Displaying Scaled Image
							// Creating Image in OPENCV
	IplImage *cvImg; // image used for output
	CvSize imgSize; // size of output image

	imgSize.width = cols_n;
	imgSize.height = rows_n;

	cvImg = cvCreateImage(imgSize, 8, 1);
	// dwt_hold is created to hold the dwt output as further operations need to be
	// carried out on dwt_output in order to display scaled images.
	vector<vector<double> > dwt_hold(rows_n, vector<double>(cols_n));
	dwt_hold = dwt_output;
	// Setting coefficients of created image to the scaled DWT output values
	for (int i = 0; i < imgSize.height; i++) {
		for (int j = 0; j < imgSize.width; j++) {
			if (dwt_output[i][j] <= 0.0) {
				dwt_output[i][j] = 0.0;
			}
			if (i <= (length2[0]) && j <= (length2[1])) {
				((uchar*)(cvImg->imageData + cvImg->widthStep*i))[j] =
					(char)((dwt_output[i][j] / max) * 255.0);
			}
			else {
				((uchar*)(cvImg->imageData + cvImg->widthStep*i))[j] =
					(char)(dwt_output[i][j]);
			}
		}
	}

	cvNamedWindow("DWT Image", 1); // creation of a visualisation window
	cvShowImage("DWT Image", cvImg); // image visualisation
	cvWaitKey();
	cvDestroyWindow("DWT Image");
	cvSaveImage("dwt.bmp", cvImg);

	// Finding IDWT

	vector<vector<double> > idwt_output(rows, vector<double>(cols));

	idwt_2d_sym(output, flag, nm, idwt_output, length);



	//Displaying Reconstructed Image

	IplImage *dvImg;
	CvSize dvSize; // size of output image

	dvSize.width = idwt_output[0].size();
	dvSize.height = idwt_output.size();

	cout << idwt_output.size() << idwt_output[0].size() << endl;
	dvImg = cvCreateImage(dvSize, 8, 1);

	for (int i = 0; i < dvSize.height; i++)
		for (int j = 0; j < dvSize.width; j++)
			((uchar*)(dvImg->imageData + dvImg->widthStep*i))[j] =
			(char)(idwt_output[i][j]);

	cvNamedWindow("Reconstructed Image", 1); // creation of a visualisation window
	cvShowImage("Reconstructed Image", dvImg); // image visualisation
	cvWaitKey();
	cvDestroyWindow("Reconstructed Image");
	cvSaveImage("recon.bmp", dvImg);







	return 0;
}