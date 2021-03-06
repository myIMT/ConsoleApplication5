#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
//#include "MyEdgeDetector.h"
#include <vector>
#include <stdlib.h>
#include <array>

//#include <opencv2/legacy/compat.hpp>

using namespace cv;
using namespace std;
using std::vector;

//![variables]
Mat src, src_gray;
Mat dst, detected_edges, angleHist, origHist,grayHist, pixelsBin;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";
int ddepth = CV_32FC1;// CV_16S;
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
//string filename = "PixelsInBinFile";
//PixelsInBinFile.open("PixelsInBinFile.txt");
int mouseX, mouseY;
Point pt1, pt2;
int clickCounter = 0;

//LineIterator it;
//LineIterator it2;
//vector<Vec3b> buf;
//![variables]

void MyLine(Mat img, Point start, Point end)
{
	
	if (clickCounter==2)
	{
		clickCounter = 0;
	}

	cout << "Drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;

	int thickness = 2;
	int lineType = 8;
	line(img,
		Point(pt1.x,pt1.y),
		Point(pt2.x, pt2.y),
		Scalar(255, 0, 0),
		thickness,
		lineType);
	imshow("GrayScale Image", img);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		system("cls");
		clickCounter++;
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		if (clickCounter==1)
		{
			cout << "clickCounter= " << clickCounter << "\n";
			pt1.x = x;
			pt1.y = y;
			cout << "Left button of the mouse is clicked - position (" << pt1.x << ", " << pt1.y << ")" << endl;
		}
		else if (clickCounter==2)
		{
			cout << "clickCounter= " << clickCounter << "\n";
			pt2.x = x;
			pt2.y = y;
			cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;

			MyLine(src_gray, pt1, pt2);

			// grabs pixels along the line (pt1, pt2)
			// from 8-bit 3-channel image to the buffer
			LineIterator it(src_gray, pt1, pt2, 8);
			LineIterator it2 = it;
			vector<double> buf(it.count);

			//for (int i = 0; i < it.count; i++, ++it)
			//{
			//	buf[i] = (const Vec3b)*it;
			//	cout << "buf[i] = " << buf[i] << "\n";
			//}
				

			// alternative way of iterating through the line
			for (int i = 0; i < it2.count; i++, ++it2)
			{
				cout << "it2.pos()= " << it2.pos() << "\n";
				double val = (double)src.at<uchar>(it2.pos());
				//Vec3b val = src.at<Vec3b>(it2.pos());
				cout << "val = " << val << "\n";
				//cout << "gray image= "<< (double)src_gray.at<uchar>(it2.pos()
				//buf[i] = val;
				buf.push_back(val);
				//cout << "buf[i] = " << buf[i] << "\n";
			}
			//cout << "buf= " << Mat(buf) << "\n";
			//cerr << Mat(buf) << endl;
		}
		else
		{
			pt1.x = 0;
			pt1.y = 0;
			pt2.x = 0;
			pt2.y = 0;
		}
		
		

	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		//cout << "position (" << x << ", " << y << ")" << "\n"<<endl;
		//cout << "intensity= " << (double)src.at<uchar>(Point(x, y)) << "\n";

	}
}
//set the callback function for any mouse event
//setMouseCallback("GrayScale Image", CallBackFunc, NULL);
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//Calculate Pixels per Angle Bins
Mat PixelsInBin(Mat grayImage, Mat AngleHist,int row_i, int col_j,int AngleHistSeq)
{
	PixelsInBinFile.open("PixelsInBinFile.txt", ios::app);
	//ofstream PixelsInBinFile;
	//string filename = "PixelsInBinFile_" + std::to_string(row_i) + std::to_string(col_j) + std::to_string(AngleHistSeq);
	//PixelsInBinFile.open(filename + ".txt");

	Mat test(grayImage.size().height,grayImage.size().width, CV_8UC1, Scalar(0, 0, 0));
	int x = 1 * AngleHistSeq;
	int A, B;
	int y = (AngleHistSeq-1) * 5;

	if (AngleHistSeq==1)
	{
		A = 0;
		B = 5;
	}
	else
	{
		A = y + 1;
		B = AngleHistSeq*5;
	}
	
	const float* Row_i = grayImage.ptr<float>(row_i);
	if (Row_i[col_j]>=A && Row_i[col_j]<B)
	{
		PixelsInBinFile << "GrayImage["<< row_i << ", " << col_j << "]= " << Row_i[col_j] << "\n";
		PixelsInBinFile << "A= " << A << ", " << "B= " << B << "\n";
		PixelsInBinFile << "\n";
	}
	//cout << "A= " << A << "\n";
	//cout << "B= " << B << "\n";
	//cout << "\n";
	PixelsInBinFile.close();
	return test;
	//1*0 = 0 -- -1+1 = 0
	//1*5 = 5 -- 1*5 = 5

	//5+1 = 6
	//2*5 = 10

	//10+1 = 11
	//3*5 = 15

	//15+1 = 16
	//4*5 = 20
}

// Computes the 1D histogram.
cv::MatND getHistogram(const cv::Mat &data, string file) {
	cv::MatND hist;

	ofstream myAngleHistFile;
	myAngleHistFile.open(file + ".txt");

	// Compute histogram
	//calcHist(&data, 1, 0, Mat(), angle_hist, 1, &histSize, &histRange, uniform, myAccumulate);
	calcHist(&data,
		1, // histogram from 1 image only
		channels, // the channel used
		cv::Mat(), // no mask is used
		hist, // the resulting histogram
		1, // it is a 1D histogram
		&histSize, // number of bins
		&histRange, // pixel value range
		uniform,
		myAccumulate
	);


	// Plot the histogram
	int hist_w = 512; int hist_h = 400;   myAngleHistFile << "hist_w= " << hist_w << "\n";    myAngleHistFile << "hist_h= " << hist_h << "\n";
	int bin_w = cvRound((double)hist_w / histSize);   myAngleHistFile << "bin_w= " << bin_w << "\n";

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));   myAngleHistFile << "Black Image of size(h,w)= (" << hist_h << "," << hist_w << ")" << "\n";
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	myAngleHistFile << "AngleLimit= " << AngleLimit << "\n";
	myAngleHistFile << "binSize= " << binSize << "\n";
	myAngleHistFile << "(number of bins) histSize= " << histSize << "\n";
	myAngleHistFile << "hist size= " << hist.size() << "\n";

	myAngleHistFile << "\n";    myAngleHistFile << "\n";

	//for (int i = 0; i<histSize; i++) {
	//	myAngleHistFile << "(Indices)-Value(" << i << "): " << hist.at<float>(i) << "\n";
	//}

	//myAngleHistFile << "\n";    myAngleHistFile << "\n";
	//for (MatConstIterator_<float> it = hist.begin<float>(); it != hist.end<float>(); it++) {
	//	myAngleHistFile << "(MatConstIterator)-Value: " << *it << "\n";
	//}
	//myAngleHistFile << "\n";    myAngleHistFile << "\n";

	//myAngleHistFile << "Line(" << 71 << ")= (" << bin_w*(70) << "," << hist_h - cvRound(hist.at<float>(70)) << ") TO (" << bin_w*(71) << "," << hist_h - cvRound(hist.at<float>(71)) << ")" << "\n";
	for (int i = 1; i < histSize; i++)
	{
		myAngleHistFile << "histSize(" << i - 1 << ")= " << hist.at<float>(i-1) << "   ------   ";
		myAngleHistFile << "Line(" << i - 1 << ")= (" << bin_w*(i - 1) << "," << hist_h - cvRound(hist.at<float>(i - 1)) << ") TO (" << bin_w*(i) << "," << hist_h - cvRound(hist.at<float>(i)) << ")" << "\n";

		

		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	myAngleHistFile << "\n";    myAngleHistFile << "\n";
	myAngleHistFile << "hist= " << hist << "\n";
	myAngleHistFile << "\n";    myAngleHistFile << "\n";
	myAngleHistFile << "histImage= " << histImage << "\n";
	namedWindow(file, 1);    imshow(file, histImage);
	myAngleHistFile.close();
	//waitKey(0);
	return hist;
}

int main()
{
	ofstream myEdgeDetectorFile;
	myEdgeDetectorFile.open("myEdgeDetectorFile.txt");

	//![load]
	//src = imread(argv[1], IMREAD_COLOR); // Load an image
	src = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	IplImage* img = cvLoadImage("20161215 02.33_368L.jpg");
	imshow("Original Image", src);


	if (src.empty())
	{
		return -1;
	}
	//![load]

	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
	//![convert_to_gray]
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	//![convert_to_gray]
	myEdgeDetectorFile << "src_gray= "<<src_gray << "\n";
	myEdgeDetectorFile << "src_gray size= " << src_gray.size() << "\n";
	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
	//grayHist = getHistogram(src_gray,"GrayScaleImageHist");
	imshow("GrayScale Image", src_gray);
	//set the callback function for any mouse event
	setMouseCallback("GrayScale Image", CallBackFunc, NULL);
	//![reduce_noise]
	//Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));
	//![reduce_noise]
	//cout << "src_gray = " << (double)src_gray.at<uchar>(2, 24) << "\n";
	//imshow("Blurred Image", detected_edges);
	//![canny]

	//Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	//![canny]
	//imshow("Canny Image", detected_edges);

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	//![sobel]

	//cv::Mat angle(src.size(), CV_64F);

	Mat Mag(src_gray.size(), CV_32FC1);
	Mat Angle(src_gray.size(), CV_32FC1);

	//convertScaleAbs(grad_x, abs_grad_x);
	//convertScaleAbs(grad_y, abs_grad_y);
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	//imshow("Sobel Demo - Simple Edge Detector", grad);
	//Mat orientation;
	////Mat orientation = Mat::zeros(abs_grad_x.rows, abs_grad_y.cols, CV_32F); //to store the gradients
	////grad_x.convertTo(grad_x,CV_32F);
	////grad_y.convertTo(grad_y, CV_32F);
	//phase(grad_x, grad_y, orientation,true);
	//cv::normalize(orientation, orientation, 0x00, 0xFF, cv::NORM_MINMAX, CV_8U);
	//namedWindow("Orientation", CV_WINDOW_AUTOSIZE);
	//imshow("Orientation", orientation);
	////myEdgeDetectorFile << "grad_x= " << grad_x << "\n";
	////myEdgeDetectorFile << "\n";
	////myEdgeDetectorFile << "grad_y= " << grad_y << "\n";
	////myEdgeDetectorFile << "\n";
	//myEdgeDetectorFile << "orientation size= " << orientation.size() << "\n";
	//myEdgeDetectorFile << "orientation= " << orientation << "\n";
	////myEdgeDetectorFile << "\n";

	double minM, maxM, minA, maxA;

	cartToPolar(grad_x, grad_y, Mag, Angle, true);

	angleHist = getHistogram(Angle,"AngleMatrixHist");
	//cv::minMaxLoc(Mag, &minM, &maxM);
	//cv::minMaxLoc(Angle, &minA, &maxA);
	myEdgeDetectorFile << "angleHist size(w,h)= "<< angleHist.size().width << ", "<< angleHist.size().height << "\n";
	myEdgeDetectorFile << "Original Image size= " << src.size() << "\n";
	myEdgeDetectorFile << "Angle size= " << Angle.size() << "\n";
	myEdgeDetectorFile << "\n";
	myEdgeDetectorFile << "angleHist= " << angleHist << "\n";

	//ofstream PixelsInBinFile;
	////string filename = "PixelsInBinFile";
	//PixelsInBinFile.open("PixelsInBinFile.txt");
	std::array<std::vector<int>, 72> vvv{ {} };
	cout << "gray scale image type= " << type2str(src_gray.type()) << "\n";
	for (size_t i = 0; i < Angle.rows; i++)
	{
		//myHistogramFile << i << "\n";
		const float* Row_i = Angle.ptr<float>(i);
		//const float* RowSG_i = src_gray.ptr<float>(i);
		//float RowSG2_i = src_gray.at<float>(i);
		//cout << "i" << i << "\n";
		
		for (size_t j = 0; j < Angle.cols; j++)
		{
			/// Establish the number of bins
			//int intensity = src_gray.at<int>(j,i);
			int histSize = 256;
			myEdgeDetectorFile << "(i,j)= (" << i << ", " << j << ")= " << Row_i[j] << "\n";
			//myEdgeDetectorFile << "(i,j)= (" << i << ", " << j << ")= " << Row_i[j] << "\n";
			myEdgeDetectorFile << "Row_i[j]/binSize= (" << Row_i[j] << ", " << binSize << ")= " << int(Row_i[j]/binSize) << "\n";
			//myEdgeDetectorFile << "src_gray[i=" << i << ",j=" <<  j << "]=" << RowSG_i[j] << "\n";
			myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << (double)src_gray.at<uchar>(i,j) << "\n";
			
			//vvv[i].at(j) = RowSG_i[j];
			//vvv[int(Row_i[j] / binSize)].push_back(RowSG_i[j]);
			vvv[int(Row_i[j] / binSize)].push_back((double)src_gray.at<uchar>(i, j));
			//if (int(Row_i[j] / binSize)==53)
			//{
			//	myEdgeDetectorFile << "----------------------------------"<<"\n";
			//	myEdgeDetectorFile << "Row_i[j]/binSize= (" << Row_i[j] << "/ " << binSize << ")= " << int(Row_i[j] / binSize) << "\n";
			//	//myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << RowSG_i[j] << "\n";
			//	myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << (double)src_gray.at<uchar>(i,j) << "\n";
			//}
			myEdgeDetectorFile << "\n";
			myEdgeDetectorFile << "\n";
			myEdgeDetectorFile << "\n";
			//myEdgeDetectorFile << Row_i[j] << "\n";

			//if (i==6 && j==274)
			//{
			//	cout << "src_gray[i=" << i <<",j=" << j <<"]="<<RowSG_i[j]<<"\n";
			//}
			//for (size_t k = 1; k < angleHist.rows; k++)
			//{
			//	binID = angleHist[k];
			//	pixelsBin = PixelsInBin(src_gray, angleHist, i,j,k);
			//}
		}
	}

	ofstream ArrayVectorFile;
	ArrayVectorFile.open("ArrayVectorFile.txt");
	ArrayVectorFile << "vvv size= " << vvv.size() << "\n";
	int mySum, myMean, mySD;
	for (int i = 0; i<vvv.size(); i++)
	{
		mySum, mySD, myMean = 0;
		//std::ArrayVectorFile << "v.size()= " << vvv.size() << "\n";
		ArrayVectorFile << "vvv[" << i << "].size()" << vvv[i].size() << "\n";
		for (int j = 0; j<vvv[i].size(); j++)
		{
			//ArrayVectorFile << "vvv size= " << vvv.size() << "\n";
			//ArrayVectorFile << "vvv[" << i<< "].size()" << vvv[i].size() << "\n";
			//if (!vvv[i].empty())
			//{
				ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
				mySum += vvv[i].at(j);
				//myMean = mean(vvv[i],NULL);
			//}
			//if (i==53)
			//{
			//	ArrayVectorFile << "\n";
			//	ArrayVectorFile << "\n";
			//	ArrayVectorFile << "\n";
			//	ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
			//}
			//std::cout << "2- i= "<<i<<"\n";
		}
		cout << "sum= " << mySum << "\n";
		//cout << "i= " << i << ", j= " << j << ", value= " << v[i].at(j)<< "\n";
		//std::cout << "3- i= "<<i<<"\n";
	}
	ArrayVectorFile.close();
	//myEdgeDetectorFile << "vvv= " << vvv << "\n";
	myEdgeDetectorFile.close();

	waitKey(0);
	return 0;
}