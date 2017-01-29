//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
////#include "iostream"
//#include <opencv2/opencv.hpp>
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include <iostream>
//#include <fstream>
////#include "MyEdgeDetector.h"
//#include <vector>
//
////#include "math.h"
////#include <stdlib.h>
//#include <array>
//#include "GraphUtils.h"
//#include "cvplot.h"
////#include <opencv2/legacy/compat.hpp>
//using namespace cv;
//using namespace std;
//using std::vector;
//
////![variables]
//Mat src, src_gray, srcCopy;
//Mat dst, detected_edges, angleHist, origHist, grayHist, pixelsBin, BinaryImg;
//int edgeThresh = 1;
//int lowThreshold;
//int const max_lowThreshold = 100;
//int ratio = 3;
//int kernel_size = 3;
//const char* window_name = "Edge Map";
//int ddepth = CV_32FC1;// CV_16S;
//int scale = 1;
//int delta = 0;
///// Generate grad_x and grad_y
//Mat grad_x, grad_y;
//Mat abs_grad_x, abs_grad_y;
//Mat grad;
////------------------Angle Histogram Parameters--------
//int binSize = 5;
//int AngleLimit = 360;
///// Establish the number of bins
//int histSize = AngleLimit / binSize;
////int histSize = 72;
///// Set the ranges ( for B,G,R) )
//float rangeA[] = { 0, 360 };
//const float* histRange = { rangeA };
//Mat angle_hist;
//bool uniform = true;
//bool myAccumulate = false;
//int channels[1];
//int binID;
//
//ofstream PixelsInBinFile;
//std::array<std::vector<int>, 2> vvv{ {} };
//
//int threshval = 60;
////string filename = "PixelsInBinFile";
////PixelsInBinFile.open("PixelsInBinFile.txt");
//int mouseX, mouseY;
//Point pt1, pt2;
//int clickCounter = 0, lineCounter = 0, pixelCounter = 0, drawCounter=0;
//std::vector<int> allObjPixelsCount;
////std::vector<double> buf;
//vector<Point> points1;
//vector<Point> points2;
//
//double angle_value_D = 45;
//int length = 100;
//Point firstLine_pt2;
//Point firstLine_pt1;
//Point pt4;
////int lineCounter = 0;
////LineIterator it;
////LineIterator it2;
////vector<Vec3b> buf;
////![variables]
//
//void MyLine(Mat img, Point start, Point end)
//{
//
//	if (clickCounter ==/*1*/2)
//	{
//		//clickCounter = 0;
//	}
//
//	cout << "Drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;
//
//	int thickness = 2;
//	int lineType = 8;
//
//		//double angle_value_R = atan(angle_value_D*(PI/180));
//	//end.x = (int)round(start.x + length * cos(angle_value_D * CV_PI / 180.0));
//	//end.y = (int)round(start.y + length * sin(angle_value_D * CV_PI / 180.0));
//
//	////Direction vector going from 'start' to 'end'
//	//Point v;
//	//v.x = end.x - start.x;
//	//v.y = end.y - start.y;
//
//	////normalize v
//	//int mag = sqrt(v.x*v.x + v.y*v.y);
//	//v.x = v.x / mag;
//	//v.y = v.y / mag;
//
//	////rotate and swap
//	//int tempX = -v.x;
//	//v.x = v.y;
//	//v.y = tempX;
//
//	////new line at 'end' pointing in v-direction
//	//Point C;
//	//C.x = end.x + v.x * length;
//	//C.y = end.y + v.y * length;
//	
//	line(img,
//		Point(start.x, start.y),
//		Point(end.x, end.y),
//		Scalar(255, 0, 0),
//		thickness,
//		lineType);
//
//	lineCounter++;
//	cout << "centre= " << (Point(start.x, start.y) + Point(end.x, end.y)) / 2 << "\n";
//
//	imshow("Drawn Image", img);
//}
//
//void CallBackFunc(int event, int x, int y, int flags, void* userdata)
//{
//	Mat newSrc = src;
//
//	if (event == EVENT_LBUTTONDOWN)
//	{
//		system("cls");
//		clickCounter++;
//		drawCounter++;
//		//double val = (double)BinaryImg.at<uchar>(y,x);
//		//cerr << "val= " << val << endl;
//		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//
//		if (drawCounter == 1)
//		{
//			double theta = clickCounter *10;
//			int length = 150;
//			cout << "clickCounter= " << clickCounter << "\n";
//			pt1.x = x;
//			pt1.y = y;
//			cout << "Left button of the mouse is clicked - position (" << pt1.x << ", " << pt1.y << ")" << endl;
//
//			///Draw line based pnone point (pt1, angle(theta) and length of line
//			pt2.x = (int)round(x + length * cos(theta * CV_PI / 180.0));
//			pt2.y = (int)round(y + length * sin(theta * CV_PI / 180.0));
//			MyLine(src, pt1, pt2);
//
//			cv::LineIterator it(src_gray, pt1, pt2, 8);
//			std::vector<double> buf(it.count);
//			std::vector<double> objPixelbuf(it.count);
//			//std::vector<cv::Point> points(it.count);
//			//points;
//			//imshow("image", BinaryImg);
//			//cout << "line= " << lineCounter << " -- no. of pixels= " << it.count << "\n";
//			for (int i = 0; i < it.count; i++, ++it)
//			{
//				double val = (double)src_gray.at<uchar>(it.pos());
//				if (val == 255)
//				{
//					objPixelbuf[i] = val;
//					pixelCounter++;
//					points1.push_back(pt1);
//					points2.push_back(pt2);
//				}
//				buf[i] = val;
//				cout << buf[i] << "\n";
//			}
//			cout << "line= " << lineCounter << " -- no. of pixels in object= " << pixelCounter << "\n";
//			allObjPixelsCount.push_back(pixelCounter);
//			drawCounter = 0;
//		}
//		else if (clickCounter == 2)
//		{
//			/*drawCounter = 0;*/
//
//			////int tempX = pt2.x - pt1.x;
//			////int tempY = pt2.y - pt1.y;
//			////if (tempX <tempY)
//			////{
//			////	pt2.x = pt1.x;
//			////	pt2.y = y;
//			////}
//			////else if (tempX >tempY)
//			////{
//			////	pt2.x = x;
//			////	pt2.y = pt1.y;
//			////}
//			////else
//			////{
//			////	pt2.x = x;
//			////	pt2.y = y;
//			////}
//			//	pt2.x = x;
//			//	pt2.y = y;
//			////cout << "clickCounter= " << clickCounter << "\n";
//			////pt2.x = x;
//			////pt2.y = y;
//			//cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;
//
//			//if (lineCounter == 0)
//			//{
//			//	//firstLine_pt1 = pt1;
//			//	//firstLine_pt2 = pt2;
//			//}
//
//			//if (lineCounter==1 /* = 2 */)
//			//{
//			//	//pt1 = firstLine_pt1;
//			//	//pt2 = firstLine_pt2;
//			//	////Direction vector going from 'start' to 'end'
//			//	//Point v;
//			//	//v.x = pt2.x - pt1.x;
//			//	//v.y = pt2.y - pt1.y;
//
//			//	////normalize v
//			//	//int mag = sqrt(v.x*v.x + v.y*v.y);
//			//	//v.x = v.x / mag;
//			//	//v.y = v.y / mag;
//
//			//	////rotate and swap
//			//	//int tempX = -v.x;
//			//	//v.x = v.y;
//			//	//v.y = tempX;
//
//			//	////new line at 'end' pointing in v-direction
//			//	//Point pt3;
//			//	//pt3.x = pt2.x + v.x * length;
//			//	//pt3.y = pt2.y + v.y * length;
//			//	//pt4.x = pt2.x + v.x * (-length);
//			//	//pt4.y = pt2.y + v.y * (-length);
//
//			//	//pt1 = pt2;
//			//	//pt2 = pt3;
//			//}
//			//MyLine(src, pt1, pt2);
//
//			////imshow("Drawn Image", src);
//			////lineCounter++;
//			//cv::LineIterator it(src_gray, pt1, pt2, 8);
//			//std::vector<double> buf(it.count);
//			//std::vector<double> objPixelbuf(it.count);
//			////std::vector<cv::Point> points(it.count);
//			////points;
//			////imshow("image", BinaryImg);
//			////cout << "line= " << lineCounter << " -- no. of pixels= " << it.count << "\n";
//			//for (int i = 0; i < it.count; i++, ++it)
//			//{
//			//	double val = (double)src_gray.at<uchar>(it.pos());
//			//	if (val == 255)
//			//	{
//			//		objPixelbuf[i] = val;
//			//		pixelCounter++;
//			//		points1.push_back(pt1);
//			//		points2.push_back(pt2);
//			//	}
//			//	buf[i] = val;
//			//	cout << buf[i] << "\n";
//			//}
//			//cout << "line= " << lineCounter << " -- no. of pixels in object= " << pixelCounter << "\n";
//			//allObjPixelsCount.push_back(pixelCounter);
//
//
//			////cout << "object pixel count= " << Mat(allObjPixelsCount) << "\n";
//
//
//			//ofstream file;
//			//file.open("buf.csv");
//			//file << Mat(buf) << "\n";
//			//file.close();
//
//			//if (lineCounter==2)
//			//{
//			//	////new line at 'end' pointing in opposite-v-direction
//
//			//	//pt1 = pt2;
//			//	//pt2 = pt4;
//			//	//MyLine(src, pt1, pt2);
//			//}
//		}
//		else
//		{
//			pt1.x = 0;
//			pt1.y = 0;
//			pt2.x = 0;
//			pt2.y = 0;
//		}
//	}
//	else if (event == EVENT_RBUTTONDOWN)
//	{
//		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//		cout << "object pixel count= " << Mat(allObjPixelsCount) << "\n";
//		cout << "average of object-pixel-count= " << mean(allObjPixelsCount) << "\n";
//		vector<int> ans;
//		for (size_t i = 0; i < allObjPixelsCount.size(); i++)
//		{
//			cout << "i= " << i << "  ||  " << allObjPixelsCount[i] << "-" << mean(allObjPixelsCount)[0] << "= " << allObjPixelsCount[i] - mean(allObjPixelsCount)[0] << "\n";
//			ans.push_back(abs(allObjPixelsCount[i] - mean(allObjPixelsCount)[0]));
//		}
//		double min, max;
//		Point min_loc, max_loc;
//		minMaxLoc(Mat(ans), &min, &max, &min_loc, &max_loc);
//		cout << "ANSWER= " << allObjPixelsCount[min_loc.y] << "\n";
//		//cout << "ANSWER line coordinates1= " << points1[min_loc.y] << "\n";
//		//cout << "ANSWER line coordinates2= " << points2[min_loc.y] << "\n";
//		//line(src, points1[min_loc.y], points2[min_loc.y], Scalar(0, 0, 255),2,8);
//		//imshow("Drawn Image", src);
//	}
//	else if (event == EVENT_MBUTTONDOWN)
//	{
//		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//	}
//	else if (event == EVENT_MOUSEMOVE)
//	{
//		//cout << "position (" << x << ", " << y << ")" << "\n"<<endl;
//		if (clickCounter == 1)
//		{
//			//if (x==pt1.x)
//			//{
//			//	cout << "pt1.x= " << pt1.x <<"\n" << endl;
//			//	cout << "x= " << x << "\n" << endl;
//			//}
//			//else if (y==pt1.y)
//			//{
//			//	cout << "pt1.y= " << pt1.y << "\n" << endl;
//			//	cout << "y= " << y << "\n" << endl;
//			//}
//			//else
//			//{
//			//	system("cls");
//			//}
//		}
//		//cout << "intensity= " << (double)src.at<uchar>(Point(x, y)) << "\n";
//
//	}
//	//else if (event == eve)
//	//{
//
//	//}
//}
//
//int main()
//{
//	src = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
//
//	//Plotting graph with Shervin Emami library
//	uchar *dataMat = src.data;
//	showUCharGraph("Pixel Values", dataMat, src.size().height*src.size().width);
//
//	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
//	Mat tempSrcGray = src_gray;
//	imshow("grayscale image", src_gray);
//	setMouseCallback("grayscale image", CallBackFunc, NULL);
//
//	waitKey(0);
//	return 0;
//}