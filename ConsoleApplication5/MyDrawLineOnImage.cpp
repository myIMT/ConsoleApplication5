//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "iostream"
//#include <opencv2/opencv.hpp>
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <stdlib.h>
//#include <array>
////#include <opencv2/legacy/compat.hpp>
//
//using namespace cv;
//using namespace std;
//using std::vector;
//
//Mat src, src_gray;
//Point pt1, pt2;
//int clickCounter;
//
//void MyLine(Mat img, Point start, Point end)
//{
//	cout << "Drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;
//
//	int thickness = 20;
//	int lineType = 8;
//	line(img,
//		Point(pt1.x,pt1.y),
//		Point(pt2.x, pt2.y),
//		Scalar(255, 0, 0),
//		5,
//		8);	imshow("Original Image", src);
//}
//
//void CallBackFunc(int event, int x, int y, int flags, void* userdata)
//{
//	if (event == EVENT_LBUTTONDOWN)
//	{
//		clickCounter++;
//		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//		if (clickCounter == 1)
//		{
//			cout << "clickCounter= " << clickCounter << "\n";
//			pt1.x = x;
//			pt1.y = y;
//			cout << "Left button of the mouse is clicked - position (" << pt1.x << ", " << pt1.y << ")" << endl;
//		}
//		else if (clickCounter == 2)
//		{
//			cout << "clickCounter= " << clickCounter << "\n";
//			pt2.x = x;
//			pt2.y = y;
//			cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;
//
//			MyLine(src, pt1, pt2);
//
//			//// grabs pixels along the line (pt1, pt2)
//			//// from 8-bit 3-channel image to the buffer
//			//LineIterator it(src_gray, pt1, pt2, 8);
//			//LineIterator it2 = it;
//			//vector<double> buf(it.count);
//
//			////for (int i = 0; i < it.count; i++, ++it)
//			////{
//			////	buf[i] = (const Vec3b)*it;
//			////	//cout << "buf[i] = " << buf[i] << "\n";
//			////}
//
//
//			//// alternative way of iterating through the line
//			//for (int i = 0; i < it2.count; i++, ++it2)
//			//{
//			//	cout << "it2.pos()= " << it2.pos() << "\n";
//			//	double val = (double)src.at<uchar>(it2.pos());
//			//	//Vec3b val = src.at<Vec3b>(it2.pos());
//			//	cout << "val = " << val << "\n";
//			//	//buf[i] = val;
//			//	buf.push_back(val);
//			//	//cout << "buf[i] = " << buf[i] << "\n";
//			//}
//			//cout << "buf= " << Mat(buf) << "\n";
//			////cerr << Mat(buf) << endl;
//		}
//		else
//		{
//			pt1.x = 0;
//			pt1.y = 0;
//			pt2.x = 0;
//			pt2.y = 0;
//		}
//
//
//
//	}
//	else if (event == EVENT_RBUTTONDOWN)
//	{
//		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//	}
//	else if (event == EVENT_MBUTTONDOWN)
//	{
//		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//	}
//	else if (event == EVENT_MOUSEMOVE)
//	{
//		//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
//
//	}
//}
//
//int main()
//{
//	ofstream myEdgeDetectorFile;
//	myEdgeDetectorFile.open("myEdgeDetectorFile.txt");
//
//	src = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
//	imshow("Original Image", src);
//
//	//set the callback function for any mouse event
//	setMouseCallback("Original Image", CallBackFunc, NULL);
//
//	if (src.empty())
//	{
//		return -1;
//	}
//	myEdgeDetectorFile << "src= " << src_gray << "\n";
//	myEdgeDetectorFile.close();
//
//	waitKey(0);
//	return 0;
//}
//
////int main()
////{
////	// Create black empty images
////	Mat image = Mat::zeros(400, 400, CV_8UC3);
////
////	// Draw a line 
////	line(image, Point(15, 20), Point(70, 50), Scalar(110, 220, 0), 2, 8);
////	imshow("Image", image);
////
////	waitKey(0);
////	return(0);
////}