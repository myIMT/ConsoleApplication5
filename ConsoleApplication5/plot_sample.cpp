//// Matlab style plot functions for OpenCV by Changbo (zoccob@gmail).
//
////#include "cv.h"
////#include "highgui.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/opencv.hpp>
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include <iostream>
//#include <fstream>
//#include "cvplot.h"
//#include <stdlib.h>
//
//#include <algorithm>
//#include <iostream>
//#include <list>
//#include <numeric>
//#include <random>
//#include <vector>
//
//#define rowPtr(imagePtr, dataType, lineIndex) \
//	    (dataType *)(imagePtr->imageData + (lineIndex) * imagePtr->widthStep)
//
//using namespace cv;
//using namespace std;
//
//Mat img;
//Mat grayImg;
//Point point_one;
//Point point_two;
//int clickCount = 0;
//			struct buffer {
//				std::vector<double> pixValues;
//				Point2f startPoint;
//				Point2f endPoint;
//				//int j;
//				//int angle;
//				//int value;
//			};
//vector<buffer> Profiles;
//int ProfilesCount = 0;
//
//template <typename T>
//cv::Mat plotGraph(std::vector<T>& vals, int YRange[2])
//{
//
//	auto it = minmax_element(vals.begin(), vals.end());
//	float scale = 1. / ceil(*it.second - *it.first);
//	float bias = *it.first;
//	int rows = YRange[1] - YRange[0] + 1;
//	cv::Mat image = Mat::zeros(rows, vals.size(), CV_8UC3);
//	image.setTo(0);
//	for (int i = 0; i < (int)vals.size() - 1; i++)
//	{
//		cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i + 1, rows - 1 - (vals[i + 1] - bias)*scale*YRange[1]), Scalar(255, 0, 0), 1);
//	}
//
//	return image;
//}
//
//
//void CallBackFunc(int event, int x, int y, int flags, void* userdata)
//{
//	if (event == EVENT_LBUTTONDOWN)
//	{
//		if (clickCount == 0)
//		{
//			cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//			//cout << "Pixel value = (" << grayImg.at<uchar>(Point(x,y)) << ")" << endl;
//			point_one = Point(x, y);
//			clickCount += 1;
//		}
//		else if (clickCount == 1)
//		{
//			cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//			//cout << "Pixel value = (" << grayImg.at<uchar>(Point(x,y)) <<")" << endl;
//			point_two = Point(x, y);
//			clickCount += 1;
//		}
//		else if (clickCount == 2)
//		{
//			#pragma region DrawLines
//							int thickness = 0.2;
//							int lineType = 8;
//							line(grayImg,
//								point_one,
//								point_two,
//								Scalar(255, 0, 0),
//								thickness,
//								lineType);
//			
//							//profileLinesCount += 1;
//							//std::string profileLinesCount1 = std::to_string(profileLinesCount);
//							imshow("Line-", grayImg); 
//			#pragma endregion
//
//			#pragma region LinePixels
//			
//							// grabs pixels along the line (pt1, pt2)
//							LineIterator it1A(grayImg, point_one, point_two, 8);		// Lines after centroid
//
//							LineIterator it11A = it1A;
//
//							vector<float> pixelsOnLineA;
//							vector<vector<float>> pixels;
//			
//							Mat linePixel;
//			
//							// Record pixels under line A (After centroid)
//							for (int l = 0; l < it1A.count; l++, ++it1A)
//							{
//								Profiles.push_back(buffer());
//								Profiles[ProfilesCount].startPoint = point_one;
//								Profiles[ProfilesCount].endPoint = point_two;
//								double valA = (double)grayImg.at<uchar>(it1A.pos());
//								pixelsOnLineA.push_back(valA);
//								linePixel.push_back(valA);
//
//								Profiles[ProfilesCount].pixValues.push_back(valA);// (double)tempSrc3.at<uchar>(it1.pos());
//
//								ProfilesCount += 1;
//							}
//			#pragma endregion
//		}
//		else
//		{
//			clickCount = 0;
//		}
//
//	}
//	else if (event == EVENT_RBUTTONDOWN)
//	{
//		//cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//	}
//	else if (event == EVENT_MBUTTONDOWN)
//	{
//		//cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//	}
//	else if (event == EVENT_MOUSEMOVE)
//	{
//		//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
//
//	}
//}
//
//
//int main(int argc, char* argv[])
//{
//		// Read image from file 
//		img = imread("20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg");
//		cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
//
//		//Create a window
//		namedWindow("My Window", 1);
//	
//		//set the callback function for any mouse event
//		setMouseCallback("My Window", CallBackFunc, NULL);
//		
//		//show the image
//		imshow("My Window", grayImg);
//
//
////#pragma region Plot-Mat
////		vector<int> numbers(100);
////		std::iota(numbers.begin(), numbers.end(), 0);
////
////		int range[2] = { 0, 100 };
////		cv::Mat lineGraph = plotGraph(numbers, range);
////
////		imshow("plot", lineGraph);
////#pragma endregion
//
//
//	//// load an image
//	//char *imagefile = "20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg";
//
//	////IplImage *image = cvLoadImage(imagefile);
//	//Mat myImage = imread("20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg", CV_LOAD_IMAGE_UNCHANGED);
//
//	//IplImage *image = cvLoadImage(imagefile);
//	//cv::Mat m = cv::cvarrToMat(image);
//	////IplImage *myImage = image;
//	//if (m.empty())
//	//{
//	//	std::cout << "image error: " << imagefile << std::endl << std::flush;
//	//	return -1;
//	//}
//
//	//// show an image
//	////cvShowImage("original", image);
//	//imshow("original", m);
//
//	//// plot and label:
//	////
//	//// template<typename T>
//	//// void plot(const string figure_name, const T* p, int count, int step = 1,
//	////		     int R = -1, int G = -1, int B = -1);
//	////
//	//// figure_name: required. multiple calls of this function with same figure_name
//	////              plots multiple curves on a single graph.
//	//// p          : required. pointer to data.
//	//// count      : required. number of data.
//	//// step       : optional. step between data of two points, default 1.
//	//// R, G,B     : optional. assign a color to the curve.
//	////              if not assigned, the curve will be assigned a unique color automatically.
//	////
//	//// void label(string lbl):
//	////
//	//// label the most recently added curve with lbl.
//	////
//	//
//	//// specify a line to plot
//	//int the_line = 1;
//	//
//	//int key = -1;
//	//cout << "image height" << m.size().height << "\n";
//
//	//IplImage* image2 = cvCloneImage(&(IplImage)m);
//
//	//while (the_line < m.size().height)
//	//{
//	//	unsigned char *pb = rowPtr(image2, unsigned char, the_line);
//	//	int width = m.size().width;
//	//	
//	//	CvPlot::plot("RGB", pb+0, width, 3);
//	//	CvPlot::label("B");
//	//	CvPlot::plot("RGB", pb+1, width, 3, 255, 0, 0);
//	//	CvPlot::label("G");
//	//	CvPlot::plot("RGB", pb+2, width, 3, 0, 0, 255);
//	//	CvPlot::label("R");
//	//	
//
//	//	key = cvWaitKey(0);
//
//	//	if (key == 32)
//	//	{
//	//		// plot the next line
//	//		the_line++;
//	//		// clear previous plots
//	//		CvPlot::clear("RGB");
//	//	}
//	//	else
//	//	{
//	//		break;
//	//	}
//	//}
//
//	//cvReleaseImage(&image);
//	waitKey(0);
//	return 0;
//}
//
//
//
