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
//#include "DrawlinesFromPointOnLine1.h"
//
//#include <algorithm>
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
//int length = 4;
//Point firstLine_pt2;
//Point firstLine_pt1;
//Point pt4;
//Point baseLine1, baseLine2;
//int lineSpacing = 20;
////LineIterator it2;
////int lineCounter = 0;
////LineIterator it;
////LineIterator it2;
////vector<Vec3b> buf;
//int thresh = 4;
//int max_thresh = 255;
//RNG rng(12345);
////![variables]
//
//template <typename T>
//cv::Mat plotGraph(std::vector<T>& vals, int YRange[2])
//{
//	//for (size_t i = 0; i < plotData.size(); i++)
//	//{
//	//	std::vector<T>& vals = plotData[i].numbers;
//	//	int YRange[2] = plotData[i].range;
//	//}
//	auto it = minmax_element(vals.begin(), vals.end());
//	float scale = 1. / ceil(*it.second - *it.first);
//	float bias = *it.first;
//	int rows = YRange[1] - YRange[0] + 1;
//	cv::Mat image = Mat::zeros(rows, vals.size(), CV_8UC3);
//	image.setTo(0);
//	for (int i = 0; i < (int)vals.size() - 1; i++)
//	{
//		cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i + 1, rows - 1 - (vals[i + 1] - bias)*scale*YRange[1]), Scalar(255, 0, 0), 1);
//		//cv::line(image, cv::Point(i, rows - 1 - (vals[i] - 1.23)*scale*YRange[1]), cv::Point(i + 1, rows - 1 - (vals[i + 1] - 1.23)*scale*YRange[1]), Scalar(0, 0, 255), 1);
//	}
//
//	return image;
//}
//
//void PickupPixels()
//{
//				//To pick up pixels under line
//				cv::LineIterator it(src_gray, pt1, pt2, 8);
//				//To store pixel picked up, from under line
//				std::vector<double> buf(it.count);
//				Mat temp = src_gray;
//				LineIterator it2 = it;
//
//				vector<int> numbers(it.count);
//
//				for (int i = 0; i < it.count; i++, ++it)
//				{
//					double val = (double)src_gray.at<uchar>(it.pos());				
//					buf[i] = val;
//					numbers.push_back(val);
//					cout << "position= " << it.pos() << ",  value= " << val << "\n";
//
//				}
//					//000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
//				//if (remove("Profile_DataFile.txt") != 0)
//				//	perror("Error deleting file");
//				//else
//				//	puts("File successfully deleted");
//
//				//				ofstream Profile_DataFile;
//				//				Profile_DataFile.open("Profile_DataFile.txt");
//				//				Profile_DataFile << Mat(numbers) << "\n";
//				//							//ContainerFile.open("ContainerFile_" + smi + ".txt");
//				//							//for (int i = 0; i < numbers.size(); i++)
//				//							//{
//				//							//	Profile_DataFile << "Profile[" << i << "].value= " << numbers.at(i) << "\n";
//				//							//}
//				//				Profile_DataFile.close();
//					//000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
//				#pragma region Plot-Mat
//						/*vector<int> numbers(100);*/
//						//std::iota(numbers.begin(), numbers.end(), 0);
//				
//						int range[2] = { 0, it.count };
//						cv::Mat lineGraph = plotGraph(numbers, range);
//				
//						imshow("plot", lineGraph);
//				#pragma endregion
//}
//
//void MyLine(Mat img, Point start, Point end)
//{
//	//cout << "baseLine1= " << baseLine1 << ",  baseLine2= " << baseLine2 << "\n";
//	//system("cls");
//
//	if (clickCounter ==/*1*/2)
//	{
//		clickCounter = 0;
//	}
//
//	//cout << "Drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;
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
//	drawCounter++;
//	//cout << "centre= " << (Point(start.x, start.y) + Point(end.x, end.y)) / 2 << "\n";
//	PickupPixels();
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
//
//		cout << "x= " << x << " ,y= " << y << "\n";
//		//circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
//		//imshow("Cont", src);
//
//		if (clickCounter == 1)
//		{
//			circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
//			//imshow("Cont", src);
////			double theta = clickCounter *10;
////			int length = 150;
//			pt1.x = x;
//			pt1.y = y;
////			baseLine1.x = pt1.x;
////			baseLine1.y = pt1.y;
////
////			/////Draw line based pnone point (pt1, angle(theta) and length of line
////			//pt2.x = (int)round(x + length * cos(theta * CV_PI / 180.0));
////			//pt2.y = (int)round(y + length * sin(theta * CV_PI / 180.0));
////			//MyLine(src, pt1, pt2);
////
////			drawCounter = 0;
//		}
//		else if (clickCounter == 2)
//		{
//			circle(src, Point(x, y), 1, (255, 0,0), 5, 8, 0);
//			pt2.x = x;
//			pt2.y = y;
//			MyLine(src, pt1, pt2);
//			//imshow("Cont", src);
//
//			clickCounter = 0;
//			drawCounter = 0;
//		}
//		else
//		{
//
//		}
////
/////*
////			int tempX = pt2.x - pt1.x;
////			int tempY = pt2.y - pt1.y;
////			if (tempX <tempY)
////			{
////				pt2.x = pt1.x;
////				pt2.y = y;
////			}
////			else if (tempX >tempY)
////			{
////				pt2.x = x;
////				pt2.y = pt1.y;
////			}
////			else
////			{
////				pt2.x = x;
////				pt2.y = y;
////			}
////				pt2.x = x;
////				pt2.y = y;
////			cout << "clickCounter= " << clickCounter << "\n";
////			pt2.x = x;
////			pt2.y = y;
////			cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;*/
////
////#pragma region Perpendicular Line
////			//	////Direction vector going from 'start' to 'end'
////			//	//Point v;
////			//	//v.x = pt2.x - pt1.x;
////			//	//v.y = pt2.y - pt1.y;
////
////			//	////normalize v
////			//	//int mag = sqrt(v.x*v.x + v.y*v.y);
////			//	//v.x = v.x / mag;
////			//	//v.y = v.y / mag;
////
////			//	////rotate and swap
////			//	//int tempX = -v.x;
////			//	//v.x = v.y;
////			//	//v.y = tempX;
////
////			//	////new line at 'end' pointing in v-direction
////			//	//Point pt3;
////			//	//pt3.x = pt2.x + v.x * length;
////			//	//pt3.y = pt2.y + v.y * length;
////			//	//pt4.x = pt2.x + v.x * (-length);
////			//	//pt4.y = pt2.y + v.y * (-length);  
////#pragma endregion
////			pt2.x = x;
////			pt2.y = y;
////			baseLine2.x = pt2.x;
////			baseLine2.y = pt2.y;
////			MyLine(src, pt1, pt2);
////
////			//To pick up pixels under line
////			cv::LineIterator it(src_gray, pt1, pt2, 8);
////			//To store pixel picked up, from under line
////			std::vector<double> buf(it.count);
////
////			LineIterator it2 = it;
////
////			for (int i = 0; i < it.count; i++, ++it)
////			{
////				double val = (double)src_gray.at<uchar>(it.pos());				
////				buf[i] = val;
////			}
////			ofstream file;
////			file.open("buf.csv");
////			file << Mat(buf) << "\n";
////			file.close();
////
////
////			int j = 0;
////			for (size_t i = 0; i < it.count; i++, ++it2)
////			{	// For the very 1st line at baseline's 1st coordinate
////				if (i == 0)
////				{
////					//cout << "1st LINE ON BASELINE:" << "\n";
////#pragma region Line Length=4
////					j++;
////					pt1 = it2.pos();
////					pt2 = Point(it2.pos().x + 100, it2.pos().y + 100);
////					//cout << "point from LineIterator= " << pt1 << "\n";
////#pragma endregion
////
////#pragma region Plot Orthogonal Lines
////					//Direction vector going from 'start' to 'end'
////					Point v;
////					v.x = baseLine2.x - baseLine1.x;
////					v.y = baseLine2.y - baseLine1.y;
////					//normalize v
////					int mag = sqrt(v.x*v.x + v.y*v.y);
////					v.x = v.x / mag;
////					v.y = v.y / mag;
////					//rotate and swap
////					int tempX = -v.x;
////					v.x = v.y;
////					v.y = tempX;
////					//new line at 'end' pointing in v-direction
////					Point pt3;
////					pt3.x = pt1.x + v.x * 4;
////					pt3.y = pt1.y + v.y * 4;
////					////pt4.x = pt1.x + v.x * (-length);
////					////pt4.y = pt1.y + v.y * (-length);  
////					////re-align to pt1 & pt2
////					////pt1 = pt2;
////					pt2 = pt3;
////#pragma endregion
////					//cout << "From pt1= " << pt1 << "  to  " << pt2 << "   (pt1 should be equal to baseline2)" << "\n";
////					MyLine(src, pt1, pt2);
////
////				}
////				else if (i == 20 * j)
////				{
////					pt1 = it2.pos();
////					pt2 = Point(it2.pos().x + 100, it2.pos().y + 100);
////					//cout << "point from LineIterator= " << pt1 << "\n";
////
////#pragma region Plot Orthogonal Lines
////					////Direction vector going from 'start' to 'end'
////					Point v;
////					v.x = pt1.x - baseLine1.x;
////					v.y = pt1.y - baseLine1.y;
////					//cout << "Direction vector going from 'start'-"<<pt1<<" to 'end'-"<<pt2 << "  = "<<v.x<<", "<<v.y<<"\n";
////					//normalize v
////					int mag = sqrt(v.x*v.x + v.y*v.y);
////					v.x = v.x / mag;
////					v.y = v.y / mag;
////					//cout << "Normalizing v" << "\n";
////					//rotate and swap
////					int tempX = -v.x;
////					v.x = v.y;
////					v.y = tempX;
////					//cout << "Rotating & swapping v" << "\n";
////					//new line at 'end' pointing in v-direction
////					pt4.x = pt1.x + v.x * (length);
////					pt4.y = pt1.y + v.y * (length);  
////					//cout << "New line:" << "\n";
////					//cout << "at" <<pt2<<" and "<< pt4<< "\n";
////					//re-align to pt1 & pt2
////					//pt1 = pt2;
////					pt2 = pt4;
////					//cout << "Re-Aligning to pt1 & pt2" << "\n";
////					//cout << "\n";cout << "\n";
////#pragma endregion
////					MyLine(src, pt1, pt2);
////					j++;
////				}
////				else
////				{
////				}
////			}
////
////		}
////		else
////		{
////			pt1.x = 0;
////			pt1.y = 0;
////			pt2.x = 0;
////			pt2.y = 0;
////		}
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
//		//cout << "x= " << x << " ,y= " << y << "\n";
//		//circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
//		//imshow("Cont", src);
//	}
//}
//
//int main()
//{
//	src = imread("20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
//	Mat testSrc = src;
//	//Plotting graph with Shervin Emami library
//	//uchar *dataMat = src.data;
//	//showUCharGraph("Pixel Values", dataMat, src.size().height*src.size().width);
//
//	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
//	Mat tempSrcGray = src_gray;
//	imshow("grayscale image", src_gray);
//
//	setMouseCallback("grayscale image", CallBackFunc, NULL);
//	//cv::Mat FltrBinaryImg = threshval < 128 ? (src_gray < threshval) : (src_gray > threshval);
//
//	//Mat FltrLabelImage;
//	//Mat FltrStats, FltrCentroids;
//	//int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
//	//string nFltrLabelsString = std::to_string(nFltrLabels);
//	//	std::vector<cv::Vec3b> FltrColors(nFltrLabels);
//	//	FltrColors[0] = cv::Vec3b(0, 0, 0);
//	//	ofstream connectedComponentsWithStats_MatrixFile;
//	//	connectedComponentsWithStats_MatrixFile.open("connectedComponentsWithStats_" + nFltrLabelsString + "_MatrixFile.txt");
//
//	//	for (int FltrLabel = 1; FltrLabel < 2/*nFltrLabels*/; ++FltrLabel)
//	//	{
//	//		FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
//	//		
//	//		connectedComponentsWithStats_MatrixFile << "Component No. " << FltrLabel << std::endl;
//	//		connectedComponentsWithStats_MatrixFile << "CC_STAT_LEFT -- The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
//	//		connectedComponentsWithStats_MatrixFile << "CC_STAT_TOP -- The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
//	//		connectedComponentsWithStats_MatrixFile << "CC_STAT_WIDTH --  The horizontal size of the bounding box= " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
//	//		connectedComponentsWithStats_MatrixFile << "CC_STAT_HEIGHT -- The vertical size of the bounding box. = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
//	//		connectedComponentsWithStats_MatrixFile << "CC_STAT_AREA -- The total area (in pixels) of the connected component.  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
//	//		connectedComponentsWithStats_MatrixFile << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;
//
//	//		//circle(src, Point(111, 10), 1, (0, 0, 255), 50, 8, 0);
//	//		//imshow("Plot Dot on SRC Image", src);
//	//		std::string s = std::to_string(FltrLabel);
//	//		// Get the mask for the i-th contour
//	//		if (FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) >= 300)
//	//		{
//	//			Mat mask2_i = FltrLabelImage == FltrLabel;
//	//			imwrite("mask2_i_" + s + ".bmp", mask2_i);
//	//		}
//	//		Mat mask_i = FltrLabelImage == FltrLabel;
//	//		string name = "mask_i_";
//
//	//		imwrite("mask_i_" + s + ".bmp", mask_i);
//	//		Mat Points;
//
//	//		double rectSize_b, rectSize_a;
//	//		Point maskCentroid;
//
//	//		//vector<Point> pts;
//	//		findNonZero(mask_i, Points);
//
//	//		RotatedRect box = minAreaRect(Points);
//	//		
//	//		Point2f vtx[4];
//	//		box.points(vtx);			
//	//		//box.angle returns angles in degrees
//	//		connectedComponentsWithStats_MatrixFile << "minAreaRect Angle= " << box.angle +180<< "\n";
//
//	//		maskCentroid = Point(FltrCentroids.at<double>(FltrLabel, 0),FltrCentroids.at<double>(FltrLabel, 1));
//
//	//		// Draw the bounding box
//	//		Mat tempSrc1 = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED);;
//	//		vector<double> lengths(4);
//	//		for (int i = 0; i < 4; i++)
//	//		{
//	//			line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
//	//			lengths.push_back(norm((vtx[(i + 1) % 4]) - (vtx[i])));
//	//		}
//	//		imshow("Bounding Box", tempSrc1);
//
//	//		rectSize_b = *max_element(lengths.begin(), lengths.end());
//
//	//		Point2d u;
//
//	//		u = Point2d(cos((box.angle + 180)* CV_PI / 180.0), sin((box.angle + 180))* CV_PI / 180.0);
//	//		circle(tempSrc1, u, 1, Scalar(100, 100, 100), 1, 8, 0);
//	//		imshow("u", tempSrc1);
//
//	//		line(tempSrc1, u, maskCentroid, Scalar(40, 255, 123), 1, LINE_AA);
//	//		imshow("line from centroid to u", tempSrc1);
//
//	//		Point2d u2 = u;
//	//		Point2d w;
//	//		vector<Point2d> Oxy(20);
//	//		double d = 0.1*rectSize_b;
//	//		Point2d centroid = maskCentroid;
//	//		Point2d TempCentroid = maskCentroid;
//	//		Point2d NewCoordSys;
//	//		circle(src, maskCentroid, 1, Scalar(0, 255, 0), 1, 8, 0);
//	//		Point t = centroid;
//	//		cout << "u2= " << u2 << "\n";
//	//		
//	//		double magU = sqrt(u2.x*u2.x + u2.y*u2.y);
//	//		u2.x = u2.x / magU;
//	//		u2.y = u2.y / magU;
//	//		for (size_t i = 0; i < 10; i++)
//	//		{
//	//				//rotate and swap
//	//		/*		double tempX = u.x;
//	//				u.x = -u.y;
//	//				u.y = tempX;*/
//
//	//				w.x = u.x + u2.x*d;
//	//				w.y = u.y + u2.y*d;
//	//				circle(src, w, 1, Scalar(0, 0, 255), 1, 8, 0);
//
//	//				Point2d tempW;
//	//				tempW.x = TempCentroid.x + u2.x*d;
//	//				//tempW.x = tempW.x - maskCentroid.x;
//	//				tempW.y = TempCentroid.y + u2.y*d;
//	//				//tempW.y = maskCentroid.y - tempW.y;
//	//				circle(src, tempW, 1, Scalar(0, 255, 255), 1, 8, 0);
//	//				TempCentroid = tempW;
//
//	//				u = w;
//	//			//}
//	//		}
//	//		imshow("Plot Image", src);
//
//	//		connectedComponentsWithStats_MatrixFile << "\n" << std::endl;
//
//	//	}
//	//	connectedComponentsWithStats_MatrixFile.close();
//	waitKey(0);
//	return 0;
//}