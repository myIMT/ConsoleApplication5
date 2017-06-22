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

//#include "math.h"
//#include <stdlib.h>
#include <array>
#include "GraphUtils.h"
#include "cvplot.h"
#include "DrawlinesFromPointOnLine1.h"

#include <algorithm>
//#include <opencv2/legacy/compat.hpp>
using namespace cv;
using namespace std;
using std::vector;

#pragma region Variables
//![variables]
Mat src, src_gray, srcCopy;
Mat dst, detected_edges, angleHist, origHist, grayHist, pixelsBin, BinaryImg;
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

int threshval = 60;
//string filename = "PixelsInBinFile";
//PixelsInBinFile.open("PixelsInBinFile.txt");
int mouseX, mouseY;
Point pt1, pt2;
int clickCounter = 0, lineCounter = 0, pixelCounter = 0, drawCounter = 0;
std::vector<int> allObjPixelsCount;
//std::vector<double> buf;
vector<Point> points1;
vector<Point> points2;

double angle_value_D = 45;
int length = 4;
Point firstLine_pt2;
Point firstLine_pt1;
Point pt4;
Point baseLine1, baseLine2;
int lineSpacing = 20;
//LineIterator it2;
//int lineCounter = 0;
//LineIterator it;
//LineIterator it2;
//vector<Vec3b> buf;
int thresh = 4;
int max_thresh = 255;
RNG rng(12345);

vector<int> numbers;
vector<vector<int>> values;
bool distanceSet = false;
Point distancePoint1, distancePoint2, distancePoint3, heigthPoint;
double distancePoint;
#pragma endregion


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
void Plot()
{
	////IplImage *graphImg = drawIntGraph(&values[0][0], values[0].size(), NULL, -25, 25, 400, 180, "X Angle (blue is truth, green is POSIT)");
	//drawIntGraph(&values[0][0], values[0].size(), NULL, -25, 25, 400, 180, "X Angle (blue is truth, green is POSIT)");
	////values.push_back(numbers);
#pragma region My Min and Max
	int myMin = -1;
	int myMax = -1;
	int myWidth = -1;
	int myHeight = -1;
	vector<int> smallestElement;
	vector<int> biggestElement;
	//int smallestSize;
	vector<int> mySize;
	for (int k = 0; k < values.size(); k++)
	{
		auto small = min_element(begin(values[k]), end(values[k]));
		smallestElement.push_back(*small);

		auto biggest = max_element(begin(values[k]), end(values[k]));
		biggestElement.push_back(*biggest);

		//cout << "Line " << k << " -- smallest element -- " << *small << "\n";
		//cout << "Line " << k << " --  biggest element -- " << *biggest << "\n";
		//cout << "Line " << k << " --  width -- " << values[k].size() << "\n";

		mySize.push_back(values[k].size());
	}
	auto smallestSize = min_element(begin(mySize), end(mySize));
	auto biggesttSize = max_element(begin(mySize), end(mySize));

	auto smallElement = min_element(begin(smallestElement), end(smallestElement));
	auto bigElement = max_element(begin(biggestElement), end(biggestElement));

	//cout << "smallest element -- " << *smallElement << "\n";
	//cout << "biggest element -- " << *bigElement << "\n";
	//cout << "smallest size -- " << *smallestSize << "\n";
	//cout << "biggest size -- " << *biggesttSize << "\n";

#pragma endregion


	IplImage *graphImg = drawIntGraph(&values[0][0], values[0].size(), NULL, *smallElement, *bigElement, *biggesttSize, *bigElement, "My Profile lines",true);
	//showIntGraph("Rotation Angle", &values[0][0], values[0].size());

	for (int i = 0; i < values.size(); i++)
	{
		//cout << "i= " << i << "/n";

		//showIntGraph("Plots", &values[i][0], values[i].size());
		drawIntGraph(&values[i][0], values[i].size(), graphImg, *smallElement, *bigElement, *biggesttSize, *bigElement);
		////cout << "POINTS (pt1, pt2)= (" << pt1 << ", " << pt2 << ")" << endl;
		////IplImage *graphImg = drawIntGraph(&values[i][0], values[i][].size(), NULL,-25, 25, 400, 180, "X Angle (blue is truth, green is POSIT)");

		//cvSaveImage("my_graph.jpg", graphImg);
		//cvReleaseImage(&graphImg);

		for (int j = 1; j < values[i].size(); j++)
		{
			//cout << "i= " << i << " , " << ";= " << j<< "/n";
			//cout << values[i][j];
			////cout << vec[i][j];
			//IplImage *graphImg = drawFloatGraph(&floatVec1[0], floatVec1.size(), NULL,
			////	-25, 25, 400, 180, "X Angle (blue is truth, green is POSIT)");
			//drawIntGraph(&values[i][j], values[i].size(), graphImg, -25, 25, 400, 180);
			////cvSaveImage("my_graph.jpg", graphImg);
			////cvReleaseImage(&graphImg);
		}
		//cout << "\n";
		//cout << "&&&&&&&&&&&&&&&&&&&&&&&";
		////cout << "-------------------------------------------------------------------------------------------------";
		//cout << "\n";
		//cout << "\n";
	}
	showImage(graphImg);

	cvSaveImage("my_graph.jpg", graphImg);
	cvReleaseImage(&graphImg);

}
void PickupPixels()
{
	//To pick up pixels under line
	cv::LineIterator it(src_gray, pt1, pt2, 8);
	//To store pixel picked up, from under line
	std::vector<double> buf(it.count);
	Mat temp = src_gray;
	LineIterator it2 = it;

	/*vector<int> numbers(it.count);*/

	for (int i = 0; i < it.count; i++, ++it)
	{
		double val = (double)src_gray.at<uchar>(it.pos());
		buf[i] = val;
		numbers.push_back(val);
		//cout << "position= " << it.pos() << ",  value= " << val << "\n";

	}
	values.push_back(numbers);
	//000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
	//if (remove("Profile_DataFile.txt") != 0)
	//	perror("Error deleting file");
	//else
	//	puts("File successfully deleted");

	//				ofstream Profile_DataFile;
	//				Profile_DataFile.open("Profile_DataFile.txt");
	//				Profile_DataFile << Mat(numbers) << "\n";
	//							//ContainerFile.open("ContainerFile_" + smi + ".txt");
	//							//for (int i = 0; i < numbers.size(); i++)
	//							//{
	//							//	Profile_DataFile << "Profile[" << i << "].value= " << numbers.at(i) << "\n";
	//							//}
	//				Profile_DataFile.close();
	//000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#pragma region Plot-Mat

	//values.push_back(numbers);
	//for (int i = 0; i < values.size(); i++)
	//{
	//	cout << "POINTS (pt1, pt2)= (" << pt1 << ", " << pt2 << ")" << endl;

	//	for (int j = 0; j < values[i].size(); j++)
	//	{
	//		cout << values[i][j];
	//		//cout << vec[i][j];
	//	}

	//	cout << "-------------------------------------------------------------------------------------------------";
	//	cout << "\n";
	//	cout << "\n";
	//}




	////int range[2] = { 0, it.count };
	//showIntGraph("Rotation Angle", &numbers[0], numbers.size());

//#pragma region Multiple Plots
//	IplImage *graphImg = drawIntGraph(&floatVec1[0], floatVec1.size(), NULL,
//		-25, 25, 400, 180, "X Angle (blue is truth, green is POSIT)");
//	drawFloatGraph(&floatVec2[0], floatVec2.size(), graphImg, -25, 25, 400, 180);
//	cvSaveImage("my_graph.jpg", graphImg);
//	cvReleaseImage(&graphImg);
//#pragma endregion

	//cv::Mat lineGraph = plotGraph(numbers, range);

	//imshow("plot", lineGraph);
#pragma endregion
}

void MyLine(Mat img, Point start, Point end)
{
	//cout << "baseLine1= " << baseLine1 << ",  baseLine2= " << baseLine2 << "\n";
	//system("cls");

	if (clickCounter ==/*1*/2)
	{
		//clickCounter = 0;
	}

	//cout << "Drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;

	int thickness = 0.5;
	int lineType = 8;

	//double angle_value_R = atan(angle_value_D*(PI/180));
	//end.x = (int)round(start.x + length * cos(angle_value_D * CV_PI / 180.0));
	//end.y = (int)round(start.y + length * sin(angle_value_D * CV_PI / 180.0));

	////Direction vector going from 'start' to 'end'
	//Point v;
	//v.x = end.x - start.x;
	//v.y = end.y - start.y;

	////normalize v
	//int mag = sqrt(v.x*v.x + v.y*v.y);
	//v.x = v.x / mag;
	//v.y = v.y / mag;

	////rotate and swap
	//int tempX = -v.x;
	//v.x = v.y;
	//v.y = tempX;

	////new line at 'end' pointing in v-direction
	//Point C;
	//C.x = end.x + v.x * length;
	//C.y = end.y + v.y * length;

	line(img,
		Point(start.x, start.y),
		Point(end.x, end.y),
		Scalar(255, 0, 0),
		thickness,
		lineType);

	lineCounter++;
	drawCounter++;
	//cout << "centre= " << (Point(start.x, start.y) + Point(end.x, end.y)) / 2 << "\n";
	PickupPixels();
	imshow("Drawn Image", img);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	Mat newSrc = src;

	if (event == EVENT_LBUTTONDOWN)
	{
		//system("cls");
		clickCounter++;
		drawCounter++;

		//cout << "x= " << x << " ,y= " << y << "\n";
		//circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
		//imshow("Cont", src);

		if (clickCounter == 1)
		{
			if (!distanceSet)
			{
				cout << "Point 1: (" << x << ", "<< y << ")" << "\n";

				distancePoint1 = Point(x, y);
				circle(src, distancePoint1, 1, (0, 0, 255), 5, 8, 0);
			}
			else
			{
				////system("cls");

				//circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
				////imshow("Cont", src);
				////			double theta = clickCounter *10;
				////			int length = 150;
				//pt1.x = x;
				//pt1.y = y;
				//cout << "point 1= " << pt1 << "\n";
				////			baseLine1.x = pt1.x;
				////			baseLine1.y = pt1.y;
				////
				////			/////Draw line based pnone point (pt1, angle(theta) and length of line
				////			//pt2.x = (int)round(x + length * cos(theta * CV_PI / 180.0));
				////			//pt2.y = (int)round(y + length * sin(theta * CV_PI / 180.0));
				////			//MyLine(src, pt1, pt2);
				////
				////			drawCounter = 0;
			}
		}
		else if (clickCounter == 2)
		{
			if (!distanceSet)
			{
				cout << "Point 2: (" << x << ", " << y << ")" << "\n";

				distancePoint2 = Point(x, y);
				circle(src, distancePoint2, 1, (0, 0, 255), 5, 8, 0);

				distancePoint3 = distancePoint2 - distancePoint1;
				distancePoint = sqrt(distancePoint3.ddot(distancePoint3));

				cout << "Distance between two points (line width): " << distancePoint << "\n";
				cout << "Drawing straight linebased on Point 1 and Point 2" << "\n";
				circle(src, Point(distancePoint2.x, distancePoint1.y), 1, (0, 0, 255), 5, 8, 0);
				//MyLine(src, distancePoint1, Point(distancePoint2.x, distancePoint1.y));
				cout << "Now, in relation one of existing points, click one more time to specify where line-drawing should end" << "\n";

				distanceSet = true;

				//clickCounter = 0;
				//drawCounter = 0;
			}
			else
			{
				//MyLine();
				//circle(src, Point(x, y), 1, (255, 0, 0), 5, 8, 0);
				//pt2.x = x;
				//pt2.y = y;
				//cout << "point 2= " << pt2 << "\n";
				//MyLine(src, pt1, pt2);
				////imshow("Cont", src);
				////1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
				//////values.push_back(numbers);
				////for (int i = 0; i < values.size(); i++)
				////{
				////	cout << "POINTS (pt1, pt2)= (" << pt1 << ", " << pt2 << ")" << endl;

				////	for (int j = 0; j < values[i].size(); j++)
				////	{
				////		cout << values[i][j];
				////		//cout << vec[i][j];
				////	}

				////	cout << "-------------------------------------------------------------------------------------------------";
				////	cout << "\n";
				////	cout << "\n";
				////}
				////1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
				//clickCounter = 0;
				//drawCounter = 0;
					
			}
		}
		else if(clickCounter == 3)
		{
			if (distanceSet)
			{
				cout << "Point 3: (" << x << ", " << y << ")" << "\n";

				heigthPoint.y = y;
				heigthPoint.x = distancePoint1.x;
				circle(src, heigthPoint, 1, (0, 0, 255), 5, 8, 0);
				cout << "heigthPoint: " << heigthPoint << "\n";
				for (int m = distancePoint1.y; m < heigthPoint.y; m+=2)
				{					
					//int yn = distancePoint1.y + m;
					MyLine(src, Point(distancePoint1.x, m), Point(distancePoint2.x, m));
				}
			}
			clickCounter = 0;
			distanceSet = false;
			cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << "\n";
				
		}
		else
		{

		}
		//
		///*
		//			int tempX = pt2.x - pt1.x;
		//			int tempY = pt2.y - pt1.y;
		//			if (tempX <tempY)
		//			{
		//				pt2.x = pt1.x;
		//				pt2.y = y;
		//			}
		//			else if (tempX >tempY)
		//			{
		//				pt2.x = x;
		//				pt2.y = pt1.y;
		//			}
		//			else
		//			{
		//				pt2.x = x;
		//				pt2.y = y;
		//			}
		//				pt2.x = x;
		//				pt2.y = y;
		//			cout << "clickCounter= " << clickCounter << "\n";
		//			pt2.x = x;
		//			pt2.y = y;
		//			cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;*/
		//
		//#pragma region Perpendicular Line
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
		//#pragma endregion
		//			pt2.x = x;
		//			pt2.y = y;
		//			baseLine2.x = pt2.x;
		//			baseLine2.y = pt2.y;
		//			MyLine(src, pt1, pt2);
		//
		//			//To pick up pixels under line
		//			cv::LineIterator it(src_gray, pt1, pt2, 8);
		//			//To store pixel picked up, from under line
		//			std::vector<double> buf(it.count);
		//
		//			LineIterator it2 = it;
		//
		//			for (int i = 0; i < it.count; i++, ++it)
		//			{
		//				double val = (double)src_gray.at<uchar>(it.pos());				
		//				buf[i] = val;
		//			}
		//			ofstream file;
		//			file.open("buf.csv");
		//			file << Mat(buf) << "\n";
		//			file.close();
		//
		//
		//			int j = 0;
		//			for (size_t i = 0; i < it.count; i++, ++it2)
		//			{	// For the very 1st line at baseline's 1st coordinate
		//				if (i == 0)
		//				{
		//					//cout << "1st LINE ON BASELINE:" << "\n";
		//#pragma region Line Length=4
		//					j++;
		//					pt1 = it2.pos();
		//					pt2 = Point(it2.pos().x + 100, it2.pos().y + 100);
		//					//cout << "point from LineIterator= " << pt1 << "\n";
		//#pragma endregion
		//
		//#pragma region Plot Orthogonal Lines
		//					//Direction vector going from 'start' to 'end'
		//					Point v;
		//					v.x = baseLine2.x - baseLine1.x;
		//					v.y = baseLine2.y - baseLine1.y;
		//					//normalize v
		//					int mag = sqrt(v.x*v.x + v.y*v.y);
		//					v.x = v.x / mag;
		//					v.y = v.y / mag;
		//					//rotate and swap
		//					int tempX = -v.x;
		//					v.x = v.y;
		//					v.y = tempX;
		//					//new line at 'end' pointing in v-direction
		//					Point pt3;
		//					pt3.x = pt1.x + v.x * 4;
		//					pt3.y = pt1.y + v.y * 4;
		//					////pt4.x = pt1.x + v.x * (-length);
		//					////pt4.y = pt1.y + v.y * (-length);  
		//					////re-align to pt1 & pt2
		//					////pt1 = pt2;
		//					pt2 = pt3;
		//#pragma endregion
		//					//cout << "From pt1= " << pt1 << "  to  " << pt2 << "   (pt1 should be equal to baseline2)" << "\n";
		//					MyLine(src, pt1, pt2);
		//
		//				}
		//				else if (i == 20 * j)
		//				{
		//					pt1 = it2.pos();
		//					pt2 = Point(it2.pos().x + 100, it2.pos().y + 100);
		//					//cout << "point from LineIterator= " << pt1 << "\n";
		//
		//#pragma region Plot Orthogonal Lines
		//					////Direction vector going from 'start' to 'end'
		//					Point v;
		//					v.x = pt1.x - baseLine1.x;
		//					v.y = pt1.y - baseLine1.y;
		//					//cout << "Direction vector going from 'start'-"<<pt1<<" to 'end'-"<<pt2 << "  = "<<v.x<<", "<<v.y<<"\n";
		//					//normalize v
		//					int mag = sqrt(v.x*v.x + v.y*v.y);
		//					v.x = v.x / mag;
		//					v.y = v.y / mag;
		//					//cout << "Normalizing v" << "\n";
		//					//rotate and swap
		//					int tempX = -v.x;
		//					v.x = v.y;
		//					v.y = tempX;
		//					//cout << "Rotating & swapping v" << "\n";
		//					//new line at 'end' pointing in v-direction
		//					pt4.x = pt1.x + v.x * (length);
		//					pt4.y = pt1.y + v.y * (length);  
		//					//cout << "New line:" << "\n";
		//					//cout << "at" <<pt2<<" and "<< pt4<< "\n";
		//					//re-align to pt1 & pt2
		//					//pt1 = pt2;
		//					pt2 = pt4;
		//					//cout << "Re-Aligning to pt1 & pt2" << "\n";
		//					//cout << "\n";cout << "\n";
		//#pragma endregion
		//					MyLine(src, pt1, pt2);
		//					j++;
		//				}
		//				else
		//				{
		//				}
		//			}
		//
		//		}
		//		else
		//		{
		//			pt1.x = 0;
		//			pt1.y = 0;
		//			pt2.x = 0;
		//			pt2.y = 0;
		//		}
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		/*system("cls");*/

		cout << "RIGHT" << endl;
		//Plot();
		//values.push_back(numbers);
		//for (int i = 0; i < values.size(); i++)
		//{
		//	cout << "RIGHT - points (pt1, pt2)= (" << pt1 << ", " << pt2 << ")" << endl;

		//	for (int j = 0; j < values[i].size(); j++)
		//	{
		//		cout << values[i][j];
		//		//cout << vec[i][j];
		//	}
		//	
		//	cout << "-------------------------------------------------------------------------------------------------";
		//	cout << "\n";
		//	cout << "\n";
		//}
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		//cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		circle(src_gray, Point(x,y), 1, (0, 0, 255), 5, 8, 0);

		if (clickCounter == 2)
		{
			if (pt1.y == y)
			{
				circle(src, pt2, 1, (0, 0, 255), 5, 8, 0);
			}
		}
		//cout << "x= " << x << " ,y= " << y << "\n";
		//circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
		//imshow("Cont", src);
	}
}

int main()
{
	src = imread("20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	Mat testSrc = src;
	//Plotting graph with Shervin Emami library
	//uchar *dataMat = src.data;
	//showUCharGraph("Pixel Values", dataMat, src.size().height*src.size().width);

	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	Mat tempSrcGray = src_gray;
	imshow("grayscale image", src_gray);

	cout << "Click on image:" << "\n";
	cout << "Specify two points on image - denoting line width:" << "\n";
	setMouseCallback("grayscale image", CallBackFunc, NULL);
	char key = (char)waitKey(0);   // explicit cast
	if (key >= 0)
	{
		cout << "Keypressed= " << key << "\n";
		waitKey(9999);
	}
	else
	{
		waitKey(0);
	}

	//int keyPressed = waitKey(0);
	
	//return 0;
}