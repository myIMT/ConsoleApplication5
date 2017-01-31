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
//#include <opencv2/legacy/compat.hpp>
using namespace cv;
using namespace std;
using std::vector;

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
int clickCounter = 0, lineCounter = 0, pixelCounter = 0, drawCounter=0;
std::vector<int> allObjPixelsCount;
//std::vector<double> buf;
vector<Point> points1;
vector<Point> points2;

double angle_value_D = 45;
int length = 10;
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
//![variables]

void MyLine(Mat img, Point start, Point end)
{
	cout << "baseLine1= " << baseLine1 << ",  baseLine2= " << baseLine2 << "\n";

	if (clickCounter ==/*1*/2)
	{
		clickCounter = 0;
	}

	cout << "Drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;

	int thickness = 2;
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

	imshow("Drawn Image", img);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	Mat newSrc = src;

	if (event == EVENT_LBUTTONDOWN)
	{
		system("cls");
		clickCounter++;
		drawCounter++;

		if (clickCounter == 1)
		{
			double theta = clickCounter *10;
			int length = 150;
			pt1.x = x;
			pt1.y = y;
			baseLine1.x = pt1.x;
			baseLine1.y = pt1.y;

			/////Draw line based pnone point (pt1, angle(theta) and length of line
			//pt2.x = (int)round(x + length * cos(theta * CV_PI / 180.0));
			//pt2.y = (int)round(y + length * sin(theta * CV_PI / 180.0));
			//MyLine(src, pt1, pt2);

			drawCounter = 0;
		}
		else if (clickCounter == 2)
		{

/*
			int tempX = pt2.x - pt1.x;
			int tempY = pt2.y - pt1.y;
			if (tempX <tempY)
			{
				pt2.x = pt1.x;
				pt2.y = y;
			}
			else if (tempX >tempY)
			{
				pt2.x = x;
				pt2.y = pt1.y;
			}
			else
			{
				pt2.x = x;
				pt2.y = y;
			}
				pt2.x = x;
				pt2.y = y;
			cout << "clickCounter= " << clickCounter << "\n";
			pt2.x = x;
			pt2.y = y;
			cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;*/

#pragma region Perpendicular Line
			//	////Direction vector going from 'start' to 'end'
			//	//Point v;
			//	//v.x = pt2.x - pt1.x;
			//	//v.y = pt2.y - pt1.y;

			//	////normalize v
			//	//int mag = sqrt(v.x*v.x + v.y*v.y);
			//	//v.x = v.x / mag;
			//	//v.y = v.y / mag;

			//	////rotate and swap
			//	//int tempX = -v.x;
			//	//v.x = v.y;
			//	//v.y = tempX;

			//	////new line at 'end' pointing in v-direction
			//	//Point pt3;
			//	//pt3.x = pt2.x + v.x * length;
			//	//pt3.y = pt2.y + v.y * length;
			//	//pt4.x = pt2.x + v.x * (-length);
			//	//pt4.y = pt2.y + v.y * (-length);  
#pragma endregion
			pt2.x = x;
			pt2.y = y;
			baseLine2.x = pt2.x;
			baseLine2.y = pt2.y;
			MyLine(src, pt1, pt2);

			//To pick up pixels under line
			cv::LineIterator it(src_gray, pt1, pt2, 8);
			//To store pixel picked up, from under line
			std::vector<double> buf(it.count);

			LineIterator it2 = it;

			for (int i = 0; i < it.count; i++, ++it)
			{
				double val = (double)src_gray.at<uchar>(it.pos());				
				buf[i] = val;
			}
			ofstream file;
			file.open("buf.csv");
			file << Mat(buf) << "\n";
			file.close();


			int j = 0;
			for (size_t i = 0; i < it.count; i++, ++it2)
			{	// For the very 1st line at baseline's 1st coordinate
				if (i == 0)
				{
					cout << "1st LINE ON BASELINE:" << "\n";
#pragma region Line Length=4
					j++;
					pt1 = it2.pos();
					pt2 = Point(it2.pos().x + 100, it2.pos().y + 100);
					cout << "point from LineIterator= " << pt1 << "\n";
#pragma endregion

#pragma region Plot Orthogonal Lines
					//Direction vector going from 'start' to 'end'
					Point v;
					v.x = baseLine2.x - baseLine1.x;
					v.y = baseLine2.y - baseLine1.y;
					//normalize v
			/*		int mag = sqrt(v.x*v.x + v.y*v.y);
					v.x = v.x / mag;
					v.y = v.y / mag;*/
					//rotate and swap
					int tempX = -v.x;
					v.x = v.y;
					v.y = tempX;
					//new line at 'end' pointing in v-direction
					Point pt3;
					pt3.x = pt1.x + v.x * 4;
					pt3.y = pt1.y + v.y * 4;
					////pt4.x = pt1.x + v.x * (-length);
					////pt4.y = pt1.y + v.y * (-length);  
					////re-align to pt1 & pt2
					////pt1 = pt2;
					pt2 = pt3;
#pragma endregion
					//cout << "From pt1= " << pt1 << "  to  " << pt2 << "   (pt1 should be equal to baseline2)" << "\n";
					MyLine(src, pt1, pt2);

				}
				else if (i == 20 * j)
				{
					pt1 = it2.pos();
					pt2 = Point(it2.pos().x + 100, it2.pos().y + 100);
					cout << "point from LineIterator= " << pt1 << "\n";

#pragma region Plot Orthogonal Lines
					////Direction vector going from 'start' to 'end'
					Point v;
					v.x = pt1.x - baseLine1.x;
					v.y = pt1.y - baseLine1.y;
					cout << "Direction vector going from 'start'-"<<pt1<<" to 'end'-"<<pt2 << "  = "<<v.x<<", "<<v.y<<"\n";
					//normalize v
					//int mag = sqrt(v.x*v.x + v.y*v.y);
					//v.x = v.x / mag;
					//v.y = v.y / mag;
					//cout << "Normalizing v" << "\n";
					//rotate and swap
					int tempX = -v.x;
					v.x = v.y;
					v.y = tempX;
					cout << "Rotating & swapping v" << "\n";
					//new line at 'end' pointing in v-direction
					pt4.x = pt1.x + v.x * (length);
					pt4.y = pt1.y + v.y * (length);  
					cout << "New line:" << "\n";
					cout << "at" <<pt2<<" and "<< pt4<< "\n";
					//re-align to pt1 & pt2
					//pt1 = pt2;
					pt2 = pt4;
					cout << "Re-Aligning to pt1 & pt2" << "\n";
					cout << "\n";cout << "\n";
#pragma endregion
					MyLine(src, pt1, pt2);
					j++;
				}
				else
				{
				}
			}

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
		if (clickCounter == 1)
		{

		}
	}
}

int main()
{
	src = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'

	//Plotting graph with Shervin Emami library
	uchar *dataMat = src.data;
	showUCharGraph("Pixel Values", dataMat, src.size().height*src.size().width);

	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	Mat tempSrcGray = src_gray;
	imshow("grayscale image", src_gray);
	setMouseCallback("grayscale image", CallBackFunc, NULL);
	cv::Mat FltrBinaryImg = threshval < 128 ? (src_gray < threshval) : (src_gray > threshval);

	Mat FltrLabelImage;
	Mat FltrStats, FltrCentroids;
	int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
	string nFltrLabelsString = std::to_string(nFltrLabels);
		std::vector<cv::Vec3b> FltrColors(nFltrLabels);
		FltrColors[0] = cv::Vec3b(0, 0, 0);

		for (int FltrLabel = 1; FltrLabel < 2/*nFltrLabels*/; ++FltrLabel)
		{
			FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
			// Get the mask for the i-th contour
			Mat mask_i = FltrLabelImage == FltrLabel;
			string name = "mask_i_";
			std::string s = std::to_string(FltrLabel);
			//imwrite("mask_i_" + s + ".bmp", mask_i);
			Mat Points;
			findNonZero(mask_i, Points);
			//Rect Min_Rect = boundingRect(Points);
			//rectangle(mask_i, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			/*rectangle(mask_i, Min_Rect.tl(), Min_Rect.br(), Scalar(0, 0, 255), 2);*/
			// Find the minimum area enclosing bounding box
			RotatedRect box = minAreaRect(Points);
			Point2f vtx[4];

			box.points(vtx);			
			cout << "minAreaRect Angle= " << box.angle +180<< "\n";
			for (size_t i = 0; i < 4; i++)
			{
				cout << "i= " << i << " -- " << vtx[i] << "\n";
				cout << "vtx[(i + 1) % 4]= " << vtx[(i + 1) % 4] << "\n";
				cout << "\n";
			}
			// Draw the bounding box
			for (int i = 0; i < 4; i++)
				line(src, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);

			imshow("Result", src);
		}

	waitKey(0);
	return 0;
}