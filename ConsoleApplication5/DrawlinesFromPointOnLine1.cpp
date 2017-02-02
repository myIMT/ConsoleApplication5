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
					int mag = sqrt(v.x*v.x + v.y*v.y);
					v.x = v.x / mag;
					v.y = v.y / mag;
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
					int mag = sqrt(v.x*v.x + v.y*v.y);
					v.x = v.x / mag;
					v.y = v.y / mag;
					cout << "Normalizing v" << "\n";
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
		cout << "x= " << x << " ,y= " << y << "\n";
		circle(src, Point(x, y), 1, (0, 0, 255), 5, 8, 0);
		imshow("Plot Dot on SRC Image", src);
	}
}

int main()
{
	src = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	Mat testSrc = src;
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
		ofstream connectedComponentsWithStats_MatrixFile;
		connectedComponentsWithStats_MatrixFile.open("connectedComponentsWithStats_" + nFltrLabelsString + "_MatrixFile.txt");

		for (int FltrLabel = 1; FltrLabel < 2/*nFltrLabels*/; ++FltrLabel)
		{
			FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
			
			connectedComponentsWithStats_MatrixFile << "Component No. " << FltrLabel << std::endl;
			connectedComponentsWithStats_MatrixFile << "CC_STAT_LEFT -- The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
			connectedComponentsWithStats_MatrixFile << "CC_STAT_TOP -- The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
			connectedComponentsWithStats_MatrixFile << "CC_STAT_WIDTH --  The horizontal size of the bounding box= " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
			connectedComponentsWithStats_MatrixFile << "CC_STAT_HEIGHT -- The vertical size of the bounding box. = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
			connectedComponentsWithStats_MatrixFile << "CC_STAT_AREA -- The total area (in pixels) of the connected component.  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
			connectedComponentsWithStats_MatrixFile << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;

			//circle(src, Point(111, 10), 1, (0, 0, 255), 50, 8, 0);
			//imshow("Plot Dot on SRC Image", src);
			std::string s = std::to_string(FltrLabel);
			// Get the mask for the i-th contour
			if (FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) >= 300)
			{
				Mat mask2_i = FltrLabelImage == FltrLabel;
				imwrite("mask2_i_" + s + ".bmp", mask2_i);
			}
			Mat mask_i = FltrLabelImage == FltrLabel;
			string name = "mask_i_";

#pragma region centroid_of_mask
			float sumx = 0, sumy = 0;
			float num_pixel = 0;
			for (size_t x = 0; x < mask_i.cols; x++)
			{
				for (size_t y = 0; y < mask_i.rows; y++)
				{
					int val = mask_i.at<uchar>(y, x);
					if (val >= 255) {
						sumx += x;
						sumy += y;
						num_pixel++;
					}
				}
			}
			Point p(sumx / num_pixel, sumy / num_pixel);
			cout << Mat(p) << endl;
#pragma endregion


			imwrite("mask_i_" + s + ".bmp", mask_i);
			Mat Points;

			double rectSize_b, rectSize_a;
			Point maskCentroid;

			//vector<Point> pts;
			findNonZero(mask_i, Points);
			//ofstream NonZeroMaskCoordinates_MatrixFile;
			//NonZeroMaskCoordinates_MatrixFile.open("NonZeroMaskCoordinates_MatrixFile.txt");

			//ofstream NonZeroMask_MatrixFile;
			//NonZeroMask_MatrixFile.open("NonZeroMask_MatrixFile.txt");
			//NonZeroMask_MatrixFile << Points<< "\n";
			//NonZeroMask_MatrixFile.close();
			//NonZeroMaskCoordinates_MatrixFile << "Points width= "<<Points.size().width << "\n";
			//NonZeroMaskCoordinates_MatrixFile << "Point height"<<Points.size().height << "\n";

			//for (size_t i = 0; i < Points.rows; i++)
			//{
			//	cout << Points.at<Point>(i).y << "\n";
			//	cout << Points.at<Point>(i).x << "\n";
			//	cout << "\n";

			//}

			//NonZeroMaskCoordinates_MatrixFile.close();

			RotatedRect box = minAreaRect(Points);
			
			Point2f vtx[4];
			box.points(vtx);			
			//box.angle returns angles in degrees
			connectedComponentsWithStats_MatrixFile << "minAreaRect Angle= " << box.angle +180<< "\n";

			maskCentroid = Point(FltrCentroids.at<double>(FltrLabel, 0),FltrCentroids.at<double>(FltrLabel, 1));

			// Draw the bounding box
			Mat tempSrc1 = imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED);;
			vector<double> lengths(4);
			for (int i = 0; i < 4; i++)
			{
				line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
				//rectSize_b = max();
				lengths.push_back(norm((vtx[(i + 1) % 4]) - (vtx[i])));
				cout << "component= " << FltrLabel << " -- " << "Line length= " << norm((vtx[(i + 1) % 4]) - (vtx[i])) << "\n";
				if (FltrLabel == 1)
				{
					/*				cout << "vtx[i]= " << vtx[i] << "\n";
					cout << "vtx[(i + 1) % 4]= " << vtx[(i + 1) % 4] << "\n";
					cout << "vtx[(i + 1)].y - vtx[i].y= " << vtx[(i + 1)].y - vtx[i].y << "\n";
					cout << "vtx[(i + 1)].x - vtx[i].x= " << vtx[(i + 1)].x - vtx[i].x << "\n";*/
					double lineAngle = atan2(vtx[(i + 1)].y - vtx[i].y, vtx[(i + 1)].x - vtx[i].x) * 180.0 / CV_PI;
					//cout << "lineAngle= " << lineAngle << "\n";
					//cout << "abs(lineAngle)"<< abs(lineAngle)<<"\n";
					//cout << "(box.angle + 180) + 10= " << (box.angle + 180) + 10 << "\n";
					//cout << "box.angle + 180) - 10= " << (box.angle + 180) - 10 << "\n";
					if (abs(lineAngle) <= (box.angle + 180) + 10 && abs(lineAngle) >= (box.angle + 180) - 10)
					{
						//cout << "COORDINATE:"<< "\n";
						line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 0, 255), 5, LINE_AA);
					}
					/*			cout << "COORDINATE: vtx[i].y= " << vtx[i].y << "\n";
					cout << "\n" << std::endl;*/
				}
			}
			rectSize_b = *max_element(lengths.begin(), lengths.end());
			//cout << "Max value: " << rectSize_b << endl;

			Point2d u;
			//Point2d uu;
			u = Point2d(cos(box.angle + 180), sin(box.angle + 180));
			Point2d w; //= u-((CV_PI/2)*(180/CV_PI));
			//rotate and swap
	/*		double tempX = -u.x;
			u.x = u.y;
			u.y = tempX;*/
			w = Point2d(u.y, -u.x);
			//int length = norm()
			vector<Point2d> Oxy(20);
			double d = 0.1*rectSize_b;

			//Point2f a(0.3f, 0.f), b(0.f, 0.4f);
			//Point pt = (a + b)*10.f;
			//cout << pt.x << ", " << pt.y << endl;

			Point2d centroid = maskCentroid;
			Point t = centroid;			
			circle(mask_i, t, 1, (0, 0, 0), 0.25, 8, 0);
			imshow("Plot Dot on SRC Image", src);
			for (size_t j = 0; j < 10; j++)
			{
				Point2d ww;
				double magW = sqrt(w.x*w.x + w.y*w.y);
				w.x = w.x/magW;
				w.y = w.y/magW;
				Point2d temp = centroid +(d*w);
				Point temp2 = temp;
				for (size_t k = 0; k < Points.rows; k++)
				{
					// << Points.at<Point>(k).y << "\n";
					//cout << Points.at<Point>(i).x << "\n";
					//cout << "\n";
					if ((temp2.x == Points.at<Point>(k).x) && (temp2.y == Points.at<Point>(k).y))
					{
						cout << "test: " << "x= "<<temp2.x<<"y= "<<temp2.y<<"\n";
					}
				}

				cout << "Ray origin= " << temp << "\n";
				circle(src, temp2, 1, (0, 0, 0), 0.125, 8, 0);
				imshow("Plot Dot on SRC Image", mask_i);
				Oxy.push_back(temp);
				centroid = temp;
			}
			imwrite("PlotDotonSRCImage.png", mask_i);
			
#pragma region quadrants
			//v.x = B.x - A.x; v.y = B.y - A.y;
			if ((box.angle + 180) < 90.0)
			{

			}
			else if ((box.angle + 180) >= 90.0 && (box.angle + 180) < 180.0)
			{

			}
			else if ((box.angle + 180) >= 180.0 && (box.angle + 180) < 270.0)
			{

			}
			else if ((box.angle + 180) >= 270.0 && (box.angle + 180) < 360.0)
			{

			}
#pragma endregion
			Point tempPoint;
			tempPoint.x = (int)round(length * cos((box.angle + 180) * CV_PI / 180.0));
			tempPoint.y = (int)round(length * sin((box.angle + 180) * CV_PI / 180.0));
			//cout << "tempPoint= " << tempPoint << "\n";

			//cout << "minAreaRect Angle= " << box.angle + 180 << "\n";
			connectedComponentsWithStats_MatrixFile << "\n" << std::endl;
			for (size_t i = 0; i < 4; i++)
			{
				/*cout << "i= " << i << " -- " << vtx[i] << "\n";
				cout << "vtx[(i + 1) % 4]= " << vtx[(i + 1) % 4] << "\n";
				cout << "\n";*/
			}

			//imwrite("boundingBox_" +s+".bmp", tempSrc1);
			//imshow("boundingBox_" + s, tempSrc1);
		}
		connectedComponentsWithStats_MatrixFile.close();
	waitKey(0);
	return 0;
}