#include <iostream>
#include "opencv2/core/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <string> 
#include <array>
//#include <opencv3_0_0/plot.hpp>
#include <numeric> 
#include "GraphUtils.h"
#include<conio.h>
#include<iomanip>

using std::vector;

using namespace cv;
using namespace std;

#pragma region Global Variables
const std::string keys =
"{help      |             | print this message    }"
"{@image    |contours     | load image            }"
"{j         |j.png        | j image        }"
"{contours  |contours.png | contours image        }"
;

int threshval = 60;
int bw_constant = 128;
vector<Vec4i> hierarchy;
Mat src, srcImg, srcImg1, srcImg2, srcImg3, srcImg4, GrayImg, hist, cannyEdge, detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;

int ddepth = CV_32FC1;// CV_16S;
int scale = 1;
int delta = 0;
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
int channels[] = { 0 };
int binID;

vector<Mat> Points;
//struct masks {
//	Mat mask;
//	Point maskCentroid;
//};
//vector<masks> maskImages;
vector<Mat> maskImages;
vector<Point> maskCentroid;

int secondBiggestArea = 0;
int secondBiggestAreaIndex = -1;
int thirdBiggestArea = 0;
int thirdBiggestAreaIndex = -1;
int secondBiggestAreaX;
int secondBiggestAreaY;
int secondBiggestAreaWidth;
int secondBiggestAreaHeight;
int thirdBiggestAreaX;
int thirdBiggestAreaY;
int thirdBiggestAreaWidth;
int thirdBiggestAreaHeight;

struct connectedComponentsWithStatsStruct {
	int COMP_STRUCT_NR;
	int CC_STAT_LEFT;
	int CC_STAT_TOP;
	int CC_STAT_WIDTH;
	int CC_STAT_HEIGHT;
	int CC_STAT_AREA;
	int CC_STAT_CENTROID_X;
	int CC_STAT_CENTROID_Y;
};
vector<connectedComponentsWithStatsStruct> CompStats;
int CompStructCount = 0;
int profileLinesCount = 0;

bool cout_output = false;
bool file_output = false;
bool imshow_output = false;
bool imwrite_output = false;
bool all_output = false;

Mat secondBiggestAreaMat, thirdBiggestAreaMat, nadirMat;
int nFltrLabels2;
Mat testMaskiRead;
Mat GrayScaleCroppedImage;
Mat cropedLeftImage;
Mat cropedRightImage;

Point pt1, pt2;
Mat src_gray;
int clickCounter = 0, lineCounter = 0, pixelCounter = 0, drawCounter = 0;

//vector<int> numbers;
vector<vector<int>> values;
bool distanceSet = false;
Point distancePoint1, distancePoint2, distancePoint3, heigthPoint;
double distancePoint;

vector<int> leftNadir;
vector<int> rightNadir;
vector<int> leftNadirPlusExtra;
vector<int> rightNadirPlusExtra;

Mat LeftImage;
Mat RightImage;
vector<Mat> LeftAndRightImages;
vector<int> MasksCount;

int countComponentsNotTouchingImageEdge;

vector<int> componentAngles, LcomponentAngles, RcomponentAngles;
////----------------------------------------------------  
#pragma endregion


template <typename T>
cv::Mat plotGraph(vector<T> & vals, int YRange[2])
{
	//vector<vector<T> > multilevel(4);

	//for (int j = 0; j < vals.size; j++)
	//{
	//	auto it = minmax_element(vals.begin(), vals.end());
	//	float scale = 1. / ceil(*it.second - *it.first);
	//	float bias = *it.first;
	//}
	
	
		auto it = minmax_element(vals.begin(), vals.end());
		float scale = 1. / ceil(*it.second - *it.first);
		float bias = *it.first;
	int rows = YRange[1] - YRange[0] + 1;
	cv::Mat image = Mat::zeros(rows, vals.size(), CV_8UC3);
	image.setTo(0);
	for (int i = 0; i < (int)vals.size() - 1; i++)
	{
		cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i + 1, rows - 1 - (vals[i + 1] - bias)*scale*YRange[1]), Scalar(255, 0, 0), 1);
		//cv::line(image, cv::Point(i, rows - 1 - (vals[i] - 1.23)*scale*YRange[1]), cv::Point(i + 1, rows - 1 - (vals[i + 1] - 1.23)*scale*YRange[1]), Scalar(0, 0, 255), 1);
	}

	return image;
}

void PickupPixels()
{
	//To pick up pixels under line
	cv::LineIterator it(src, distancePoint1, distancePoint2, 8);
	//To store pixel picked up, from under line
	std::vector<double> buf(it.count);
	Mat temp = src;
	LineIterator it2 = it;

	vector<int> numbers(it.count);

	for (int i = 0; i < it.count; i++, ++it)
	{
		double val = (double)src.at<uchar>(it.pos());
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




	int range[2] = { 0, it.count };
	showIntGraph("Rotation Angle", &numbers[0], numbers.size());

	//#pragma region Multiple Plots
	//	IplImage *graphImg = drawIntGraph(&floatVec1[0], floatVec1.size(), NULL,
	//		-25, 25, 400, 180, "X Angle (blue is truth, green is POSIT)");
	//	drawFloatGraph(&floatVec2[0], floatVec2.size(), graphImg, -25, 25, 400, 180);
	//	cvSaveImage("my_graph.jpg", graphImg);
	//	cvReleaseImage(&graphImg);
	//#pragma endregion

	cv::Mat lineGraph = plotGraph(numbers, range);

	imshow("plot", lineGraph);
#pragma endregion
}

void MyLine(Mat img, Point start, Point end)
{
	//cout << "baseLine1= " << baseLine1 << ",  baseLine2= " << baseLine2 << "\n";
	//system("cls");

	if (clickCounter ==/*1*/2)
	{
		clickCounter = 0;
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
				cout << "Point 1: (" << x << ", " << y << ")" << "\n";

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
				MyLine(src, distancePoint1, Point(distancePoint2.x, distancePoint1.y));
				cout << "Now, in relation one of existing points, click one more time to specify where line-drawing should end" << "\n";

				//distanceSet = true;

			/*	clickCounter = 0;
				drawCounter = 0;*/
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
		//else if (clickCounter == 3)
		//{
		//	if (distanceSet)
		//	{
		//		cout << "Point 3: (" << x << ", " << y << ")" << "\n";

		//		heigthPoint.y = y;
		//		heigthPoint.x = distancePoint1.x;
		//		circle(src, heigthPoint, 1, (0, 0, 255), 5, 8, 0);
		//		cout << "heigthPoint: " << heigthPoint << "\n";
		//		for (int m = distancePoint1.y; m < heigthPoint.y; m += 20)
		//		{
		//			//int yn = distancePoint1.y + m;
		//			MyLine(src, Point(distancePoint1.x, m), Point(distancePoint2.x, m));
		//		}
		//	}
		//	clickCounter = 0;
		//	distanceSet = false;
		//	cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << "\n";

		//}
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
		circle(src_gray, Point(x, y), 1, (0, 0, 255), 5, 8, 0);

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

//void ExtractNadirAreas(vector<int> a)
void ExtractNadirAreas(Mat SourceImage, int leftColNo, int rightColNo)
{
	if (all_output) { imshow("SourceImage", SourceImage); };

	int x0 = 0;
	int y0 = 0;
	int x1 = leftColNo; 
	int y1 = SourceImage.size().height;;
	int x2 = rightColNo;
	int y2 = y1;
	int x3 = SourceImage.size().width;
	int y3 = y1;

	Rect myLeftROI(x0, y0, x1, SourceImage.size().height);
	Rect myRightROI(x2, y0, SourceImage.size().width - rightColNo, SourceImage.size().height);

	Mat croppedLeftImage = SourceImage(myLeftROI);
	Mat croppedRightImage = SourceImage(myRightROI);

	if (all_output) {imshow("croppedLeftImage", croppedLeftImage);};
	if (all_output) {imshow("croppedRightImage", croppedRightImage);};

	LeftImage = croppedLeftImage;
	RightImage = croppedRightImage;
	LeftAndRightImages.push_back(LeftImage);
	LeftAndRightImages.push_back(RightImage);


	////if (all_output){ cout << "size of a = " << a.size() << endl;};
	//sort(a.begin(), a.end());	/// Sort vector of components-areas
	//reverse(a.begin(), a.end());	/// Reverse sorted vector of components-areas
	////if (all_output){ cout << "a-vector-sorted with +4= " << endl;};	/// Print sorted-reversed component-areas
	//for (size_t i = 0; i != a.size(); ++i)
	//	//if (all_output){ cout << a[i] << " ";};
	////if (all_output){ cout << "\n"; //if (all_output){ cout << "\n"; //if (all_output){ cout << "\n";}
	//secondBiggestArea = a[1];	//if (all_output){ cout << "second biggest = " << a[1] << "\n";}
	////secondBiggestAreaIndex = a.size()-1;	//if (all_output){ cout << "second biggest Index = " << a.size() - 1 << "\n";}
	//thirdBiggestArea = a[2];	//if (all_output){ cout << "third biggest = " << a[2] << "\n";}
	////thirdBiggestAreaIndex = a.size() - 2;	//if (all_output){ cout << "third biggest Index = " << a.size() - 2 << "\n";}
}

/// <summary>                                                            
/// Calculates components                           
/// </summary>                                                           
/// <param name="GrayScaleSrcImg">Grayscale image (Mat) of original image.</param> 
Mat GetConnectedComponent(vector<Mat> myLeftAndRightImages)
{
	Mat GrayScaleSrcImg, resultantImage;
	for (int mats = 0; mats < myLeftAndRightImages.size(); mats++)
	{
		std::string smats = std::to_string(mats);

		if (all_output) { imshow("myLeftAndRightImages[" + smats + "]", myLeftAndRightImages.at(mats)); };

		GrayScaleSrcImg = myLeftAndRightImages.at(mats);
		if (!all_output) { cout << "03 - CALCULATING COMPONENTS ON BLURRED GRAYSCALE SOURCE IMAGE ('GrayImg')" << "\n"; };

		cv::Mat FltrBinaryImg = threshval < 128 ? (GrayScaleSrcImg < threshval) : (GrayScaleSrcImg > threshval);
		if (!all_output) { cout << "04 - CONVERTING BLURRED GRAYSCALE SOURCE IMAGE TO BINARY: 'FltrBinaryImg'" << "\n"; };

		cv::Mat FltrLabelImage;
		cv::Mat FltrStats, FltrCentroids;

		int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
		if (!all_output) { cout << "05 - CALCULATING CONNECTEDCOMPONENTSWITHSTATS ON 'FltrBinaryImg': 'FltrLabelImage'" << "\n"; };
#pragma region connectedComponentsWithStats
		ofstream connectedComponentsWithStats;
		connectedComponentsWithStats.open("connectedComponentsWithStats.txt");
		if (!all_output) { cout << "06 - WRITING CONNECTEDCOMPONENTSWITHSTATS RESULTS TO FILE: 'connectedComponentsWithStats.txt'" << "\n"; };
		//if (all_output){ imshow("Labels", FltrLabelImage);
		//connectedComponentsWithStats << "nFltrLabels= " << nFltrLabels << std::endl;
		//connectedComponentsWithStats << "size of original image= " << FltrBinaryImg.size() << std::endl;
		//connectedComponentsWithStats << "size of FltrLabelImage= " << FltrLabelImage.size() << std::endl;
		if (all_output) { imshow("FltrLabelImage2", FltrLabelImage); }
		std::vector<cv::Vec3b> FltrColors(nFltrLabels);
		FltrColors[0] = cv::Vec3b(0, 0, 0);
		//connectedComponentsWithStats << "(Filter) Number of connected components = " << nFltrLabels << std::endl << std::endl;
		//vector<vector<Point>> contours;
		//vector<Vec4i> hierarchy;
		vector<int> a;

		for (int FltrLabel = 0; FltrLabel < nFltrLabels; ++FltrLabel) {
			FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
			CompStats.push_back(connectedComponentsWithStatsStruct());												/*Initialise container*/

			connectedComponentsWithStats << "Component " << FltrLabel << std::endl;
			CompStats[CompStructCount].COMP_STRUCT_NR = FltrLabel;

			connectedComponentsWithStats << "CC_STAT_LEFT   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
			CompStats[CompStructCount].CC_STAT_LEFT = FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT);

			connectedComponentsWithStats << "CC_STAT_TOP    = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
			CompStats[CompStructCount].CC_STAT_TOP = FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP);

			connectedComponentsWithStats << "CC_STAT_WIDTH  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
			CompStats[CompStructCount].CC_STAT_WIDTH = FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH);

			connectedComponentsWithStats << "CC_STAT_HEIGHT = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
			CompStats[CompStructCount].CC_STAT_HEIGHT = FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT);

			connectedComponentsWithStats << "CC_STAT_AREA   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
			CompStats[CompStructCount].CC_STAT_AREA = FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA);

			connectedComponentsWithStats << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;
			CompStats[CompStructCount].CC_STAT_CENTROID_X = FltrCentroids.at<double>(FltrLabel, 0);
			CompStats[CompStructCount].CC_STAT_CENTROID_Y = FltrCentroids.at<double>(FltrLabel, 1);

			a.push_back(FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA));

			CompStructCount++;
		}
		if (!all_output) { cout << "07 - STORING EACH COMPONENT'S AREA IN (GLOBAL DECLARED) VECTOR: 'a'" << "\n"; };
#pragma endregion
		connectedComponentsWithStats.close();

		//#pragma region RemoveNadir
		//	ExtractNadirAreas(a);
		//	/*Mat secondBiggestAreaMat, thirdBiggestAreaMat, nadirMat;*/
		//	for (int FltrLabel = 0; FltrLabel < nFltrLabels; ++FltrLabel) {
		//		if (FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) == secondBiggestArea)
		//		{
		//			secondBiggestAreaIndex = FltrLabel;
		//			secondBiggestAreaX = FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT);
		//			secondBiggestAreaY = FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP);
		//			secondBiggestAreaWidth = FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH);
		//			secondBiggestAreaHeight = FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT);
		//		}
		//
		//		if (FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) == thirdBiggestArea)
		//		{
		//			thirdBiggestAreaIndex = FltrLabel;
		//			thirdBiggestAreaX = FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT);
		//			thirdBiggestAreaY = FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP);
		//			thirdBiggestAreaWidth = FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH);
		//			thirdBiggestAreaHeight = FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT);
		//		}
		//	}
		//
		//	if (secondBiggestAreaX > thirdBiggestAreaX)
		//	{
		//		int tempIndex = secondBiggestAreaIndex;
		//		int tempX = secondBiggestAreaX;
		//		int tempY = secondBiggestAreaY;
		//		int tempWidth = secondBiggestAreaWidth;
		//		int tempHeight = secondBiggestAreaHeight;
		//
		//		secondBiggestAreaIndex = thirdBiggestAreaIndex;
		//		secondBiggestAreaX = thirdBiggestAreaX;
		//		secondBiggestAreaY = thirdBiggestAreaY;
		//		secondBiggestAreaWidth = thirdBiggestAreaWidth;
		//		secondBiggestAreaHeight = thirdBiggestAreaHeight;
		//
		//		thirdBiggestAreaIndex = tempIndex;
		//
		//		thirdBiggestAreaIndex = tempIndex;
		//		thirdBiggestAreaX = tempX;
		//		thirdBiggestAreaY = tempY;
		//		thirdBiggestAreaWidth = tempWidth;
		//		thirdBiggestAreaHeight = tempHeight;
		//	}
		//
		//	compare(FltrLabelImage, secondBiggestAreaIndex, nadirMat, CMP_EQ);
		//	//if (!all_output) { imshow("Remove Nadir (1)", nadirMat); }
		//	compare(FltrLabelImage, thirdBiggestAreaIndex, nadirMat, CMP_EQ);
		//	/*if (all_output){ imshow("nadirMat", nadirMat);*/
		//	//Rect roi(secondBiggestAreaX,secondBiggestAreaY, secondBiggestAreaWidth, secondBiggestAreaHeight);
		//	Mat CopyOfGrayScaleSrcImg = GrayScaleSrcImg;
		//	Mat LeftCopyOfGrayScaleSrcImg = GrayScaleSrcImg;
		//	Mat RightOfCopyGrayScaleSrcImg = GrayScaleSrcImg;
		//	//if (all_output){ imshow("Original Image", GrayScaleSrcImg); 
		//	//Rect rect(x, y, width, height)
		//	Rect myLeftROI(0, 0, secondBiggestAreaX, GrayScaleSrcImg.size().height);
		//	Rect myRightROI((thirdBiggestAreaX + thirdBiggestAreaWidth), 0, GrayScaleSrcImg.size().width - (thirdBiggestAreaX + thirdBiggestAreaWidth), GrayScaleSrcImg.size().height);
		//	//Mat image;
		//	//Mat croppedImage = image(myROI);
		//	cropedLeftImage = LeftCopyOfGrayScaleSrcImg(myLeftROI);
		//	cropedRightImage = RightOfCopyGrayScaleSrcImg(myRightROI);
		//	if (!all_output) { imshow("Remove Nadir (2)- cropedLeftImage", cropedLeftImage); }
		//	if (!all_output) { imshow("Remove Nadir (3) - cropedRightImage", cropedRightImage); }
		//	Mat croppedImage = Mat(cropedLeftImage.size().height, (cropedLeftImage.size().width + cropedRightImage.size().width), cropedRightImage.type(), Scalar(0, 0, 0));;
		//	//cv::Mat small_image;
		//	//cv::Mat big_image;
		//	//...
		//	//	//Somehow fill small_image and big_image with your data
		//	//	...
		//	//	small_image.copyTo(big_image(cv::Rect(x, y, small_image.cols, small_image.rows)));
		//	cropedLeftImage.copyTo(croppedImage(Rect(0, 0, cropedLeftImage.cols, cropedLeftImage.rows)));
		//	cropedRightImage.copyTo(croppedImage(Rect(cropedLeftImage.size().width, 0, cropedRightImage.cols, cropedRightImage.rows)));
		//	if (!all_output) { imshow("Remove Nadir (3) - croppedImage", croppedImage); }

		Mat croppedImage = GrayScaleSrcImg;
		//#pragma endregion

		imwrite("Grayscale_image_without_Nadir.bmp", croppedImage);
		src_gray = croppedImage;

		Mat FltrLabelImage2;
		Mat FltrStats2, FltrCentroids2;

		GrayScaleCroppedImage = croppedImage;

		//#pragma region Histogram Equilization
		//	char* source_window = "Source image";
		//	char* equalized_window = "Equalized Image";
		//	Mat dstHE;
		//	/// Convert to grayscale
		//	//cvtColor(src, src, CV_BGR2GRAY);
		//
		//	/// Apply Histogram Equalization
		//	equalizeHist(GrayScaleCroppedImage, dstHE);
		//
		//	/// Display results
		//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
		//	namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);
		//
		//	imshow(source_window, GrayScaleCroppedImage);
		//	imshow(equalized_window, dstHE);
		//#pragma endregion

		Mat FltrBinaryImg2 = threshval < 128 ? (GrayScaleCroppedImage < threshval) : (GrayScaleCroppedImage > threshval);
		imwrite("B&W_image_without_Nadir.bmp", FltrBinaryImg2);
		nFltrLabels2 = cv::connectedComponentsWithStats(FltrBinaryImg2, FltrLabelImage2, FltrStats2, FltrCentroids2, 8, CV_32S);
		MasksCount.push_back(nFltrLabels2);
		/*nFltrLabels2 = MasksCount.at(mats);*/

		if (!all_output) { cout << "08 - ATTEMPT TO READ SOURCE IMAGE AND SPLIT IT IN TWO HALVES" << "\n"; };
		if (!all_output) { cout << "09 -  -- ASSUMING TWO BLACK STRIPS ARE ALWAYS THE 2ND AND 3RD LARGEST COMPONENTS" << "\n"; };

		std::string nFltrLabels2String = std::to_string(nFltrLabels2);

		cv::Mat FltrLabelImage3;

		FltrLabelImage3 = FltrLabelImage2;
		//normalize(FltrLabelImage2, FltrLabelImage3, 0, 255, NORM_MINMAX, CV_8U);
		//if (all_output){ imshow("FltrLabelImage2", FltrLabelImage2);

		//std::vector<cv::Vec3b> FltrColors(nFltrLabels);
		//FltrColors[0] = cv::Vec3b(0, 0, 0);;
		if (!all_output) { cout << "10 - STORE ALL COMPONENTS'S IN (GLOBALLY DECLARED) VECTOR: 'maskImages'" << "\n"; };
		if (!all_output) { cout << "11 - STORE ALL COMPONENTS'S CENTROID IN (GLOBALLY DECLARED) VECTOR: 'maskCentroid'" << "\n"; };

		//AngleTest_DataFile.open("AngleTest_DataFile.txt", ios::app);
		//ofstream masksVector_DataFile;
		//masksVector_DataFile.open("masksVector_DataFile.txt", ios::app);
		std::vector<cv::Vec3b> FltrColors2(nFltrLabels2);
		FltrColors2[0] = cv::Vec3b(0, 0, 0);
		int tempCount;
		for (int FltrLabel2 = 1; FltrLabel2 < nFltrLabels2; ++FltrLabel2)
		{
			//if (MasksCount.size()>1)
			//{
			//	tempCount = MasksCount.at(0) + FltrLabel2;
			//}
			//else
			//{
			//	tempCount = FltrLabel2;
			//}

			std::string mask_index = std::to_string(FltrLabel2);
			FltrColors2[FltrLabel2] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
			//FltrColors[FltrLabel] = cv::Vec3b((255), (255), (255));
			Mat mask_i = FltrLabelImage3 == FltrLabel2;
			if (mask_i.empty())      // please, *always check* resource-loading.
			{
				cerr << "mask_i is empty - can't be loaded!" << endl;
				continue;
			}
			//maskImages.push_back(mask_i);
			imwrite("mask_" + smats + "_" + mask_index + ".bmp", mask_i);
			maskCentroid.push_back(Point(FltrCentroids.at<double>(FltrLabel2, 0), FltrCentroids2.at<double>(FltrLabel2, 1)));
		}
		//
		cv::Mat FltrDst2(croppedImage.size(), CV_8UC3);
		for (int r = 0; r < FltrDst2.rows; ++r) {
			for (int c = 0; c < FltrDst2.cols; ++c) {
				int FltrLabel2 = FltrLabelImage3.at<int>(r, c);
				cv::Vec3b &FltrPixel2 = FltrDst2.at<cv::Vec3b>(r, c);
				FltrPixel2 = FltrColors2[FltrLabel2];
			}
		}
		if (all_output) { imshow(nFltrLabels2String + "-Connected Components", FltrDst2); };
		if (all_output) { imwrite(mats + "-" + nFltrLabels2String + "-Connected Components.bmp", FltrDst2); };


		resultantImage = FltrDst2;
		if (!all_output) { cout << "12 - RETURNING IMAGE OF ALL COMPONENTS: 'FltrDst'" << "\n"; };
	}
	if (MasksCount.size() >1)
	{
		nFltrLabels2 = MasksCount.at(0) + MasksCount.at(1);
	}
	return resultantImage;
}

void SetNadirToBlack(Mat nadirRemoveImage)
{
	cout << "width of image= " << nadirRemoveImage.size().width << "\n";
	Mat nadirRemovedGray;
	//cvtColor(nadirRemoveImage, nadirRemovedGray, CV_BGR2GRAY);
	nadirRemovedGray = nadirRemoveImage;
#pragma region Test width even or odd
	if ((nadirRemovedGray.size().width & 1) == 0)
		printf("EVEN!\n");
	else
		printf("ODD!\n");
#pragma endregion

	int value = 12;
	int middle = nadirRemovedGray.cols / 2;
	Mat lrImage = nadirRemovedGray;
	Mat testImage;
#pragma region cycle through left-cols pixels
	for (int r = 0; r < lrImage.rows; r++)
	{
		for (int lr = middle; lr > 0; lr--)
		{
			/*		if (!all_output) { cout << "BEFORE -- (r, lr)= (" << r << "," << lr << ")" << "\n"; };
			if (!all_output) { cout << "BEFORE -- (double)lrImage.at<uchar>(r, lr)= " << (double)lrImage.at<uchar>(r, lr) << "\n"; };*/

			/*lrImage.at<uchar>(r, lr) = 0;*/
			//if (lr == (lr-5))
			//{
			//	lrImage.at<uchar>(r, lr) = 0;
			//}
			if ((double)lrImage.at<uchar>(r, lr) <= value)
			{
				/*	if (!all_output) { cout << "(r, lr)= (" << r << "," << lr << ")" << "\n"; };
				if (!all_output) { cout << "(double)lrImage.at<uchar>(r, lr)= " << (double)lrImage.at<uchar>(r, lr) << "\n"; };*/
				lrImage.at<uchar>(r, lr) = 0;
			}
			else
			{
				/*		if (!all_output) { cout << "NOT >= 128" << "\n"; };*/
				/*		if (!all_output) { cout << "AFTER -- (r, lr)= (" << r << "," << lr << ")" << "\n"; };
				if (!all_output) { cout << "AFTER -- (double)lrImage.at<uchar>(r, lr)= " << (double)lrImage.at<uchar>(r, lr) << "\n"; };*/
			}
		}

		for (int rr = middle; rr < lrImage.size().width; rr++)
		{
			if (lrImage.at<uchar>(r, rr) <= value)
			{
				lrImage.at<uchar>(r, rr) = 0;
			}
			else
			{

			}
		}


	}
	testImage = lrImage;
	if (all_output) { imshow("Nadir removed image", testImage); };
#pragma endregion
}

void ColumnsAnalysis(Mat SourceImage)
{
	/*Mat src = imread("salt.tif", 1);*/
	Mat filteredImage;

	//Apply median filter
	medianBlur(SourceImage, filteredImage, 15);
	if (all_output) {imshow("source", SourceImage);};
	if (all_output) { imwrite("GrayScaleImageColumns.bmp", SourceImage); };
	if (all_output) { imshow("filteredImage", filteredImage); };
	if (all_output) { imwrite("BlurredGrayScaleImageColumns.bmp", filteredImage); };
	ofstream columnMatData_DataFile;
	columnMatData_DataFile.open("columnMatData_DataFile.csv");
	columnMatData_DataFile << filteredImage << "\n";
	columnMatData_DataFile.close();

	struct columns {
		int col;
		int values;
	};
	vector<Mat> columnData;

	//columnData.push_back(columns);
	ofstream columnData_DataFile;
	columnData_DataFile.open("columnData_DataFile.csv");

	for (size_t i = 0; i < filteredImage.cols; i++)
	{
		//if (!all_output) { cout << "i= " << i << " -- value= " << filteredImage.col(i) << "\n"; };
		if (all_output) { columnData_DataFile << "i= " << i << " -- value= " << filteredImage.col(i) << "\n"; };
		columnData.push_back(filteredImage.col(i));
		//columnData.push_back(columns());												/*Initialise container*/
		//columnData[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);	/*Store bin (newAngle/5)*/
		//container[containerCount].i = i;											///Store row position
		//container[containerCount].j = j;											///Store column position
		//container[containerCount].angle = newAngle.ptr<float>(i)[j];				///Store new Angle value
		//container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);
	}	
	columnData_DataFile.close();

	struct columnSumStruct
	{
		int value;
		int columnNo;
	};
	vector<columnSumStruct> columnSum;

	vector<int> columnThresSum;
	int sum, thresSum;

	//
	ofstream columnThresSumData_DataFile;
	columnThresSumData_DataFile.open("columnThresSumData_DataFile.txt");

	ofstream columnSumData_DataFile;
	columnSumData_DataFile.open("columnSumData_DataFile.txt");

	for (size_t j = 0; j < columnData.size(); j++)
	{
		Mat tempCol = columnData.at(j);
		sum = 0;
		thresSum = 0;

		for (size_t k = 0; k < tempCol.size().height; k++)
		{
			//if (!all_output) { columnSumData_DataFile << "col= " << j << "-- value= " << sum << "\n"; };
			sum += tempCol.at<uchar>(k, 0);;

			if (tempCol.at<uchar>(k, 0) <=15)
			{
				thresSum += tempCol.at<uchar>(k, 0);
			}
		}
		columnSum.push_back(columnSumStruct());
		columnSum[j].value = sum;
		columnSum[j].columnNo = j;

		columnThresSum.push_back(thresSum);

		if (all_output) { columnThresSumData_DataFile << "col= " << j << "-- value= " << thresSum << "\n"; };
		if (all_output) { columnSumData_DataFile << "col= " << j << "-- value= " << sum << "\n"; };
	}
	columnThresSumData_DataFile.close();
	columnSumData_DataFile.close();

	//centre of image
	int c = columnSum.size() /2;
	//int h2 = 0;

	int leftWalkerZero = 2;
	bool firstLeftWalkerZero = false;
	int rightWalkerZero = 2;
	Mat nadirChangedImage = filteredImage;
	//vector<int> leftNadir;
	//vector<int> rightNadir;
	float percentageValue = 5;
	int zeroThreshold = 1000;
	int extraWidth = nadirChangedImage.size().width*percentageValue /100;
	int percentageWidth = 0;
	//vector<int> leftNadirPlusExtra;
	//vector<int> rightNadirPlusExtra;

#pragma region walk left
	//starting at the centre - walk left
	for (int h1 = c; h1 < columnSum.size(); h1--)
	{
		//if the pixel-intensity sum per column equals zero - record that column number 
		if (columnSum[h1].value <= zeroThreshold)
		{
			leftNadir.push_back(h1);
			firstLeftWalkerZero = true;
			leftWalkerZero = 1;
			/*if (!all_output) { cout << "h1 = " << h1 << " , walking left: SUM = " << columnSum[h1].value << "\n"; };*/
		}

		//if (h1 == leftNadir.back() - extraWidth)
		//{
		//	if (!all_output) { cout << "percentageWidth = " << c << "\n"; };
		//}

		//if (leftWalkerZero == 2 && firstLeftWalkerZero == true)
		//{
		//	leftWalkerZero = 1;
		//}
		//else if (leftWalkerZero == 2 && firstLeftWalkerZero == false)
		//{
		//	for (int m = 0; m < columnData.at(h1).rows; m++)
		//	{
		//		//if (!all_output) { cout << "columnData.at(h1).rows = " << columnData.at(h1).rows << "\n"; };
		//		//if (!all_output) { cout << "columnData.at(h1).cols = " << columnData.at(h1).cols << "\n"; };
		//		for (int o = 0; o < nadirChangedImage.size().width; o++)
		//		{
		//			for (int p = 0; p < nadirChangedImage.size().height; p++)
		//			{
		//				//if (h1 == )
		//				//{

		//				//}
		//				//nadirChangedImage.at<uchar>(, m) = 0;
		//			}
		//		}
		//		nadirChangedImage.at<uchar>(m, 0) = 0;
		//	}
		//}
		//else
		//{
		//	leftWalkerZero = 0;
		//}

		//if (columnSum[h1].value != 0)
		//{

		//}
	}


	//leftNadir = recording of column numbers for all column-pixel-intensity-zero-sums
	//extraWidth = 2 % of total width (to add to last column-pixel-intensity-zero-sum recording)
	//leftNadir.back = last column-pixel-intensity-zero-sum recording
	if (all_output) { cout << "leftNadir.back() - extraWidth = " << leftNadir.back() - extraWidth << "\n"; };
	if (all_output) { cout << "columnSum[leftNadir.back() - extraWidth] = " << columnSum[leftNadir.back() - extraWidth].value << "\n"; };

	//starting at last recording, add extraWidth steps
	for (int h2 = leftNadir.back()-1; h2 >= leftNadir.back() - extraWidth; h2--)
	{
		leftNadirPlusExtra.push_back(h2);
		if (all_output) { cout << "h2 = " << h2 << "\n"; };
		if (all_output) { cout << "columnSum[h2] = " << columnSum[h2].value << "\n"; };
	}
#pragma endregion

	//imshow("nadirChangeImage", nadirChangedImage);

#pragma region walk right
	//for (int h1 = c; h1 < columnSum.size(); h1++)

	//starting at the centre - walk right
	for (int h3 = c; h3 < columnSum.size(); h3++)
	{
		//if the pixel-intensity sum per column equals zero - record that column number 
		if (columnSum[h3].value <= zeroThreshold)
		{
			rightNadir.push_back(h3);
		}
	}


	//rightNadir = recording of column numbers for all column-pixel-intensity-zero-sums
	//extraWidth = 2 % of total width (to add to last column-pixel-intensity-zero-sum recording)
	//rightNadir.back = last column-pixel-intensity-zero-sum recording
	if (all_output) { cout << "rightNadir.back() - extraWidth = " << rightNadir.back() + extraWidth << "\n"; };
	if (all_output) { cout << "columnSum[rightNadir.back() - extraWidth] = " << columnSum[rightNadir.back() + extraWidth].value << "\n"; };

	//starting at last recording, add extraWidth steps
	for (int h4 = rightNadir.back() + 1; h4 <= rightNadir.back() + extraWidth; h4++)
	{
		rightNadirPlusExtra.push_back(h4);
	}
#pragma endregion

	//imshow("nadirChangeImage", nadirChangedImage);
	ExtractNadirAreas(SourceImage, leftNadirPlusExtra.back(), rightNadirPlusExtra.back());
}

void CalcHist(Mat histSrc)
{
	Mat gray;
	cvtColor(histSrc, gray, CV_BGR2GRAY);

	//Mat gray = imread("image.jpg", 0);
	namedWindow("Gray", 1);    
	if (all_output) { imshow("Gray", gray); };

	// Initialize parameters
	int histSize = 256;    // bin size
	cout << "histSize= " << histSize  << "\n";
	float range[] = { 0, 255 };
	cout << "range= " << range << "\n";
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
	cout << "histSize= " << histSize << "\n";

	// Show the calculated histogram in command window
	double total;
	total = gray.rows * gray.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		cout << " " << binVal;
	}

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);

		//rectangle(histImage, Point(i*bin_w, histImage.rows),
		//	Point((i + 1)*bin_w, histImage.rows - cvRound(hist.at<float>(i))),
		//	Scalar::all(0), -1, 8, 0);
	}

	namedWindow("Result", 1);    
	if (all_output) { imshow("Result", histImage); };

	waitKey(0);
}

void CalcHistEq()
{
	Mat dst;

	char* source_window = "Source image";
	char* equalized_window = "Equalized Image";

	if (!src.data)
	{
		cout << "Usage: ./Histogram_Demo <path_to_image>" << endl;
		return;
	}

	/// Convert to grayscale
	cvtColor(src, src, CV_BGR2GRAY);

	///Filter Image
	//bilateralFilter(src, src, 15, 80, 80);
	//blur(src, src, Size(5, 5), Point(-1, -1));
	//GaussianBlur(src, src, Size(5, 5), 5);

	/// Apply Histogram Equalization
	equalizeHist(src, dst);

	/// Display results
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);

	imshow(source_window, src);
	imshow(equalized_window, dst);

	SetNadirToBlack(dst);

	/// Wait until user exits the program
	waitKey(0);
}

bool ComponentTouchImageEdge(Mat CompMat)
{
	///Walk along cannyEdge rows
	for (size_t i = 0; i < CompMat.rows; i++)
	{
		// Walk along CompMat rows & columns
		for (size_t j = 0; j < CompMat.cols; j++)
		{
			// if cannyEdge pixel intensity is non-zero
			if ((int)CompMat.at<uchar>(i, j) != 0)
			{
				if ((i == 0) || (i == CompMat.size().width) || (j == 0) || (j ==CompMat.size().height-1))
				{
					return true;
				}
				else
				{
 					countComponentsNotTouchingImageEdge++;
					return false;
				}
			}
		}
	}
}

int findmax(int vals[], int n) {
	int max = vals[0];
	for (int i = 0; i<n; i++)
		if (max<vals[i]) max = vals[i];
	return max;
}

void Histogram(vector<int> myVector)
{
	int* vals = myVector.data();

	//int vals = a;
	int n = myVector.size();
	int fsize;
	//cout << "Enter number of data points:";
	//cin >> n;
	//int vals[5];
	for (int i = 0; i<n; i++) {
		cout << "Value" << i + 1 << ":" ;
		cout << vals[i] << "\n"; //intialize data
	}
	
	int m = findmax(vals, n); //find max value of data points
	if (m>n) fsize = m + 1;
	else fsize = n;
	//vector<int> myFreq;
	int* freq = new int[n];
	//int freq[fsize]; //declare frequency array with an appropriate size
					//The size of frequency array can be the size of the vals array
					//or the max value of the vals array items plus 1

	for (int i = 0; i<fsize; i++) //initialize frequency array
		freq[i] = 0;

	//compute frequencies
	for (int i = 0; i < n; i++)
		freq[vals[i]]++;

	//print histogram
	ofstream ComponentAnglesHistogramFile;
	ComponentAnglesHistogramFile.open("ComponentAnglesHistogramFile.txt");

	ComponentAnglesHistogramFile << "\n....Histogram....\n\n";
	for (int i = 0; i<fsize; i++) {
		if (freq[i] !=0)
		{
			ComponentAnglesHistogramFile << left;
			ComponentAnglesHistogramFile << setw(5) << i;
			ComponentAnglesHistogramFile << setw(5) << freq[i];
			for (int j = 1; j <= freq[i]; j++) ComponentAnglesHistogramFile << "*";
			ComponentAnglesHistogramFile << "\n";
		}
		else
		{

		}
	}
	ComponentAnglesHistogramFile.close();
}
//Mat RemoveNadir(Mat GrayScaleSrcImg)
//{
//	Mat BinaryNadirImage = threshval < 128 ? (GrayScaleSrcImg < threshval) : (GrayScaleSrcImg > threshval);
//
//	if (!all_output) { cout << "BinaryNadirImage.size().width= " << BinaryNadirImage.size().width << "\n"; };
//
//	double c;
//	double cc;
//
//	if (BinaryNadirImage.size().width % 2 == 0)
//	{
//		//is even
//		c = BinaryNadirImage.size().width / 2;
//		cc = c + 1;
//	}
//	else
//	{
//		double f = floor(BinaryNadirImage.size().width);
//		c = f + 1;
//		cc = 0;
//	}
//
//
//	double nadirCenterX = BinaryNadirImage.size().width / 2; 
//	Point nadirCenterPoint = Point(nadirCenterX, 1);
//	if (!all_output) { cout << "NadirCenter point= " << nadirCenterX << "\n"; };
//  if (!all_output) { imshow("Centre of Nadir", tempNadirSrc);}
//	Mat tempNadirSrc = imread(file);
//	circle(tempNadirSrc, nadirCenterPoint, 1, Scalar(255, 0, 0), 1, 8, 0);
//	imshow("Centre of Nadir", tempNadirSrc);
//
//	//for (size_t x = 0; j < BinaryNadirImage.cols; j++)
//	//{
//	//	/*				ComponentsLoop << "			Walk along Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";*/
//	//	//Bin_Analysis << "Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";
//	//	///if cannyEdge pixel intensity id non-zero
//	//	if ((int)cannyEdge.at<uchar>(i, j) != 0)
//	//	{
//
//	//	}
//	//}
//	return tempNadirSrc;
//}

#pragma region filters
// 1

/** Global Variables */
int alpha = 100;
int beta = 100;
int gamma_cor = 100;
Mat img_original, img_corrected, img_gamma_corrected;

void basicLinearTransform(const Mat &img, const double alpha_, const int beta_)
{
	Mat res;
	img.convertTo(res, -1, alpha_, beta_);

	hconcat(img, res, img_corrected);
}

void gammaCorrection(const Mat &img, const double gamma_)
{
	CV_Assert(gamma_ >= 0);
	//![changing-contrast-brightness-gamma-correction]
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

	Mat res = img.clone();
	LUT(img, lookUpTable, res);
	//![changing-contrast-brightness-gamma-correction]

	hconcat(img, res, img_gamma_corrected);
}

void on_linear_transform_alpha_trackbar(int, void *)
{
	double alpha_value = alpha / 100.0;
	int beta_value = beta - 100;
	basicLinearTransform(img_original, alpha_value, beta_value);
}

void on_linear_transform_beta_trackbar(int, void *)
{
	double alpha_value = alpha / 100.0;
	int beta_value = beta - 100;
	basicLinearTransform(img_original, alpha_value, beta_value);
}

void on_gamma_correction_trackbar(int, void *)
{
	double gamma_value = gamma_cor / 100.0;
	gammaCorrection(img_original, gamma_value);
}

// 2

void GammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);

	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{

			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}
#pragma endregion

/// <summary>                                                            
/// Operations performed on each component.                             
/// </summary>                                                           
/// <param name="">xxx.</param> 
int main(int argc, char *argv[])
{
		string file = "20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg";
		//string file = "20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg";
		//string file = "20161215 02.33_368.jpg";
		//string file = "20140612_MINEGARDEN_SURVEY2_00_14_50.jpg";
		//string file = "20140612_Minegarden_Survey_SIDESCAN_Renavigated_R1.jpg";

		if (!all_output) { cout << "--------------------------------------- START ---------------------------------------" << "\n"; }
		src = cv::imread(file);
		if (src.empty())
		{
			//if (!all_output) { cout << "src image is null" << "\n"; };
		}
		else
		{
			//if (!all_output) { cout << "src image is NOT null" << "\n"; };
		}
		//src = cv::imread("20161215 02.33_368R2.jpg");
		//src = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg");
		//C:\Users\JW\Documents\Visual Studio 2015\Projects\ConsoleApplication5\ConsoleApplication5\20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg
		//src = cv::imread("double ripple example_RR.jpg");
		Mat tempSourceImage = src;
		//if (!all_output) { imshow("0 - Source Img", src); };
		//if (!all_output) { cout << "00 - READING SOURCE IMAGE: 'src'" << "\n"; };

		int myArray[] = { 16, 2, 77, 40, 12071 };
		int myN = (sizeof(myArray) / sizeof(*myArray));
		//Histogram(myArray, myN);

#pragma region Image pre-processing
		//	////src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg");
		//////src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg");
		//////Mat src = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg"); 
		//////if (all_output){ imshow("0 - src", src);
		//srcImg = src;
		//bilateralFilter(src, srcImg1, 15, 80, 80);
		//blur(src, srcImg, Size(5, 5), Point(-1, -1));
		//GaussianBlur(src, srcImg2, Size(5, 5), 5);
		//int value = 128;
		//threshold(src, srcImg3, value, 255, CV_THRESH_OTSU);
		////cvAdaptiveThreshold(src, srcImg, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 13, 1);
		////	if (!all_output) { cout << "CV_THRESH_OTSU value = '" << value << "\n"; };
		//////if (!all_output){ cout << "01 - BLURRING SOURCE IMAGE: 'srcImg'" << "\n";}
		////Mat tempFilteredImage = srcImg;
		//if (!all_output) { imshow("bilateralFilter - filtered image - srcImg", srcImg1); }
		//if (!all_output) { imshow("blur - filtered image - srcImg", srcImg); }
		//if (!all_output) { imshow("GaussianBlur - filtered image - srcImg", srcImg2); }
		//if (!all_output) { imshow("threshold128 - filtered image - srcImg", srcImg3); }
		////cv::cvtColor(srcImg, GrayImg, cv::COLOR_BGR2GRAY);
		////if (!all_output){ cout << "02 - CONVERTING 'srcImg' TO GRAYSCALE " << "\n";}
		////if (!all_output) { imshow("2 - Blur GrayImg", GrayImg); }  
#pragma endregion


#pragma region Apply Gamma-correction functions
//
//	Mat dst(src.rows, src.cols, src.type());
//	GammaCorrection(src, dst, 0.6);
//
//	imshow("2", dst);
//	cv::cvtColor(dst, GrayImg, cv::COLOR_BGR2GRAY);
//	if (!all_output) { imshow("2 - Filtered GrayImg", GrayImg); }
////	img_original = src;
////	img_corrected = Mat(img_original.rows, img_original.cols * 2, img_original.type());
////	img_gamma_corrected = Mat(img_original.rows, img_original.cols * 2, img_original.type());
////
////	hconcat(img_original, img_original, img_corrected);
////	hconcat(img_original, img_original, img_gamma_corrected);
////
////	namedWindow("Brightness and contrast adjustments", WINDOW_AUTOSIZE);
////	namedWindow("Gamma correction", WINDOW_AUTOSIZE);
////
////	createTrackbar("Alpha gain (contrast)", "Brightness and contrast adjustments", &alpha, 500, on_linear_transform_alpha_trackbar);
////	createTrackbar("Beta bias (brightness)", "Brightness and contrast adjustments", &beta, 200, on_linear_transform_beta_trackbar);
////	createTrackbar("Gamma correction", "Gamma correction", &gamma_cor, 200, on_gamma_correction_trackbar);
////
////	while (true)
////	{
////		imshow("Brightness and contrast adjustments", img_corrected);
////		imshow("Gamma correction", img_gamma_corrected);
////
////		int c = waitKey(30);
////		if (c == 27)
////			break;
////	}
////
////	imwrite("linear_transform_correction.png", img_corrected);
////	imwrite("gamma_correction.png", img_gamma_corrected);
#pragma endregion

		cvtColor(src, GrayImg, CV_BGR2GRAY);

		ColumnsAnalysis(GrayImg);

		//Mat GrayscaleImage = GrayImg;
		//if (!all_output) { imshow("GrayscaleImage", GrayscaleImage); }

		//Mat linePixelsTempGraySrc3 = GrayImg;
		//SetNadirToBlack(GrayImg);
		//CalcHist(src);
		//CalcHistEq();

#pragma region Components /*and nadir removal*/
		Mat components = GetConnectedComponent(LeftAndRightImages);
#pragma endregion

		//Mat RemovedNadirImage = RemoveNadir(GrayImg);

#pragma region Histogram Equilization
//	char* source_window = "Source image";
//	char* equalized_window = "Equalized Image";
//	Mat dstHE;
//	/// Convert to grayscale
//	//cvtColor(src, src, CV_BGR2GRAY);
//
//	/// Apply Histogram Equalization
//	equalizeHist(GrayScaleCroppedImage, dstHE);
//
//	/// Display results
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);
//
//	imshow(source_window, GrayScaleCroppedImage);
//	imshow(equalized_window, dstHE);
#pragma endregion

	//if (all_output) { imshow("3 - components", components); }

	//if (all_output){ imshow("maskImages[0]", Mat(maskImages[0]));
	//Mat tempSrc1 = imread(file, CV_LOAD_IMAGE_UNCHANGED);
#pragma region Test Images
	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg");   
#pragma endregion

#pragma region MyRegion
		if (!all_output) { cout << "13 - STORING LOOPING-OVER-CANNY-IMAGES RESULTS IN TEXT FILE: 'ComponentsLoop.txt'" << "\n"; }
		//ofstream ComponentsLoop;
		//ComponentsLoop.open("ComponentsLoop.txt");

		//ofstream Bin_Analysis;
		//Bin_Analysis.open("Bin_Analysis.csv");
		if (!all_output) { cout << "14 - STORING BIN ANALYSIS RESULTS IN TEXT FILE: 'Bin_Analysis.CSV'" << "\n"; }

		//ofstream ComponentAngle;
		//ComponentAngle.open("ComponentAngle.txt");
		ofstream AngleTest_DataFile;
		AngleTest_DataFile.open("AngleTest_DataFile.txt", ios::app);

		//temp variable - to by-pass background mask
		//int maskElements = nFltrLabels2 - 1;

#pragma region Loop through each component
		//start with "mi=0 till mi<maskElement"
		//then just add a check inside for loop to skip mi=0 to ensure u address all elements of maskElement


		//333333333333333333333333333333333333333333333333333333333333333333333
		// This file records wether components are on image boundary or not
		ofstream ComponentTouchImageEdgeFile;
		ComponentTouchImageEdgeFile.open("ComponentTouchImageEdgeFile.txt");
		//333333333333333333333333333333333333333333333333333333333333333333333


		ofstream ComponentAnglesFile;
		ComponentAnglesFile.open("ComponentAnglesFile.txt");
		//333333333333333333333333333333333333333333333333333333333333333333333

		// Loop through both left and right image (after nadir removal)
		for (int LRimages = 0; LRimages < LeftAndRightImages.size(); LRimages++)
		{
			std::string sLRimages = std::to_string(LRimages);
			Mat linePixelsTempGraySrc3;
			cvtColor(src, linePixelsTempGraySrc3, cv::COLOR_BGR2GRAY);  //LeftAndRightImages.at(LRimages);
			Mat tempSrc1 = LeftAndRightImages.at(LRimages);
			Mat tempGraySrc = LeftAndRightImages.at(LRimages);

			if (!all_output) { imshow(sLRimages + "-left and right image", LeftAndRightImages.at(LRimages)); }

			// Loop through all component's (per left and right image)
			for (size_t mi = 1; mi < MasksCount.at(LRimages); mi++)
			{
				std::string smi = std::to_string(mi);
				Mat tempComponent = imread("mask_" + sLRimages + "_" + smi + ".bmp", CV_LOAD_IMAGE_UNCHANGED);
				//if (!all_output && LRimages == 0 && mi == 197) 
				//{ 
				//	imshow("BEFORE_" + sLRimages + "_" + "component_" + smi, tempComponent); 
				//};

				if (all_output) { cout << "LeftAndRightImages= " << sLRimages+ " -- " + "component= " + smi << "\n"; }

#pragma region Components laying on image border
				// Ignore component if touching image edge
				if (ComponentTouchImageEdge(tempComponent))
				{
					ComponentTouchImageEdgeFile << "Component " << smi << " is on image boundary" << "\n";
					if (all_output) { cout << "Component touching image edge" << "\n"; }
				}
				else // Process all components that's not topuching image edge, further
				{
					ComponentTouchImageEdgeFile << "Component " << smi << " is NOT on image boundary" << "\n";
					if (all_output && LRimages == 0) { imshow(sLRimages + "_" + "component_" + smi, tempComponent); };
					Mat GrayComponents;
					GrayComponents = tempComponent;
					Sobel(GrayComponents, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
					Sobel(GrayComponents, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
					Mat Mag(GrayComponents.size(), CV_32FC1);
					Mat Angle(GrayComponents.size(), CV_32FC1);
					cartToPolar(grad_x, grad_y, Mag, Angle, true);
					//////77777777777777777777777777777777777777777777777777777777777777777777777777777777777
					//ofstream Angle_DataFile;
					//Angle_DataFile.open("Angle_DataFile_" + smi + ".csv");
					//Angle_DataFile << Angle << "\n";
					//Angle_DataFile.close();
					//////77777777777777777777777777777777777777777777777777777777777777777777777777777777777

					std::array<std::vector<int>, 72> vvv{ {} };
					struct element {
						int bin;
						int i;
						int j;
						int angle;
						int value;
					};
					vector<element> container;
					int containerCount = 0;

					Canny(GrayComponents, cannyEdge, 100, 200);
					if (all_output) { imshow("5 - component cannyEdge - " + smi, cannyEdge); }

					Mat newAngle = Mat(Angle.size().height, Angle.size().width, Angle.type(), Scalar(0, 0, 0));

#pragma region CannyEdge Calculation-Angle per Component Calculation
					///Walk along cannyEdge rows
					for (size_t i = 0; i < cannyEdge.rows; i++)
					{
						///Walk along cannyEdge rows & columns
						for (size_t j = 0; j < cannyEdge.cols; j++)
						{
							/*				ComponentsLoop << "			Walk along Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";*/
											//Bin_Analysis << "Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";
											///if cannyEdge pixel intensity id non-zero
							if ((int)cannyEdge.at<uchar>(i, j) != 0)
							{
								//ComponentsLoop << "				Non-Zero Canny pixel value (coordinates - (x,y) ) = " << i << ", " << j << "\n";
								//ComponentsLoop << "				Storing Non-Zero Canny pixel value in container" << "\n";

								//Bin_Analysis << "Canny Edge Non-zero pixel (coordinates - (x,y) ) = " << i << ", " << j << "\n";

								newAngle.ptr<float>(i)[j] = Angle.ptr<float>(i)[j];							/*Create new Angle matrix*/
								container.push_back(element());												/*Initialise container*/
								container[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);	/*Store bin (newAngle/5)*/
								container[containerCount].i = i;											///Store row position
								container[containerCount].j = j;											///Store column position
								container[containerCount].angle = newAngle.ptr<float>(i)[j];				///Store new Angle value
								container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);			///Store canny pixel intensity
								containerCount++;
							}
						}
					}
#pragma endregion

					//ComponentsLoop << "Finished walking on Canny Edge" << "\n";
					//ComponentsLoop << "\n";
					//ComponentsLoop << "\n";

					//////8888888888888888888888888888888888888888888888888888888888888888888888888888888
					//ofstream newAngle_DataFile;
					//newAngle_DataFile.open("newAngle_DataFile" + smi + ".csv");
					//newAngle_DataFile << newAngle << "\n";
					//newAngle_DataFile.close();
					//////8888888888888888888888888888888888888888888888888888888888888888888888888888888

					//Bin_Analysis.close();
					//////999999999999999999999999999999999999999999999999999999
					ofstream ContainerFile;
					ContainerFile.open("ContainerFile_" + smi + ".txt");
					for (int i = 0; i < container.size(); i++)
					{
						ContainerFile << "container[" << i << "].bin= " << container[i].bin << "\n";
						ContainerFile << "container[" << i << "].i= " << container[i].i << "\n";
						ContainerFile << "container[" << i << "].j= " << container[i].j << "\n";
						ContainerFile << "container[" << i << "].angle= " << container[i].angle << "\n";
						ContainerFile << "container[" << i << "].value= " << container[i].value << "\n";
						ContainerFile << "\n";
						ContainerFile << "\n";
					}
					ContainerFile.close();
					//////999999999999999999999999999999999999999999999999999999
					int maxCount = 0;
					struct maxCountStruct {
						int bin;
						int angle;
						int size;
					};
					vector<maxCountStruct> maxCountContainer;
					int temp = 0;
					struct MaxElementStruct {
						int bin = 0;
						int angle = 0;
						int size = 0;
					};
					MaxElementStruct mes;

					//ofstream KeepingTrackOfContainers_DataFile;
					//KeepingTrackOfContainers_DataFile.open("KeepingTrackOfContainers_DataFile.csv");

					/*ComponentsLoop << "			For every element in container (total size = " << container.size() << ")" << "\n";*/
					/// Grouping bin values together and counting them
					/// For every element in container
					for (size_t l = 0; l < container.size(); l++)
					{
						/*ComponentsLoop << "			container[l] = " << l << "\n";*/
						/// If new container is empty (at start)
						if (maxCountContainer.empty())
						{
							/*	ComponentsLoop << "				Initial element (new container empty) in new container[l] = " << l << "\n";*/
							maxCountContainer.push_back(maxCountStruct());		/// Initialise new container
							maxCountContainer[l].bin = container[l].bin;		/// Store container bin value in new container bin field
							maxCountContainer[l].angle = container[l].angle;	/// Store container angle value in new container angle field
							maxCountContainer[l].size += 1;						/// Increment new container size field (counting elements with similair bin values)
						}
						else  /// If not at start (new container contains first element)
						{
							/// For every element in new container (new container contains at least one element thus far) & its elements will increase with every loop
							for (size_t m = 0; m < maxCountContainer.size(); m++)
							{
								/*ComponentsLoop << "				For every element in new container[m] (as it's filling up ) = " << m << "\n";*/

								//KeepingTrackOfContainers_DataFile << "container iterator (l): " << l << "\n";
								//KeepingTrackOfContainers_DataFile << "maxCountContainer iterator (m)= " << m << "\n";
								/// 
								if (maxCountContainer[m].bin == container[l].bin)
								{
									//ComponentsLoop << "					When container element [l] already exist in new container [m] (don't re-add it, just update it's count ) = " << l << ", " << m << "\n";
									maxCountContainer[m].size += 1;
									break;
								}
								else if (m == maxCountContainer.size() - 1)
								{
									maxCountContainer.push_back(maxCountStruct());
									maxCountContainer[maxCountContainer.size() - 1].bin = container[l].bin;
									maxCountContainer[maxCountContainer.size() - 1].angle = container[l].angle;
									maxCountContainer[maxCountContainer.size() - 1].size += 1;
									break;
								}

								//////666666666666666666666666666666666666666666666666666666666666666666666666666666
								//ofstream maxCountContainer_DataFile;
								//maxCountContainer_DataFile.open("maxCountContainer_DataFile_" + smi + ".csv");
								//for (int i = 0; i < maxCountContainer.size(); i++)
								//{
								//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].i= " << i << "\n";
								//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].bin= " << maxCountContainer[i].bin << "\n";
								//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].angle= " << maxCountContainer[i].angle << "\n";
								//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].size= " << maxCountContainer[i].size << "\n";
								//	maxCountContainer_DataFile << "\n";
								//	maxCountContainer_DataFile << "\n";
								//}
								//maxCountContainer_DataFile.close();
								//maxCountContainer_DataFile.close();
								//////666666666666666666666666666666666666666666666666666666666666666666666666666666

						/*		ComponentsLoop << "		Adjusting element with the highest frequency" << "\n";*/
								if (maxCountContainer[m].size > temp)	///Find bin with the most elements
								{
									temp = maxCountContainer[m].size;
									mes.bin = (int)maxCountContainer[m].bin;	///Bin with most elements (bin ID)
									mes.angle = (int)maxCountContainer[m].angle;
									mes.size = (int)maxCountContainer[m].size;
								}
								/*	ComponentsLoop << "		Element with highest frequency (" << mes.size << "= " << mes.angle << "\n";*/
							}
						}
					}
					//KeepingTrackOfContainers_DataFile.close();
					//AngleTest_DataFile.open("AngleTest_DataFile.txt");
					if (all_output) { AngleTest_DataFile << "Component " << smi << " has angle " << mes.angle << " with centroid " << maskCentroid.at(mi) << "\n"; };
					if (all_output) { cout << "Component " << smi << " has angle " << mes.angle << " with centroid " << maskCentroid.at(mi) << "\n"; };
					
					componentAngles.push_back(mes.angle);
					if (LRimages == 0)
					{
						LcomponentAngles.push_back(mes.angle);
						if (!all_output) { ComponentAnglesFile << LRimages << "," << mes.angle << "\n"; };
					}
					else
					{
						RcomponentAngles.push_back(mes.angle);
						if (!all_output) { ComponentAnglesFile << LRimages << "," << mes.angle << "\n"; };
					}
					//anglesComponent[LRimages]
					//if (!all_output) { AngleTest_DataFile << "The biggest number is: " << mes.size << " at bin " << mes.bin << endl; }
					//if (!all_output){ AngleTest_DataFile << "Angle (mes)- " << smi << "= " << mes.angle << "\n";}
					//ComponentAngle << "Angle (mes)- " << smi << "\n";
					/*Mat tempGraySrc = GrayImg;*/
					for (size_t n = 0; n < container.size(); n++)
					{
						if (container[n].bin == mes.bin)
						{
							tempGraySrc.at<uchar>(container[n].i, container[n].j) = 255;
						}
					}
					// Using tempSrc2 just for drawing points and lines - for display purposes only
					Mat tempSrc2 = imread(file, CV_LOAD_IMAGE_UNCHANGED);
					//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
					//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg", CV_LOAD_IMAGE_UNCHANGED);
					//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg");
#pragma region Bounding Box
					vector<double> lengths(4);
					double rectSize_b;
					size_t imgCount = 0;
					////if (all_output){ cout << "maskImages.size()= " << maskImages.size() << "\n";}
					//for (imgCount; imgCount < maskImages.size(); imgCount++)
					//{
					Mat tempPoints;
					findNonZero(imread("mask_" + sLRimages + "_" + smi + ".bmp", CV_LOAD_IMAGE_UNCHANGED), tempPoints);
					//if (!all_output) { cout << "tempPoints = " << tempPoints << "\n"; };
					Points.push_back(tempPoints);
					//}
					Point2f vtx[4];
					RotatedRect box = minAreaRect(Points[countComponentsNotTouchingImageEdge - 1]); //only the first Mat Points
					//if (!all_output){ AngleTest_DataFile << "RotatedRect box = minAreaRect(Points[mi])" << box.angle << "\n";}
					//if (!all_output) { AngleTest_DataFile << "###########################################################" << "\n"; };
					AngleTest_DataFile.close();
					box.points(vtx);
					/*Mat tempSrc1 = imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);*/
					for (int i = 0; i < 4; i++)
					{
						line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
						line(tempSrc2, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
						lengths.push_back(norm((vtx[(i + 1) % 4]) - (vtx[i])));
					}
					if (all_output) { imshow("Bounding Box", tempSrc1); }
					//if (all_output){ imwrite("Bounding_Box" + smi + ".bmp", tempSrc1);
					////if (all_output){ cout << "minAreaRect Angle - "<<smi<<"= " << box.angle + 180 << "\n";}
					//if (all_output){ cout << "minAreaRect width= " << box.size.width << "\n";}
					//if (all_output){ cout << "minAreaRect height= " << box.size.height << "\n";}
#pragma endregion
					Mat plotImage = src;	//plotImage used just to draw - for display purposes
					circle(plotImage, maskCentroid.at(mi), 1, Scalar(0, 255, 0), 1, 8, 0);
					circle(tempSrc2, maskCentroid.at(mi), 1, Scalar(0, 255, 0), 1, 8, 0);

#pragma region walk in edge angle direction
					Point2f u, u2, u22, v;
					Point2f w1, w2;
					//if (all_output){ cout << "cos((mes.angle)* CV_PI / 180.0)= " << cos((mes.angle)* CV_PI / 180.0) << "\n";}
					//if (all_output){ cout << "sin((mes.angle)* CV_PI / 180.0)= " << sin((mes.angle)* CV_PI / 180.0) << "\n";}
					u = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
					u2 = u;
					rectSize_b = *max_element(lengths.begin(), lengths.end());
					double d = 0.1*rectSize_b;
					double normU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
					////if (all_output){ cout << "normU= " << normU << "\n";}
					v = Point2f(u.x / normU, u.y / normU);
					//Mat tempSrcW1 = src, tempSrcW2 = src;
					for (size_t i = 0; i < 3; i++)
					{
						if (i == 0)
						{	// starting point = center of mask
							w1.x = maskCentroid.at(mi).x + v.x*d;	//one side
							w1.y = maskCentroid.at(mi).y + v.y*d;

							w2.x = maskCentroid.at(mi).x - v.x*d;	//other side
							w2.y = maskCentroid.at(mi).y - v.y*d;
						}
						else
						{	// points on either-side of mask center point
							w1.x = u2.x + v.x*d;		//one side
							w1.y = u2.y + v.y*d;

							w2.x = u22.x - v.x*d;		//other side
							w2.y = u22.y - v.y*d;
						}
						////if (all_output){ cout << "i - " << i << "2-Plot here= " << w1 << ", " << w2 << "\n";}
						//circle(plotImage, w1, 1, Scalar(0, 0, 255), 1, 8, 0);
						//circle(plotImage, w2, 1, Scalar(255, 0, 0), 1, 8, 0);

						circle(tempSrc2, w1, 1, Scalar(55, 55, 55), 1, 8, 0);
						circle(tempSrc2, w2, 1, Scalar(55, 55, 55), 1, 8, 0);
						//circle(tempSrcW1, w1, 1, Scalar(0, 0, 0), 1, 8, 0);
						//circle(tempSrcW2, w2, 1, Scalar(255, 255, 255), 1, 8, 0);
						u2 = w1;
						u22 = w2;
					}
					if (all_output) { imshow("walk in edge angle direction - component nr." + smi, tempSrc2); };
					//rectangle(tempSrcW1, cv::Point2f(10, 10), cv::Point2f(src.size().width - 10, src.size().height - 10), cv::Scalar(255, 0, 0));
					//if (all_output){ imshow("tempSrcW1- " + smi, tempSrcW1);
					//if (all_output){ imwrite("tempSrcW1_" + smi + ".bmp", tempSrcW1);

					//rectangle(tempSrcW2, cv::Point2f(10, 10), cv::Point2f(src.size().width - 10, src.size().height - 10), cv::Scalar(0, 255, 0));
					//if (all_output){ imshow("tempSrcW2- " + smi, tempSrcW2);
					//if (all_output){ imwrite("tempSrcW2_" + smi + ".bmp", tempSrcW2);
#pragma endregion

					struct buffer {
						std::vector<double> pixValues;
						Point2f startPoint;
						Point2f endPoint;
						//int j;
						//int angle;
						//int value;
					};
					vector<buffer> Profiles;
					int ProfilesCount = 0;

#pragma region walk perpendicular in edge angle direction
					Point2f uu, uu2, uu22, vv, ep11, ep12, ep21, ep22;
					Point2f ww1, ww2;
					uu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
					uu2 = uu;
					rectSize_b = *max_element(lengths.begin(), lengths.end());
					//double dd = 0.1*rectSize_b;
					double normUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
					vv = Point2f(uu.x / normUU, uu.y / normUU);
					u = vv;
					//rotate and swap
					double tempXX = vv.x;
					vv.x = -vv.y;
					vv.y = tempXX;
					int e = 20;

					Mat tempGraySrc3;
					cv::cvtColor(src, tempGraySrc3, cv::COLOR_BGR2GRAY);
					Mat tempSrc3 = tempGraySrc;//imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
					//Mat linePixelsTempGraySrc3 = tempGraySrc3;
					int i = -1;
					/*std::string ii = std::to_string(i);*/
					Mat imageArray[10];
					bool beginning = true;

					for (i = -1; i < box.size.width / 2; i++)
					{
						std::string ii = std::to_string(i);

						if (i == -1)
						{
							ww1.x = maskCentroid.at(mi).x;// +vv.x*d;	//one side
							ww1.y = maskCentroid.at(mi).y;// +vv.y*d;

							ww2.x = maskCentroid.at(mi).x;// -vv.x*d;	//other side
							ww2.y = maskCentroid.at(mi).y;// -vv.y*d;

													   //end points of profile
							ep11.x = ww1.x - ((box.size.width + e) / 2) * uu.x;
							ep11.y = ww1.y - ((box.size.width + e) / 2) * uu.y;
							ep12.x = ww1.x + ((box.size.width + e) / 2) * uu.x;
							ep12.y = ww1.y + ((box.size.width + e) / 2) * uu.y;

							ep21.x = ww2.x - ((box.size.width + e) / 2) * uu.x;
							ep21.y = ww2.y - ((box.size.width + e) / 2) * uu.y;
							ep22.x = ww2.x + ((box.size.width + e) / 2) * uu.x;
							ep22.y = ww2.y + ((box.size.width + e) / 2) * uu.y;
						}
						else if (i == 0)
						{	// starting point = center of mask
							ww1.x = maskCentroid.at(mi).x + vv.x*d;	//one side
							ww1.y = maskCentroid.at(mi).y + vv.y*d;

							ww2.x = maskCentroid.at(mi).x - vv.x*d;	//other side
							ww2.y = maskCentroid.at(mi).y - vv.y*d;

							//end points of profile
							ep11.x = ww1.x - ((box.size.width + e) / 2) * uu.x;
							ep11.y = ww1.y - ((box.size.width + e) / 2) * uu.y;
							ep12.x = ww1.x + ((box.size.width + e) / 2) * uu.x;
							ep12.y = ww1.y + ((box.size.width + e) / 2) * uu.y;

							ep21.x = ww2.x - ((box.size.width + e) / 2) * uu.x;
							ep21.y = ww2.y - ((box.size.width + e) / 2) * uu.y;
							ep22.x = ww2.x + ((box.size.width + e) / 2) * uu.x;
							ep22.y = ww2.y + ((box.size.width + e) / 2) * uu.y;

							beginning = false;
						}
						else
						{	// points on either-side of mask center point
							ww1.x = uu2.x + vv.x*d;		//one side
							ww1.y = uu2.y + vv.y*d;

							ww2.x = uu22.x - vv.x*d;	//other side
							ww2.y = uu22.y - vv.y*d;

							//end points of profile
							ep11.x = ww1.x - ((box.size.width + e) / 2) * uu.x;
							ep11.y = ww1.y - ((box.size.width + e) / 2) * uu.y;
							ep12.x = ww1.x + ((box.size.width + e) / 2) * uu.x;
							ep12.y = ww1.y + ((box.size.width + e) / 2) * uu.y;

							ep21.x = ww2.x - ((box.size.width + e) / 2) * uu.x;
							ep21.y = ww2.y - ((box.size.width + e) / 2) * uu.y;
							ep22.x = ww2.x + ((box.size.width + e) / 2) * uu.x;
							ep22.y = ww2.y + ((box.size.width + e) / 2) * uu.y;
						}
						circle(tempSrc2, ww2, 1, Scalar(255, 0, 0), 1, 8, 0); //turqoise
						circle(tempSrc2, ww1, 1, Scalar(55, 0, 0), 1, 8, 0); //
																				 //circle(tempSrc2, ww2, 1, Scalar(255, 255, 10), 1, 8, 0); //
						circle(tempSrc2, ep11, 1, Scalar(0, 0, 255), 1, 8, 0); //
						circle(tempSrc2, ep12, 1, Scalar(0, 0, 255), 1, 8, 0); //
						circle(tempSrc2, ep21, 1, Scalar(0, 0, 255), 1, 8, 0); //
						circle(tempSrc2, ep22, 1, Scalar(0, 0, 255), 1, 8, 0); //

						uu2 = ww1;
						uu22 = ww2;

#pragma region DrawLines
						int thickness = 0.2;
						int lineType = 8;
						line(tempGraySrc3,
							Point(ep11.x, ep11.y),
							Point(ep12.x, ep12.y),
							Scalar(255, 0, 0),
							thickness,
							lineType);

						profileLinesCount += 1;
						std::string profileLinesCount1 = std::to_string(profileLinesCount);
						if (all_output) { imshow("Profile Line-" + profileLinesCount1, tempGraySrc3); };

						line(tempGraySrc3,
							Point(ep21.x, ep21.y),
							Point(ep22.x, ep22.y),
							Scalar(255, 0, 0),
							thickness,
							lineType);

						profileLinesCount += 1;
						std::string profileLinesCount2 = std::to_string(profileLinesCount);
						if (all_output) { imshow("Profile Line-" + profileLinesCount2, tempGraySrc3); };
#pragma endregion

#pragma region LinePixels

						// grabs pixels along the line (pt1, pt2)
						// from 8-bit 3-channel image to the buffer
						LineIterator it1B(tempGraySrc3, Point(ep11), Point(ep12), 8);		// Lines before centroid
						LineIterator it1A(tempGraySrc3, Point(ep21), Point(ep22), 8);		// Lines after centroid
						//LineIterator it2(tempSrc3, Point(ep21), Point(ep22), 8);
						LineIterator it11B = it1B;
						LineIterator it11A = it1A;
						//LineIterator it22 = it2;
						//vector<Vec3b> buf(it.count);

						//ofstream file;
						vector<float> pixelsOnLineB;
						vector<float> pixelsOnLineA;
						vector<vector<float>> pixels;

						Mat linePixel;
						//float pixelsUnderLine[it1.count];

						// Record pixels under line B (Before centroid)
						for (int l = 0; l < it1B.count; l++, ++it1B)
						{
							Profiles.push_back(buffer());
							Profiles[ProfilesCount].startPoint = ep11;
							Profiles[ProfilesCount].endPoint = ep12;
							double valB = (double)linePixelsTempGraySrc3.at<uchar>(it1B.pos());
							pixelsOnLineB.push_back(valB);
							linePixel.push_back(valB);
							//5555555555555555555555555555555555555555555555555555555555555555
				/*			ofstream tempGraySrc3DataFile;
							tempGraySrc3DataFile.open("linePixelsTempGraySrc3" + profileLinesCount1 + "B.csv");
							tempGraySrc3DataFile << linePixelsTempGraySrc3 << "\n";
							tempGraySrc3DataFile << "\n";
							tempGraySrc3DataFile << "\n";
							tempGraySrc3DataFile.close();*/
							//55555555555555555555555555555555555555555555555555555555555555555555

							////if (all_output){ cout << "Point(ep11.x, ep11.y), Point(ep12.x, ep12.y) = " << Point(ep11.x, ep11.y) << ", "<< Point(ep12.x, ep12.y) << "\n";
							////if (all_output){ cout << "it1.pos() = " << Point(ep11.x, ep11.y) << ", " << it1B.pos() << "\n";
							////if (all_output){ cout << "(double)tempGraySrc3.at<uchar>(it1.pos()) = " << (double)linePixelsTempGraySrc3.at<uchar>(it1B.pos()) << "\n";}
							Profiles[ProfilesCount].pixValues.push_back(valB);// (double)tempSrc3.at<uchar>(it1.pos());

							//double val = (double)src_gray.at<uchar>(it.pos());
							//buf[i] = val;

							//std::string L = std::to_string(l);
							//file.open("buf_" + format("(%d,%d)", Profiles[ProfilesCount].startPoint, Profiles[ProfilesCount].endPoint) + ".csv", ios::app);
							//file.open("buf_" + L + ".csv", ios::app);
							//file << Profiles[ProfilesCount].startPoint << "\n";
							//file << Profiles[ProfilesCount].endPoint << "\n";
							//file << Mat(Profiles[ProfilesCount].pixValues) << "\n";
							//file.close();
							ProfilesCount += 1;
						}

						// Record pixels under line A (After centroid)
						for (int l = 0; l < it1A.count; l++, ++it1A)
						{
							Profiles.push_back(buffer());
							Profiles[ProfilesCount].startPoint = ep21;
							Profiles[ProfilesCount].endPoint = ep22;
								double valA = (double)linePixelsTempGraySrc3.at<uchar>(it1A.pos());
							pixelsOnLineA.push_back(valA);
							linePixel.push_back(valA);
							//5555555555555555555555555555555555555555555555555555555555555555
				/*			ofstream tempGraySrc3DataFile;
							tempGraySrc3DataFile.open("linePixelsTempGraySrc3" + profileLinesCount2 + "A.csv");
							tempGraySrc3DataFile << linePixelsTempGraySrc3 << "\n";
							tempGraySrc3DataFile << "\n";
							tempGraySrc3DataFile << "\n";
							tempGraySrc3DataFile.close();*/
							//55555555555555555555555555555555555555555555555555555555555555555555
							////if (all_output){ cout << "Point(ep11.x, ep11.y), Point(ep12.x, ep12.y) = " << Point(ep21.x, ep21.y) << ", " << Point(ep22.x, ep22.y) << "\n";}
							////if (all_output){ cout << "it1.pos() = " << Point(ep21.x, ep21.y) << ", " << it1A.pos() << "\n";}
							////if (all_output){ cout << "(double)tempGraySrc3.at<uchar>(it1.pos()) = " << (double)linePixelsTempGraySrc3.at<uchar>(it1A.pos()) << "\n";}
							Profiles[ProfilesCount].pixValues.push_back(valA);// (double)tempSrc3.at<uchar>(it1.pos());

																			  //double val = (double)src_gray.at<uchar>(it.pos());
																			  //buf[i] = val;

							//std::string L = std::to_string(l);
							//file.open("buf_" + format("(%d,%d)", Profiles[ProfilesCount].startPoint, Profiles[ProfilesCount].endPoint) + ".csv", ios::app);
							//file.open("buf_" + L + ".csv", ios::app);
							//file << Profiles[ProfilesCount].startPoint << "\n";
							//file << Profiles[ProfilesCount].endPoint << "\n";
							//file << Mat(Profiles[ProfilesCount].pixValues) << "\n";
							//file.close();
							ProfilesCount += 1;
						}
#pragma endregion

						//4444444444444444444444444444444444444444444444444444444444444444444

						if (pixelsOnLineB.empty())
						{
							////if (all_output){ cout << "pixelsOnLine is empty" << "\n";}
						}
						else
						{
							//ofstream PixelsOnLineBFile;
							//PixelsOnLineBFile.open("PiixelsOnLineB" + profileLinesCount1 + ".csv");
							//PixelsOnLineBFile << Mat(pixelsOnLineB) << "\n";;
							//PixelsOnLineBFile << "\n";
							//PixelsOnLineBFile << "\n";
							//PixelsOnLineBFile.close();
						}

						if (pixelsOnLineA.empty())
						{
							//if (all_output){ cout << "pixelsOnLineA is empty" << "\n";}
						}
						else
						{
							/*		ofstream PixelsOnLineAFile;
									PixelsOnLineAFile.open("PixelsOnLineA" + profileLinesCount2 + ".csv");
									PixelsOnLineAFile << Mat(pixelsOnLineA) << "\n";;
									PixelsOnLineAFile << "\n";
									PixelsOnLineAFile << "\n";
									PixelsOnLineAFile.close();*/
						}
						//4444444444444444444444444444444444444444444444444444444444444444444
					}
					if (all_output) { imshow("walk perpendicular in edge angle direction - component nr." + smi, tempSrc2); }
#pragma endregion  

#pragma region EPs
					Point2f uuu, uuu2, uuu22, vvvv, ep1, ep2;
					Point2f www1, www2;
					uuu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
					uuu2 = uuu;
					rectSize_b = *max_element(lengths.begin(), lengths.end());
					//double dd = 0.1*rectSize_b;
					double normUUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
					vvvv = Point2f(uuu.x / normUUU, uuu.y / normUUU);
					//rotate and swap
					Point2d vvvvv;
					double tempXXX = vvvv.x;
					vvvvv.x = -vvvv.y;
					vvvvv.y = tempXXX;
					for (size_t i = 0; i < 10; i++)
					{
						if (i == 0)
						{
							www1.x = maskCentroid.at(mi).x + vvvv.x*d;
							www1.y = maskCentroid.at(mi).y + vvvv.y*d;
							//ep1.x = www1.x-(box.size.width/2)*(vvvv.x*d);
							//ep1.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
							//ep2.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
							//ep2.y = www1.y - (box.size.width / 2)*(vvvv.y*d);

							www2.x = maskCentroid.at(mi).x - vvvvv.x*d;
							www2.y = maskCentroid.at(mi).y - vvvvv.y*d;
							ep1.x = www2.x - (box.size.width / 2)*(vvvv.x);
							ep1.y = www2.y - (box.size.width / 2)*(vvvv.y);
							ep2.x = www2.x + (box.size.width / 2)*(vvvv.x);
							ep2.y = www2.y + (box.size.width / 2)*(vvvv.y);
						}
						else
						{
							www1.x = uuu2.x + vvvv.x*d;
							www1.y = uuu2.y + vvvv.y*d;
							//ep1.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
							//ep1.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
							//ep2.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
							//ep2.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
							////if (all_output){ cout << "minAreaRect Angle= " << box.angle + 180 << "\n";}
							www2.x = uuu22.x - vvvv.x*d;
							www2.y = uuu22.y - vvvv.y*d;
							ep1.x = www2.x - (box.size.width / 2)*(vvvv.x);
							ep1.y = www2.y - (box.size.width / 2)*(vvvv.y);
							ep2.x = www2.x + (box.size.width / 2)*(vvvv.x);
							ep2.y = www2.y + (box.size.width / 2)*(vvvv.y);
							////if (all_output){ cout << "ww2= " << ww2 << "\n";}
						}
						////if (all_output){ cout << "i - " << i << "1-Plot here= " << ww1 << ", " << ww2 << "\n";}
						//circle(src, ww1, 1, Scalar(0, 255, 255), 1, 8, 0); //yellow
						//circle(plotImage, ww2, 1, Scalar(255, 255, 0), 1, 8, 0); //turqoise
						//circle(tempSrc2, www1, 1, Scalar(255, 255, 10), 1, 8, 0); //
						circle(tempSrc2, www2, 1, Scalar(255, 255, 100), 1, 8, 0); //
						circle(tempSrc2, ep1, 1, Scalar(255, 255, 200), 1, 8, 0);
						circle(tempSrc2, ep2, 1, Scalar(255, 255, 10), 1, 8, 0);
						uuu2 = www1;
						uuu22 = www2;
					}
					//if (all_output){ imshow("Profile Lines", tempSrc2);
#pragma endregion



		//if (all_output){ imshow("Plot Image", plotImage);
		//if (all_output){ imshow("Source - component nr." + smi, tempSrc2); 
		//if (all_output){ imwrite("Source_component_nr_" + smi + ".bmp", tempSrc2);
		//if (all_output){ imshow("Grayscale- component nr." + smi, tempGraySrc3);
		//if (all_output){ imwrite("Grayscale- component nr." + smi + ".bmp", tempGraySrc3);
		//} 
				}

#pragma endregion


			}
		}

		//vector<int> tet = { 1,2,1,3,4,2,5,2,5,26 };
		Histogram(componentAngles);

		ComponentTouchImageEdgeFile.close();
		ComponentAnglesFile.close();
#pragma endregion

#pragma endregion

		//imshow("grayscale image", src_gray);
		//setMouseCallback("0 - Source Img", CallBackFunc, NULL);

		//if (!all_output) { cout << "15 - RETRIEVING COMPONENTS (ONE-BY-ONE) FROM: 'maskImages'" << "\n"; }

		//if (!all_output) { cout << "16 - EACH COMPONENT STORED IN: 'tempComponent'" << "\n"; }

		//if (!all_output) { cout << "17 - CALCULATING EDGES/GRADIENT PER COMPONENT USING: Sobel" << "\n"; }
		//if (!all_output) { cout << "18 -  -- STORING CALCULATED EDGES RESULTS: 'grad_x' AND 'grad_y'" << "\n"; }
		//if (!all_output) { cout << "19 -  -- CALCULATING MAGNITUDE AND ANGLE MATRICES : 'Mag' AND 'Angle'" << "\n"; }
		//if (!all_output) { cout << "20 -  -- WRITING CALCULATED ANGLE RESULTS TO FILE: 'Angle_DataFile_i.csv'" << "\n"; }
		//if (!all_output) { cout << "21 -  -- CALCULATING CANNY EDGE: 'cannyEdge'" << "\n"; }
		//if (!all_output) { cout << "22 -  -- WRITING CALCULATED CANNY EDGE RESULTS TO FILE: 'cannyEdge_DataFile_I.csv'" << "\n"; }
		//if (!all_output) { cout << "23 -  -- -- FOR EACH CANNY EDGE PIXEL" << "\n"; }
		//if (!all_output) { cout << "24 -  -- -- FIND EACH NON-ZERO CANNY EDGE PIXEL" << "\n"; }
		//if (!all_output) { cout << "25 -  -- -- -- STORE EACH PIXEL IN NEW MATRIX: 'newAngle'" << "\n"; }
		//if (!all_output) { cout << "26 -  -- -- -- STORE BIN #, COORDINATES, ANGLE VALUE AND PIXEL VALUE IN: 'Container'" << "\n"; }

		//if (!all_output) { cout << "27 -  -- SORT 'Container' DATA AND WRITE RESULTS TO FILE: 'KeepingTrackOfContainers_DataFile_i.csv' AND 'ContainerFile_i.csv'" << "\n"; }

		//if (!all_output) { cout << "28 -  -- STORE SORTED DATA IN: 'maxCountContainer'" << "\n"; }

		//if (!all_output) { cout << "29 -  -- BIN'S WITH THEIR FREQUENCY ON UNIQUE ELEMENTS STORED IN: 'maxCountContainer_DataFile_i.csv'" << "\n"; }

		//if (!all_output) { cout << "30 -  -- BIN (BIN-VALUE) WITH HIGHEST FREQUENCY (WITH ITS CORRESPONDING ANGLE-VALUE AND BIN-SIZE) CALCULATED AND STORED IN: 'mes'" << "\n"; }

		//if (!all_output) { cout << "31 -  -- CALCULATING BOUNDING BOX" << "\n"; }

		//if (!all_output) { cout << "32 -  -- CALCULATING VECTOR IN DIRECTION OF PREVIOUSLY CALCULATED ANGLE (mes.angle): 'v'" << "\n"; }
		//if (!all_output) { cout << "33 -  -- -- CALCULATED VECTOR POINTS STORED IN: 'w1-one_side of cetroid point AND w2-otherside of cetroid point'" << "\n"; }
		//if (!all_output) { cout << "34 -  -- -- PLOTTING CALCULATING VECTOR ON: 'tempSrc2'" << "\n"; }


		//if (!all_output) { cout << "35 -  -- CALCULATING VECTOR IN DIRECTION PERPENDICULAR TO 'v': 'vv'" << "\n"; }
		//if (!all_output) { cout << "36 -  -- -- CALCULATED VECTOR POINTS STORED IN: 'ww1-one_side of cetroid point AND ww2-otherside of cetroid point'" << "\n"; }
		//if (!all_output) { cout << "37 -  -- -- PLOTTING CALCULATED VECTOR ON: 'tempSrc2'" << "\n"; }

		//if (!all_output) { cout << "38 -  -- CALCULATING VECTOR IN DIRECTION PERPENDICULAR TO 'v': 'vv'" << "\n"; }
		//if (!all_output) { cout << "39 -  -- -- CALCULATED END-POINTS STORED IN: Point on one side (x,y)-'ep11, ep12' AND Point on otherside (x,y)-'ep21, ep22'" << "\n"; }
		//if (!all_output) { cout << "40 -  -- -- DRAWING LINES BETWEEN POINTS (ep11,ep12) AND (ep21,ep22)" << "\n"; }
		//if (!all_output) { cout << "41 -  -- -- PLOTTING CALCULATING LINES ON: 'tempSrc2'" << "\n"; }

		//if (!all_output) { cout << "42 -  -- CALCULATING PIXEL PIXEL VALUES UNDER LINES" << "\n"; }
		//if (!all_output) { cout << "43 -  -- -- XXX" << "\n"; }
		//if (!all_output) { cout << "44 -  -- -- YYY" << "\n"; }
		//if (!all_output) { cout << "45 -  -- -- ZZZ" << "\n"; }
		//ComponentsLoop.close();
		//ComponentAngle.close();
		//if (all_output){ imshow("Plot Image", src);
		//if (all_output){ imshow("Bounding Box", tempSrc1);
		waitKey(0);
		return 0;
	//}
	//catch (cv::Exception & e)
	//{
	//	cerr << e.msg << endl; // output exception message
	//}
}