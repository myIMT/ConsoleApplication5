//#include <iostream>
//#include "opencv2/core/core.hpp"
//#include <opencv2/core/utility.hpp>
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include <opencv2/opencv.hpp>
//#include <math.h>
//#include <fstream>
//#include <string> 
//#include <array>
////#include <opencv3_0_0/plot.hpp>
//#include <numeric> 
//#include "GraphUtils.h"
//
//using std::vector;
//
//using namespace cv;
//using namespace std;
//
//#pragma region Global Variables
//const std::string keys =
//"{help      |             | print this message    }"
//"{@image    |contours     | load image            }"
//"{j         |j.png        | j image        }"
//"{contours  |contours.png | contours image        }"
//;
//
//int threshval = 60;
//int bw_constant = 128;
//vector<Vec4i> hierarchy;
//Mat src, srcImg, GrayImg, hist, cannyEdge, detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;
//
//int ddepth = CV_32FC1;// CV_16S;
//int scale = 1;
//int delta = 0;
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
//int channels[] = { 0 };
//int binID;
//
//vector<Mat> Points;
////struct masks {
////	Mat mask;
////	Point maskCentroid;
////};
////vector<masks> maskImages;
//vector<Mat> maskImages;
//vector<Point> maskCentroid;
//
//int secondBiggestArea = 0;
//int secondBiggestAreaIndex = -1;
//int thirdBiggestArea = 0;
//int thirdBiggestAreaIndex = -1;
//int secondBiggestAreaX;
//int secondBiggestAreaY;
//int secondBiggestAreaWidth;
//int secondBiggestAreaHeight;
//int thirdBiggestAreaX;
//int thirdBiggestAreaY;
//int thirdBiggestAreaWidth;
//int thirdBiggestAreaHeight;
//
//struct connectedComponentsWithStatsStruct {
//	int COMP_STRUCT_NR;
//	int CC_STAT_LEFT;
//	int CC_STAT_TOP;
//	int CC_STAT_WIDTH;
//	int CC_STAT_HEIGHT;
//	int CC_STAT_AREA;
//	int CC_STAT_CENTROID_X;
//	int CC_STAT_CENTROID_Y;
//};
//vector<connectedComponentsWithStatsStruct> CompStats;
//int CompStructCount = 0;
//int profileLinesCount = 0;
//
//bool cout_output = false;
//bool file_output = false;
//bool imshow_output = false;
//bool imwrite_output = false;
//bool all_output = false;
//
//Mat secondBiggestAreaMat, thirdBiggestAreaMat, nadirMat;
//int nFltrLabels2 = -1;
//Mat testMaskiRead;
//Mat GrayScaleCroppedImage;
//Mat cropedLeftImage;
//Mat cropedRightImage;
//
//Point pt1, pt2;
//Mat src_gray;
//int clickCounter = 0, lineCounter = 0, pixelCounter = 0, drawCounter = 0;
//
//	string file = "20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg";
////	//string file = "20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg";
////	//string file = "20161215 02.33_368.jpg";
////	//string file = "20140612_MINEGARDEN_SURVEY_00_14_50.jpg";
////----------------------------------------------------  
//#pragma endregion
//
//
//int main(int argc, char *argv[])
//{
//	Mat nadirRemoveImage = imread(file);
//
//	cout << "width of image= " << nadirRemoveImage.size().width << "\n";
//	Mat nadirRemovedGray;
//	cvtColor(nadirRemoveImage, nadirRemovedGray, CV_BGR2GRAY);
//#pragma region Test width even or odd
//	if ((nadirRemovedGray.size().width & 1) == 0)
//		printf("EVEN!\n");
//	else
//		printf("ODD!\n");
//#pragma endregion
//
//	imshow("BEFORE", nadirRemovedGray);
//	int value = 12;
//	int middle = nadirRemovedGray.cols / 2;
//	Mat lrImage = nadirRemovedGray;
//	Mat testImage;
//	Mat newMat = Mat(nadirRemovedGray.size().height, nadirRemovedGray.size().width, nadirRemovedGray.type(), Scalar(255,255,255));
//	imshow("BEFORE newMat", newMat);
//#pragma region cycle through left-cols pixels
//	for (int r = 0; r < lrImage.rows; r++)
//	{
//		for (int lr = middle; lr > 0; lr--)
//		{
//			//if (!all_output) { cout << "BEFORE -- (r, lr)= (" << r << "," << lr << ")" << "\n"; }
//			//if (!all_output) { cout << "BEFORE -- (double)lrImage.at<uchar>(r, lr)= " << (double)lrImage.at<uchar>(r, lr) << "\n"; }
//			//cout << "lr= " << lr  << "\n";
//			newMat.ptr<uchar>(r)[lr] = 0;
//			circle(nadirRemoveImage, Point(r, lr), 1, Scalar(255, 0, 0), 1, 8, 0);
//			//lrImage.at<uchar>(r,middle) = 0;
//			if (lr == (middle - 5))
//			{
//				lrImage.at<uchar>(middle,r) = 0;
//			}
//			//if ((double)lrImage.at<uchar>(r, lr) >= value)
//			//{
//			///*	if (!all_output) { cout << "(r, lr)= (" << r << "," << lr << ")" << "\n"; }
//			//	if (!all_output) { cout << "(double)lrImage.at<uchar>(r, lr)= " << (double)lrImage.at<uchar>(r, lr) << "\n"; }*/
//			//	lrImage.at<uchar>(r, middle) = 0;
//			//}
//			//else
//			//{
//			//if (!all_output) { cout << "NOT >= 128" << "\n"; }
//			//if (!all_output) { cout << "AFTER -- (r, lr)= (" << r << "," << lr << ")" << "\n"; }
//			//if (!all_output) { cout << "AFTER -- (double)lrImage.at<uchar>(r, lr)= " << (double)lrImage.at<uchar>(r, lr) << "\n"; }
//			//}
//		}
//
//		/*for (int rr = middle; rr < lrImage.size().width; rr++)
//		{
//		if (lrImage.at<uchar>(r, rr) >= value)
//		{
//		lrImage.at<uchar>(r, middle) = 0;
//		}
//		else
//		{
//
//		}
//		}*/
//
//
//	}
//	imshow("AFTER", nadirRemoveImage);
//	imshow("AFTER newMat", newMat);
//	testImage = lrImage;
//	//if (!all_output) { imshow("Nadir removed image", testImage); }
//#pragma endregion
//
//		waitKey(0);
//		return 0;
//}