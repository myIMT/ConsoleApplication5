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
//
//using namespace cv;
//using namespace std;
//
//int threshval = 60;
//int bw_constant = 128;
//vector<Vec4i> hierarchy;
//
//Mat contours;
//
//int main(int argc, char *argv[])
//{
//	contours = cv::imread("contour_1.bmp", CV_LOAD_IMAGE_UNCHANGED);
//
//	/// Find the rotated rectangles and ellipses for each contour
//	vector<RotatedRect> minRect(contours.size());
//	vector<RotatedRect> minEllipse(contours.size());
//
//	for (int i = 0; i < contours.size(); i++)
//	{
//		minRect[i] = minAreaRect(Mat(contours[i]));
//		if (contours[i].size() > 5)
//		{
//			minEllipse[i] = fitEllipse(Mat(contours[i]));
//		}
//	}
//
//	/// Draw contours + rotated rects + ellipses
//	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
//	for (int i = 0; i < contours.size(); i++)
//	{
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		// contour
//		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//		// ellipse
//		ellipse(drawing, minEllipse[i], color, 2, 8);
//		// rotated rectangle
//		Point2f rect_points[4]; minRect[i].points(rect_points);
//		for (int j = 0; j < 4; j++)
//			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
//	}
//
//	/// Show in a window
//	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//	imshow("Contours", drawing);
//}