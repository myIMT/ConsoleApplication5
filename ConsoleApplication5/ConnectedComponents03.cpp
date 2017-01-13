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

using namespace cv;
using namespace std;

const std::string keys =
"{help      |             | print this message    }"
"{@image    |contours     | load image            }"
"{j         |j.png        | j image        }"
"{contours  |contours.png | contours image        }"
;

int threshval = 60;
int bw_constant = 128;
vector<Vec4i> hierarchy;

int main(int argc, char *argv[])
{
	ofstream myConnectedComponents03file;
	myConnectedComponents03file.open("myConnectedComponents03file.txt");

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("");
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	cv::Mat srcImg;
	std::string file = parser.get<std::string>(0);
	if (file == "j") {
		file = parser.get<std::string>("j");
	}
	else if (file == "contours") {
		file = parser.get<std::string>("contours");
	}
	srcImg = cv::imread("20161215 02.33_368L.jpg", CV_LOAD_IMAGE_UNCHANGED);
	if (srcImg.empty()) {
		return -1;
	}

	string filter;		
	Mat fltrImg;
	cv::Mat fltrGrayImg;
	//for (int i = 0; i < 2; i++)
	//{
		//myConnectedComponents03file << "i= " << i << std::endl;
		//if (i==0)
		//{
		//	//Apply bilateral filter
		//	bilateralFilter(srcImg, fltrImg, 15, 80, 80);
		//	filter = "bilateralFilter result";
		//}
		//else if (i==1)
		//{
		//	//Apply median filter
		//	blur(srcImg, fltrImg, Size(5, 5), Point(-1, -1));
		//	filter = "median filter result";
		//}
		////else if (true)
		////{

		////}
		//else
		//{
		//	fltrImg = srcImg;
		//	filter = "Original";
		//}
		fltrImg = srcImg;
		filter = "src image";
		imshow(filter, fltrImg);
		imwrite(filter+".bmp", fltrImg);
		cv::cvtColor(fltrImg, fltrGrayImg, cv::COLOR_BGR2GRAY);
		//fltrGrayImg = srcImg;
		imshow("gray-scale image", fltrGrayImg);
		Mat FltrBinaryImg;
		cv::threshold(fltrGrayImg, FltrBinaryImg, 100, 255, cv::THRESH_BINARY);
		imshow("binary image", FltrBinaryImg);
		//cv::Mat FltrBinaryImg = threshval < 128 ? (fltrGrayImg < threshval) : (fltrGrayImg > threshval);
		cv::Mat FltrLabelImage;
		cv::Mat FltrStats, FltrCentroids;
		int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 4, CV_32S);
		//normalize(nFltrLabels, FltrLabelImage, 0, 255, NORM_MINMAX, CV_8U);
		cv::Mat FltrLabelImage2=FltrLabelImage;
		normalize(FltrLabelImage, FltrLabelImage2, 0, 255, NORM_MINMAX, CV_8U);
		//imshow("Labels", FltrLabelImage);
		//myConnectedComponents03file << "nFltrLabels= " << nFltrLabels << std::endl;
		//myConnectedComponents03file << "size of original image= " << fltrGrayImg.size() << std::endl;
		//myConnectedComponents03file << "size of FltrLabelImage= " << FltrLabelImage.size() << std::endl;
		//imshow("FltrLabelImage2", FltrLabelImage2);
		std::vector<cv::Vec3b> FltrColors(nFltrLabels);
		FltrColors[0] = cv::Vec3b(0, 0, 0);
		myConnectedComponents03file << "(Filter) Number of connected components = " << nFltrLabels << std::endl << std::endl;
		vector<vector<Point>> contours;
		//vector<Vec4i> hierarchy;
		for (int FltrLabel = 1; FltrLabel < nFltrLabels; ++FltrLabel) {
			FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
			myConnectedComponents03file << "Component " << FltrLabel << std::endl;
			myConnectedComponents03file << "CC_STAT_LEFT   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
			myConnectedComponents03file << "CC_STAT_TOP    = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
			myConnectedComponents03file << "CC_STAT_WIDTH  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
			myConnectedComponents03file << "CC_STAT_HEIGHT = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
			myConnectedComponents03file << "CC_STAT_AREA   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
			myConnectedComponents03file << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;

			// Get the mask for the i-th contour
			Mat mask_i = FltrLabelImage == FltrLabel;
			string name = "mask_i_";
			std::string s = std::to_string(FltrLabel);
			//strcat("mask_i_", std::to_string(i));
			if (FltrLabel==1)
			{
			imshow("mask_i_" + s, mask_i);
			}

			//=======================================================================
			findContours(mask_i.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			
			/// Draw contours
			RNG rng(12345);
			Mat drawing = Mat::zeros(fltrGrayImg.size(), CV_8UC3);

			///Find the rotated rectangles and ellipses for each contour
			//vector<RotatedRect> minRect(contours.size());
			vector<RotatedRect> minEllipse(contours.size());
			ofstream ContourMinEllipseMatrixFile;
			ContourMinEllipseMatrixFile.open("Contour_" + s + "MinEllipseMatrixFile.csv");
			for (int i = 0; i < contours.size(); i++)
			{
				//random colour for contour
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				// contour
				drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
				Mat test = Mat(contours[i]);
				minEllipse[i] = minAreaRect(test);

				ContourMinEllipseMatrixFile << minEllipse[i].angle << "\n";
				//minRect[i] = minAreaRect(Mat(contours[i]));
				/*minEllipse[i] = fitEllipse(drawing);*/
				//// ellipse
				//ellipse(drawing, minEllipse[i], color, 2, 8);
				//// rotated rectangle
				//Point2f rect_points[4]; minRect[i].points(rect_points);
				//for (int j = 0; j < 4; j++)
				//	line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
				//(x, y), (MA, ma), angle = cv2.fitEllipse(cnt);
			}
			ContourMinEllipseMatrixFile.close();
			//ofstream fitEllipseContourMatrixFile;
			//fitEllipseContourMatrixFile.open("fitEllipseContour_" + s + "MatrixFile.txt");
			//for (int i = 0; i < minRect.size(); i++)
			//{
			//	minRect[i] = minAreaRect(Mat(contours[i]));
			//	fitEllipseContourMatrixFile << "minRect[" << i << "].angle = " << minRect[i].angle << "\n";
			//	fitEllipseContourMatrixFile << "minRect[" << i << "].boundingRect = " << minRect[i].boundingRect() << "\n";
			//	fitEllipseContourMatrixFile << "minRect[" << i << "].center = " << minRect[i].center << "\n";
			//	//fitEllipseContourMatrixFile << "minRect[" << i << "].points = " << minRect[i].points << "\n";

			//}

			//for (int i = 0; i < minEllipse.size(); i++)
			//{
			//	minEllipse[i] = fitEllipse(Mat(contours[i]));
			//	fitEllipseContourMatrixFile << "minEllipse[" << i << "].angle = " << minEllipse[i].angle << "\n";
			//	fitEllipseContourMatrixFile << "minEllipse[" << i << "].boundingRect = " << minEllipse[i].boundingRect() << "\n";
			//	fitEllipseContourMatrixFile << "minEllipse[" << i << "].center = " << minEllipse[i].center << "\n";
			//	//fitEllipseContourMatrixFile << "minEllipse[" << i << "].points = " << minEllipse[i].points << "\n";
			//	fitEllipseContourMatrixFile << "\n" << "\n";
			//}
			//fitEllipseContourMatrixFile.close();

			imwrite("contour_" + s + ".bmp", drawing);

			//(x, y), (MA, ma), angle = cv2.fitEllipse(cnt);

			//if (FltrLabel==1)
			//{
				//imshow("mask_i Contour_"+s+" Image", drawing);

				cout << "hierarchy size= " << hierarchy.size() << "\n";
				ofstream ContourHierarchyMatrixFile;
				ContourHierarchyMatrixFile.open("Contour_"+s+"HierarchyMatrixFile.txt");
				ContourHierarchyMatrixFile << "[Next, Previous, First_Child, Parent]" << endl;
				for (int k = 0; k<hierarchy.size(); k++)
				{
					ContourHierarchyMatrixFile << hierarchy[k] << endl;

				}
				ContourHierarchyMatrixFile.close();
			////}


			ofstream ContourMatrixFile;	
			ContourMatrixFile.open("ContourMatrixFile_" + s + ".csv");
			ContourMatrixFile << drawing << "\n";
			ContourMatrixFile.close();
			//======================================
			myConnectedComponents03file << "mask_i_" + s << "\n";
			//imwrite("mask_i_"+s+".png", mask_i);
			//myConnectedComponents03file << "i= " << i << std::endl;
			//myConnectedComponents03file << "FltrLabelImage.size= " << FltrLabelImage.size() << std::endl;
			//myConnectedComponents03file << "mask_i= " << mask_i.size() << std::endl;
			// Compute the contour
			//findContours(mask_i, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			//findContours(mask_i, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		}
		cv::Mat FltrDst(fltrGrayImg.size(), CV_8UC3);
		//cv::imshow("filterDisplayName", FltrDst);
		for (int r = 0; r < FltrDst.rows; ++r) {
			for (int c = 0; c < FltrDst.cols; ++c) {
				int FltrLabel = FltrLabelImage.at<int>(r, c);
				cv::Vec3b &FltrPixel = FltrDst.at<cv::Vec3b>(r, c);
				FltrPixel = FltrColors[FltrLabel];
			}
		}
		cv::imshow("Connected Components", FltrDst);
		imwrite("Connected Components.bmp", FltrDst);


		//string filterDisplayName = filter + " Connected Components";
		////cv::imshow(filterDisplayName, FltrDst);
		//Mat1i labels;
		//int n_labels = connectedComponents(img, labels);
		//vector<vector<Point>> contours;
		//vector<Vec4i> hierarchy;
		//for (int i = 1; i < 2; ++i)
		//{
		//	// Get the mask for the i-th contour
		//	Mat mask_i = FltrLabelImage == i;
		//	string name = "mask_i_";
		//	std::string s = std::to_string(i);
		//	//strcat("mask_i_", std::to_string(i));
		//	imshow("mask_i_" + s, mask_i);
		//	//imwrite("mask_i.bmp", mask_i);
		//	myConnectedComponents03file << "i= " << i << std::endl;
		//	myConnectedComponents03file << "FltrLabelImage.size= " << FltrLabelImage.size() << std::endl;
		//	myConnectedComponents03file << "mask_i= " << mask_i.size() << std::endl;
		//	// Compute the contour
		//	//findContours(mask_i, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		//	findContours(mask_i, contours,hierarchy, RETR_EXTERNAL , CHAIN_APPROX_NONE);
		//	//int largest_contour_index = 0;
		//	//Rect bounding_rect;
		//	//for (int i = 0; i< contours.size(); i++)
		//	//{
		//	//	//  Find the area of contour
		//	//	double a = contourArea(contours[i], false);
		//	//	drawContours(srcImg, contours, largest_contour_index, (0, 0, 255), CV_FILLED, 8, hierarchy);
		//	//	rectangle(srcImg, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
		//	//	//namedWindow(name, CV_WINDOW_AUTOSIZE);
		//	//	string name2 = name + "contour";
		//	//	imshow(name2, srcImg);
		//	//	//imwrite(name + ".bmp", src);
		//	//	//if (a>largest_area) {
		//	//	//	largest_area = a; cout << i << " area  " << a << endl;
		//	//	//	// Store the index of largest contour
		//	//	//	largest_contour_index = i;
		//	//	//	// Find the bounding rectangle for biggest contour
		//	//	//	bounding_rect = boundingRect(contours[i]);

		//	//	//}
		//	//}
		//	////if (!contours.empty())
		//	////{
		//	////	// The first contour (and probably the only one)
		//	////	// is the one you're looking for

		//	////	// Compute the perimeter
		//	////	double perimeter_i = contours[0].size();
		//	////}
		//}
	//}

	////Apply bilateral filter
	//Mat srcImg11;
	//bilateralFilter(srcImg, srcImg11, 15, 80, 80);
	//imshow("source", srcImg);
	//imshow("bilateralFilter result", srcImg11);
	////Apply median filter
	//Mat srcImg22;
	//blur(srcImg, srcImg22, Size(5, 5), Point(-1, -1));
	//imshow("median filter result", srcImg22);

	//cv::Mat grayImg;
	//cv::Mat grayImg11;
	//cv::Mat grayImg22;
	//cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(srcImg11, grayImg11, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(srcImg22, grayImg22, cv::COLOR_BGR2GRAY);

	//cv::Mat binaryImg = threshval < 128 ? (grayImg < threshval) : (grayImg > threshval);
	//cv::Mat binaryImg11 = threshval < 128 ? (grayImg11 < threshval) : (grayImg11 > threshval);
	//cv::Mat binaryImg22 = threshval < 128 ? (grayImg22 < threshval) : (grayImg22 > threshval);
	//cv::Mat labelImage;
	//cv::Mat stats,centroids;
	//    int nLabels = cv::connectedComponents(bw, labelImage, 8);
	//int nLabels = cv::connectedComponentsWithStats(binaryImg, labelImage, stats, centroids, 8, CV_32S);
	//int nLabels11 = cv::connectedComponentsWithStats(binaryImg11, labelImage11, stats11, centroids11, 8, CV_32S);
	//int nLabels22 = cv::connectedComponentsWithStats(binaryImg22, labelImage22, stats22, centroids22, 8, CV_32S);
	//std::vector<cv::Vec3b> colors(nLabels);
	//std::vector<cv::Vec3b> colors11(nLabels11);
	//std::vector<cv::Vec3b> colors22(nLabels22);
	//colors[0] = cv::Vec3b(0, 0, 0);
	//colors11[0] = cv::Vec3b(0, 0, 0);
	//colors22[0] = cv::Vec3b(0, 0, 0);
	//myConnectedComponents03file << "Number of connected components = " << nLabels << std::endl << std::endl;
	//myConnectedComponents03file << "(11) Number of connected components = " << nLabels11 << std::endl << std::endl;
	//myConnectedComponents03file << "(22) Number of connected components = " << nLabels22 << std::endl << std::endl;

	//for (int label = 1; label < nLabels; ++label) {
	//	colors[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
	//	myConnectedComponents03file << "Component " << label << std::endl;
	//	myConnectedComponents03file << "CC_STAT_LEFT   = " << stats.at<int>(label, cv::CC_STAT_LEFT) << std::endl;
	//	myConnectedComponents03file << "CC_STAT_TOP    = " << stats.at<int>(label, cv::CC_STAT_TOP) << std::endl;
	//	myConnectedComponents03file << "CC_STAT_WIDTH  = " << stats.at<int>(label, cv::CC_STAT_WIDTH) << std::endl;
	//	myConnectedComponents03file << "CC_STAT_HEIGHT = " << stats.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;
	//	myConnectedComponents03file << "CC_STAT_AREA   = " << stats.at<int>(label, cv::CC_STAT_AREA) << std::endl;
	//	myConnectedComponents03file << "CENTER   = (" << centroids.at<double>(label, 0) << "," << centroids.at<double>(label, 1) << ")" << std::endl << std::endl;
	//}
//	for (int label11 = 1; label11 < nLabels11; ++label11) {
//		colors11[label11] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
///*		myConnectedComponents03file << "Component " << label11 << std::endl;
//		myConnectedComponents03file << "CC_STAT_LEFT   = " << stats.at<int>(label, cv::CC_STAT_LEFT) << std::endl;
//		myConnectedComponents03file << "CC_STAT_TOP    = " << stats.at<int>(label, cv::CC_STAT_TOP) << std::endl;
//		myConnectedComponents03file << "CC_STAT_WIDTH  = " << stats.at<int>(label, cv::CC_STAT_WIDTH) << std::endl;
//		myConnectedComponents03file << "CC_STAT_HEIGHT = " << stats.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;
//		myConnectedComponents03file << "CC_STAT_AREA   = " << stats.at<int>(label, cv::CC_STAT_AREA) << std::endl;
//		myConnectedComponents03file << "CENTER   = (" << centroids.at<double>(label, 0) << "," << centroids.at<double>(label, 1) << ")" << std::endl << std::endl;*/
//	}
//	for (int label22 = 1; label22 < nLabels22; ++label22) {
//		colors22[label22] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
//	}
	//cv::Mat dst(grayImg.size(), CV_8UC3);
	//for (int r = 0; r < dst.rows; ++r) {
	//	for (int c = 0; c < dst.cols; ++c) {
	//		int label = labelImage.at<int>(r, c);
	//		cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
	//		pixel = colors[label];
	//	}
	//}
	//cv::Mat dst11(grayImg11.size(), CV_8UC3);
	//for (int r = 0; r < dst11.rows; ++r) {
	//	for (int c = 0; c < dst11.cols; ++c) {
	//		int label11 = labelImage11.at<int>(r, c);
	//		cv::Vec3b &pixel11 = dst11.at<cv::Vec3b>(r, c);
	//		pixel11 = colors11[label11];
	//	}
	//}
	//cv::Mat dst22(grayImg22.size(), CV_8UC3);
	//for (int r = 0; r < dst22.rows; ++r) {
	//	for (int c = 0; c < dst22.cols; ++c) {
	//		int label22 = labelImage22.at<int>(r, c);
	//		cv::Vec3b &pixel22 = dst22.at<cv::Vec3b>(r, c);
	//		pixel22 = colors22[label22];
	//	}
	//}
	//cv::imshow("Source", srcImg);
	//cv::imshow("B&W" , binaryImg);
	//cv::imshow("Connected Components", dst);
	//cv::imshow("Source11", dst11);
	//cv::imshow("B&W11", binaryImg11);
	//cv::imshow("bilateralFilter Connected Components", FltrDst);
	//cv::imshow("median filter Connected Components", dst11);
	myConnectedComponents03file.close();

	waitKey(0);


	return 0;
}