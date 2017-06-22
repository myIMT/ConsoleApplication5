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
////----------------------------------------------------  
//#pragma endregion
//
//
//template <typename T>
//cv::Mat plotGraph(vector<T> & vals, int YRange[2])
//{
//	//vector<vector<T> > multilevel(4);
//
//	//for (int j = 0; j < vals.size; j++)
//	//{
//	//	auto it = minmax_element(vals.begin(), vals.end());
//	//	float scale = 1. / ceil(*it.second - *it.first);
//	//	float bias = *it.first;
//	//}
//	
//	
//		auto it = minmax_element(vals.begin(), vals.end());
//		float scale = 1. / ceil(*it.second - *it.first);
//		float bias = *it.first;
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
//	//To pick up pixels under line
//	cv::LineIterator it(src_gray, pt1, pt2, 8);
//	//To store pixel picked up, from under line
//	std::vector<double> buf(it.count);
//	Mat temp = src_gray;
//	LineIterator it2 = it;
//
//	vector<int> numbers(it.count);
//
//	for (int i = 0; i < it.count; i++, ++it)
//	{
//		double val = (double)src_gray.at<uchar>(it.pos());
//		buf[i] = val;
//		numbers.push_back(val);
//		cout << "position= " << it.pos() << ",  value= " << val << "\n";
//
//	}
//	//000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
//	//if (remove("Profile_DataFile.txt") != 0)
//	//	perror("Error deleting file");
//	//else
//	//	puts("File successfully deleted");
//
//	//				ofstream Profile_DataFile;
//	//				Profile_DataFile.open("Profile_DataFile.txt");
//	//				Profile_DataFile << Mat(numbers) << "\n";
//	//							//ContainerFile.open("ContainerFile_" + smi + ".txt");
//	//							//for (int i = 0; i < numbers.size(); i++)
//	//							//{
//	//							//	Profile_DataFile << "Profile[" << i << "].value= " << numbers.at(i) << "\n";
//	//							//}
//	//				Profile_DataFile.close();
//	//000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
//#pragma region Plot-Mat
//	/*vector<int> numbers(100);*/
//	//std::iota(numbers.begin(), numbers.end(), 0);
//
//	int range[2] = { 0, it.count };
//	cv::Mat lineGraph = plotGraph(numbers, range);
//
//	imshow("plot", lineGraph);
//#pragma endregion
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
//	//double angle_value_R = atan(angle_value_D*(PI/180));
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
//#pragma region ExtractNadirAreas
//void ExtractNadirAreas(vector<int> a)
//{
//	//if (all_output){ cout << "size of a = " << a.size() << endl;};
//	sort(a.begin(), a.end());	/// Sort vector of components-areas
//	reverse(a.begin(), a.end());	/// Reverse sorted vector of components-areas
//	//if (all_output){ cout << "a-vector-sorted with +4= " << endl;};	/// Print sorted-reversed component-areas
//	for (size_t i = 0; i != a.size(); ++i)
//		//if (all_output){ cout << a[i] << " ";};
//	//if (all_output){ cout << "\n"; //if (all_output){ cout << "\n"; //if (all_output){ cout << "\n";}
//	secondBiggestArea = a[1];	//if (all_output){ cout << "second biggest = " << a[1] << "\n";}
//	//secondBiggestAreaIndex = a.size()-1;	//if (all_output){ cout << "second biggest Index = " << a.size() - 1 << "\n";}
//	thirdBiggestArea = a[2];	//if (all_output){ cout << "third biggest = " << a[2] << "\n";}
//	//thirdBiggestAreaIndex = a.size() - 2;	//if (all_output){ cout << "third biggest Index = " << a.size() - 2 << "\n";}
//}
//#pragma endregion
//
///// <summary>                                                            
///// Calculates components                           
///// </summary>                                                           
///// <param name="GrayScaleSrcImg">Grayscale image (Mat) of original image.</param> 
//Mat GetConnectedComponent(Mat GrayScaleSrcImg)
//{
//	if (all_output){ cout << "03 - CALCULATING COMPONENTS ON BLURRED GRAYSCALE SOURCE IMAGE ('GrayImg')" << "\n";}
//
//	cv::Mat FltrBinaryImg = threshval < 128 ? (GrayScaleSrcImg < threshval) : (GrayScaleSrcImg > threshval);
//	if (all_output){ cout << "04 - CONVERTING BLURRED GRAYSCALE SOURCE IMAGE TO BINARY: 'FltrBinaryImg'" << "\n";}
//
//	cv::Mat FltrLabelImage;
//	cv::Mat FltrStats, FltrCentroids;
//
//	int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
//	if (all_output){ cout << "05 - CALCULATING CONNECTEDCOMPONENTSWITHSTATS ON 'FltrBinaryImg': 'FltrLabelImage'" << "\n";}
//#pragma region connectedComponentsWithStats
//	//ofstream connectedComponentsWithStats;
//	//connectedComponentsWithStats.open("connectedComponentsWithStats.txt");
//	if (all_output){ cout << "06 - WRITING CONNECTEDCOMPONENTSWITHSTATS RESULTS TO FILE: 'connectedComponentsWithStats.txt'" << "\n";}
//	//if (all_output){ imshow("Labels", FltrLabelImage);
//	//connectedComponentsWithStats << "nFltrLabels= " << nFltrLabels << std::endl;
//	//connectedComponentsWithStats << "size of original image= " << FltrBinaryImg.size() << std::endl;
//	//connectedComponentsWithStats << "size of FltrLabelImage= " << FltrLabelImage.size() << std::endl;
//	if (all_output) { imshow("FltrLabelImage2", FltrLabelImage); }
//	std::vector<cv::Vec3b> FltrColors(nFltrLabels);
//	FltrColors[0] = cv::Vec3b(0, 0, 0);
//	//connectedComponentsWithStats << "(Filter) Number of connected components = " << nFltrLabels << std::endl << std::endl;
//	//vector<vector<Point>> contours;
//	//vector<Vec4i> hierarchy;
//	vector<int> a;
//
//	for (int FltrLabel = 0; FltrLabel < nFltrLabels; ++FltrLabel) {
//		FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
//		CompStats.push_back(connectedComponentsWithStatsStruct());												/*Initialise container*/
//		
//		//connectedComponentsWithStats << "Component " << FltrLabel << std::endl;
//		CompStats[CompStructCount].COMP_STRUCT_NR = FltrLabel;
//
//		//connectedComponentsWithStats << "CC_STAT_LEFT   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
//		CompStats[CompStructCount].CC_STAT_LEFT = FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT);
//
//		//connectedComponentsWithStats << "CC_STAT_TOP    = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
//		CompStats[CompStructCount].CC_STAT_TOP = FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP);
//
//		//connectedComponentsWithStats << "CC_STAT_WIDTH  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
//		CompStats[CompStructCount].CC_STAT_WIDTH = FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH);
//
//		//connectedComponentsWithStats << "CC_STAT_HEIGHT = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
//		CompStats[CompStructCount].CC_STAT_HEIGHT = FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT);
//
//		//connectedComponentsWithStats << "CC_STAT_AREA   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
//		CompStats[CompStructCount].CC_STAT_AREA = FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA);
//
//		//connectedComponentsWithStats << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;
//		CompStats[CompStructCount].CC_STAT_CENTROID_X = FltrCentroids.at<double>(FltrLabel, 0);
//		CompStats[CompStructCount].CC_STAT_CENTROID_Y = FltrCentroids.at<double>(FltrLabel, 1);
//
//		a.push_back(FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA));
//
//		CompStructCount++;
//	}
//	if (all_output){ cout << "07 - STORING EACH COMPONENT'S AREA IN (GLOBAL DECLARED) VECTOR: 'a'" << "\n";}
//#pragma endregion
//	//connectedComponentsWithStats.close();
//
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
//#pragma endregion
//
//	imwrite("Grayscale_image_without_Nadir.bmp", croppedImage);
//	src_gray = croppedImage;
//
//	Mat FltrLabelImage2;
//	Mat FltrStats2, FltrCentroids2;
//
//	GrayScaleCroppedImage = croppedImage;
//
////#pragma region Histogram Equilization
////	char* source_window = "Source image";
////	char* equalized_window = "Equalized Image";
////	Mat dstHE;
////	/// Convert to grayscale
////	//cvtColor(src, src, CV_BGR2GRAY);
////
////	/// Apply Histogram Equalization
////	equalizeHist(GrayScaleCroppedImage, dstHE);
////
////	/// Display results
////	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
////	namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);
////
////	imshow(source_window, GrayScaleCroppedImage);
////	imshow(equalized_window, dstHE);
////#pragma endregion
//
//	Mat FltrBinaryImg2 = threshval < 128 ? (GrayScaleCroppedImage < threshval) : (GrayScaleCroppedImage > threshval);
//	imwrite("B&W_image_without_Nadir.bmp", FltrBinaryImg2);
//	nFltrLabels2 = cv::connectedComponentsWithStats(FltrBinaryImg2, FltrLabelImage2, FltrStats2, FltrCentroids2, 8, CV_32S);
//
//	if (all_output){ cout << "08 - ATTEMPT TO READ SOURCE IMAGE AND SPLIT IT IN TWO HALVES" << "\n";}
//	if (all_output){ cout << "09 -  -- ASSUMING TWO BLACK STRIPS ARE ALWAYS THE 2ND AND 3RD LARGEST COMPONENTS" << "\n";}
//
//	std::string nFltrLabels2String = std::to_string(nFltrLabels2);
//
//	cv::Mat FltrLabelImage3;
//
//	FltrLabelImage3 = FltrLabelImage2;
//	//normalize(FltrLabelImage2, FltrLabelImage3, 0, 255, NORM_MINMAX, CV_8U);
//	//if (all_output){ imshow("FltrLabelImage2", FltrLabelImage2);
//
//	//std::vector<cv::Vec3b> FltrColors(nFltrLabels);
//	//FltrColors[0] = cv::Vec3b(0, 0, 0);;
//	if (all_output){ cout << "10 - STORE ALL COMPONENTS'S IN (GLOBALLY DECLARED) VECTOR: 'maskImages'" << "\n";}
//	if (all_output){ cout << "11 - STORE ALL COMPONENTS'S CENTROID IN (GLOBALLY DECLARED) VECTOR: 'maskCentroid'" << "\n";}
//
//	//AngleTest_DataFile.open("AngleTest_DataFile.txt", ios::app);
//	//ofstream masksVector_DataFile;
//	//masksVector_DataFile.open("masksVector_DataFile.txt", ios::app);
//	std::vector<cv::Vec3b> FltrColors2(nFltrLabels2);
//	FltrColors2[0] = cv::Vec3b(0, 0, 0);
//
//	for (int FltrLabel2 = 1; FltrLabel2 < nFltrLabels2; ++FltrLabel2) 
//	{
//		std::string mask_index = std::to_string(FltrLabel2);
//		FltrColors2[FltrLabel2] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
//		//FltrColors[FltrLabel] = cv::Vec3b((255), (255), (255));
//		Mat mask_i = FltrLabelImage3 == FltrLabel2;
//		if (mask_i.empty())      // please, *always check* resource-loading.
//		{
//			cerr << "mask_i is empty - can't be loaded!" << endl; 
//			continue;
//		}
//		//maskImages.push_back(mask_i);
//		imwrite("mask_i" + mask_index + ".bmp", mask_i);
//		maskCentroid.push_back(Point(FltrCentroids.at<double>(FltrLabel2, 0), FltrCentroids2.at<double>(FltrLabel2, 1)));
//	}
//	//
//	cv::Mat FltrDst2(croppedImage.size(), CV_8UC3);
//	for (int r = 0; r < FltrDst2.rows; ++r) {
//		for (int c = 0; c < FltrDst2.cols; ++c) {
//			int FltrLabel2 = FltrLabelImage3.at<int>(r, c);
//			cv::Vec3b &FltrPixel2 = FltrDst2.at<cv::Vec3b>(r, c);
//			FltrPixel2 = FltrColors2[FltrLabel2];
//		}
//	}
//	if (!all_output) {imshow(nFltrLabels2String + "-Connected Components", FltrDst2);}
//	if (!all_output) { imwrite(nFltrLabels2String + "-Connected Components.bmp", FltrDst2); }
//
//
//	return FltrDst2;
//	if (all_output){ cout << "12 - RETURNING IMAGE OF ALL COMPONENTS: 'FltrDst'" << "\n";}
//}
//
////Mat RemoveNadir(Mat GrayScaleSrcImg)
////{
////	Mat BinaryNadirImage = threshval < 128 ? (GrayScaleSrcImg < threshval) : (GrayScaleSrcImg > threshval);
////
////	if (!all_output) { cout << "BinaryNadirImage.size().width= " << BinaryNadirImage.size().width << "\n"; }
////
////	double c;
////	double cc;
////
////	if (BinaryNadirImage.size().width % 2 == 0)
////	{
////		//is even
////		c = BinaryNadirImage.size().width / 2;
////		cc = c + 1;
////	}
////	else
////	{
////		double f = floor(BinaryNadirImage.size().width);
////		c = f + 1;
////		cc = 0;
////	}
////
////
////	double nadirCenterX = BinaryNadirImage.size().width / 2; 
////	Point nadirCenterPoint = Point(nadirCenterX, 1);
////	if (!all_output) { cout << "NadirCenter point= " << nadirCenterX << "\n"; }
////	Mat tempNadirSrc = imread(file);
////	circle(tempNadirSrc, nadirCenterPoint, 1, Scalar(255, 0, 0), 1, 8, 0);
////	imshow("Centre of Nadir", tempNadirSrc);
////
////	//for (size_t x = 0; j < BinaryNadirImage.cols; j++)
////	//{
////	//	/*				ComponentsLoop << "			Walk along Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";*/
////	//	//Bin_Analysis << "Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";
////	//	///if cannyEdge pixel intensity id non-zero
////	//	if ((int)cannyEdge.at<uchar>(i, j) != 0)
////	//	{
////
////	//	}
////	//}
////	return tempNadirSrc;
////}
//
//#pragma region filters
//// 1
//
///** Global Variables */
//int alpha = 100;
//int beta = 100;
//int gamma_cor = 100;
//Mat img_original, img_corrected, img_gamma_corrected;
//
//void basicLinearTransform(const Mat &img, const double alpha_, const int beta_)
//{
//	Mat res;
//	img.convertTo(res, -1, alpha_, beta_);
//
//	hconcat(img, res, img_corrected);
//}
//
//void gammaCorrection(const Mat &img, const double gamma_)
//{
//	CV_Assert(gamma_ >= 0);
//	//![changing-contrast-brightness-gamma-correction]
//	Mat lookUpTable(1, 256, CV_8U);
//	uchar* p = lookUpTable.ptr();
//	for (int i = 0; i < 256; ++i)
//		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
//
//	Mat res = img.clone();
//	LUT(img, lookUpTable, res);
//	//![changing-contrast-brightness-gamma-correction]
//
//	hconcat(img, res, img_gamma_corrected);
//}
//
//void on_linear_transform_alpha_trackbar(int, void *)
//{
//	double alpha_value = alpha / 100.0;
//	int beta_value = beta - 100;
//	basicLinearTransform(img_original, alpha_value, beta_value);
//}
//
//void on_linear_transform_beta_trackbar(int, void *)
//{
//	double alpha_value = alpha / 100.0;
//	int beta_value = beta - 100;
//	basicLinearTransform(img_original, alpha_value, beta_value);
//}
//
//void on_gamma_correction_trackbar(int, void *)
//{
//	double gamma_value = gamma_cor / 100.0;
//	gammaCorrection(img_original, gamma_value);
//}
//
//// 2
//
//void GammaCorrection(Mat& src, Mat& dst, float fGamma)
//{
//	CV_Assert(src.data);
//
//	// accept only char type matrices
//	CV_Assert(src.depth() != sizeof(uchar));
//
//	// build look up table
//	unsigned char lut[256];
//	for (int i = 0; i < 256; i++)
//	{
//		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
//	}
//
//	dst = src.clone();
//	const int channels = dst.channels();
//	switch (channels)
//	{
//	case 1:
//	{
//
//		MatIterator_<uchar> it, end;
//		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
//			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
//			*it = lut[(*it)];
//
//		break;
//	}
//	case 3:
//	{
//
//		MatIterator_<Vec3b> it, end;
//		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
//		{
//
//			(*it)[0] = lut[((*it)[0])];
//			(*it)[1] = lut[((*it)[1])];
//			(*it)[2] = lut[((*it)[2])];
//		}
//
//		break;
//
//	}
//	}
//}
//#pragma endregion
//
//
//
//
//
///// <summary>                                                            
///// Operations performed on each component.                             
///// </summary>                                                           
///// <param name="">xxx.</param> 
//int main(int argc, char *argv[])
//{
//
//	//cout_output = 0;
//	//file_output = 0;
//	//imshow_output = 0;
//	//imwrite_output = 0;
//	string file = "20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg";
//	//string file = "20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg";
//	//string file = "20161215 02.33_368.jpg";
//	//string file = "20140612_MINEGARDEN_SURVEY_00_14_50.jpg";
//
//	if (all_output){ cout << "--------------------------------------- START ---------------------------------------" << "\n";}
//	src = cv::imread(file);
//	if (src.empty())
//	{
//		//if (!all_output) { cout << "src image is null" << "\n"; }
//	}
//	else
//	{
//		//if (!all_output) { cout << "src image is NOT null" << "\n"; }
//	}
//	//src = cv::imread("20161215 02.33_368R2.jpg");
//	//src = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg");
//	//C:\Users\JW\Documents\Visual Studio 2015\Projects\ConsoleApplication5\ConsoleApplication5\20140612_Minegarden_Survey_SIDESCAN_Renavigated.jpg
//	//src = cv::imread("double ripple example_RR.jpg");
//	Mat tempSourceImage = src;
//	if (!all_output) { imshow("0 - Source Img", src); }
//	if (all_output){ cout << "00 - READING SOURCE IMAGE: 'src'" << "\n";}
//
//#pragma region Image pre-processing
//	////src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg");
//////src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg");
//////Mat src = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg"); 
//////if (all_output){ imshow("0 - src", src);
//////srcImg = src;
////bilateralFilter(src, srcImg, 15, 80, 80);
//blur(src, srcImg, Size(5, 5), Point(-1, -1));
////if (all_output){ cout << "01 - BLURRING SOURCE IMAGE: 'srcImg'" << "\n";}
//Mat tempFilteredImage = srcImg;
//if (!all_output) { imshow("1 - blur srcImg", srcImg); }
////cv::cvtColor(srcImg, GrayImg, cv::COLOR_BGR2GRAY);
////if (all_output){ cout << "02 - CONVERTING 'srcImg' TO GRAYSCALE " << "\n";}
////if (!all_output) { imshow("2 - Blur GrayImg", GrayImg); }  
//#pragma endregion
//
//
////#pragma region Apply Gamma-correction functions
////
////	Mat dst(src.rows, src.cols, src.type());
////	GammaCorrection(src, dst, 0.6);
////
////	imshow("2", dst);
////	cv::cvtColor(dst, GrayImg, cv::COLOR_BGR2GRAY);
////	if (!all_output) { imshow("2 - Filtered GrayImg", GrayImg); }
//////	img_original = src;
//////	img_corrected = Mat(img_original.rows, img_original.cols * 2, img_original.type());
//////	img_gamma_corrected = Mat(img_original.rows, img_original.cols * 2, img_original.type());
//////
//////	hconcat(img_original, img_original, img_corrected);
//////	hconcat(img_original, img_original, img_gamma_corrected);
//////
//////	namedWindow("Brightness and contrast adjustments", WINDOW_AUTOSIZE);
//////	namedWindow("Gamma correction", WINDOW_AUTOSIZE);
//////
//////	createTrackbar("Alpha gain (contrast)", "Brightness and contrast adjustments", &alpha, 500, on_linear_transform_alpha_trackbar);
//////	createTrackbar("Beta bias (brightness)", "Brightness and contrast adjustments", &beta, 200, on_linear_transform_beta_trackbar);
//////	createTrackbar("Gamma correction", "Gamma correction", &gamma_cor, 200, on_gamma_correction_trackbar);
//////
//////	while (true)
//////	{
//////		imshow("Brightness and contrast adjustments", img_corrected);
//////		imshow("Gamma correction", img_gamma_corrected);
//////
//////		int c = waitKey(30);
//////		if (c == 27)
//////			break;
//////	}
//////
//////	imwrite("linear_transform_correction.png", img_corrected);
//////	imwrite("gamma_correction.png", img_gamma_corrected);
////#pragma endregion
//
//	cvtColor(srcImg, GrayImg, CV_BGR2GRAY);
//
//	Mat linePixelsTempGraySrc3 = GrayImg;
//
//#pragma region Components and nadir removal
//	Mat components = GetConnectedComponent(GrayImg);
//#pragma endregion
//
//	//Mat RemovedNadirImage = RemoveNadir(GrayImg);
//	
////#pragma region Histogram Equilization
////	char* source_window = "Source image";
////	char* equalized_window = "Equalized Image";
////	Mat dstHE;
////	/// Convert to grayscale
////	//cvtColor(src, src, CV_BGR2GRAY);
////
////	/// Apply Histogram Equalization
////	equalizeHist(GrayScaleCroppedImage, dstHE);
////
////	/// Display results
////	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
////	namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);
////
////	imshow(source_window, GrayScaleCroppedImage);
////	imshow(equalized_window, dstHE);
////#pragma endregion
//
//	//if (all_output) { imshow("3 - components", components); }
//
//	//if (all_output){ imshow("maskImages[0]", Mat(maskImages[0]));
//	Mat tempSrc1 = imread(file, CV_LOAD_IMAGE_UNCHANGED);
//	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg"); 
//#pragma region MyRegion
//	if (all_output){ cout << "13 - STORING LOOPING-OVER-CANNY-IMAGES RESULTS IN TEXT FILE: 'ComponentsLoop.txt'" << "\n";}
//	//ofstream ComponentsLoop;
//	//ComponentsLoop.open("ComponentsLoop.txt");
//
//	//ofstream Bin_Analysis;
//	//Bin_Analysis.open("Bin_Analysis.csv");
//	if (all_output){ cout << "14 - STORING BIN ANALYSIS RESULTS IN TEXT FILE: 'Bin_Analysis.CSV'" << "\n";}
//
//	//ofstream ComponentAngle;
//	//ComponentAngle.open("ComponentAngle.txt");
//	ofstream AngleTest_DataFile;
//	//AngleTest_DataFile.open("AngleTest_DataFile.txt", ios::app);
//
//	int maskElements = nFltrLabels2 - 1;
//	///Loop through each component
//	for (size_t mi = 1; mi < maskElements; mi++)
//	{
//		std::string smi = std::to_string(mi);
//		//system("cls");
//		//if (!all_output) {cout << "component= " << mi;}
//		//testMaskiRead = imread("mask_i" + smi + ".bmp", CV_LOAD_IMAGE_UNCHANGED);
//
//		if (mi == 2440)
//		{
//			if (!all_output) { cout << "component= " << mi; }
//		}
//		if (true)
//		{
//		/*	ComponentsLoop << "		Component Nr. = " << mi << "\n";*/
//			//if (mi== 17 ||mi == 16 ||mi == 13 || mi == 9)
//			//{
//			Mat tempComponent = imread("mask_i" + smi + ".bmp", CV_LOAD_IMAGE_UNCHANGED);
//				//maskImages.at(mi);
//			if (all_output) { imshow("4 - component- " + smi, tempComponent); }
//			//if (all_output){ imwrite("4_component_" + smi + ".bmp", tempComponent);
//			Mat GrayComponents;
//			//cvtColor(tempComponent, GrayComponents, COLOR_BGR2GRAY);
//			//if (all_output){ imshow("GrayComponents", GrayComponents);
//			GrayComponents = tempComponent;
//			Sobel(GrayComponents, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//			Sobel(GrayComponents, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//
//			Mat Mag(GrayComponents.size(), CV_32FC1);
//			Mat Angle(GrayComponents.size(), CV_32FC1);
//			cartToPolar(grad_x, grad_y, Mag, Angle, true);
//			//////77777777777777777777777777777777777777777777777777777777777777777777777777777777777
//			//ofstream Angle_DataFile;
//			//Angle_DataFile.open("Angle_DataFile_" + smi + ".csv");
//			//Angle_DataFile << Angle << "\n";
//			//Angle_DataFile.close();
//			//////77777777777777777777777777777777777777777777777777777777777777777777777777777777777
//
//			std::array<std::vector<int>, 72> vvv{ {} };
//			struct element {
//				int bin;
//				int i;
//				int j;
//				int angle;
//				int value;
//			};
//			vector<element> container;
//			int containerCount = 0;
//
//			Canny(GrayComponents, cannyEdge, 100, 200);
//			if (all_output) { imshow("5 - component cannyEdge - " + smi, cannyEdge); }
//			//if (all_output){ imwrite("5_component_cannyEdge" + smi + ".bmp", cannyEdge);
//			//////8888888888888888888888888888888888888888888888888888888888888888888888888888888
//			//ofstream cannyEdge_DataFile;
//			//cannyEdge_DataFile.open("cannyEdge_DataFile_" + smi + ".csv");
//			//cannyEdge_DataFile << cannyEdge << "\n";
//			//cannyEdge_DataFile.close();
//			//////8888888888888888888888888888888888888888888888888888888888888888888888888888888
//
//			Mat newAngle = Mat(Angle.size().height, Angle.size().width, Angle.type(), Scalar(0, 0, 0));
//
//			///Walk along cannyEdge rows
//			for (size_t i = 0; i < cannyEdge.rows; i++)
//			{
//				///Walk along cannyEdge rows & columns
//				for (size_t j = 0; j < cannyEdge.cols; j++)
//				{
//	/*				ComponentsLoop << "			Walk along Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";*/
//					//Bin_Analysis << "Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";
//					///if cannyEdge pixel intensity id non-zero
//					if ((int)cannyEdge.at<uchar>(i, j) != 0)
//					{
//						//ComponentsLoop << "				Non-Zero Canny pixel value (coordinates - (x,y) ) = " << i << ", " << j << "\n";
//						//ComponentsLoop << "				Storing Non-Zero Canny pixel value in container" << "\n";
//
//						//Bin_Analysis << "Canny Edge Non-zero pixel (coordinates - (x,y) ) = " << i << ", " << j << "\n";
//
//						newAngle.ptr<float>(i)[j] = Angle.ptr<float>(i)[j];							/*Create new Angle matrix*/
//						container.push_back(element());												/*Initialise container*/
//						container[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);	/*Store bin (newAngle/5)*/
//						container[containerCount].i = i;											///Store row position
//						container[containerCount].j = j;											///Store column position
//						container[containerCount].angle = newAngle.ptr<float>(i)[j];				///Store new Angle value
//						container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);			///Store canny pixel intensity
//						containerCount++;
//					}
//				}
//			}
//			//ComponentsLoop << "Finished walking on Canny Edge" << "\n";
//			//ComponentsLoop << "\n";
//			//ComponentsLoop << "\n";
//
//			//////8888888888888888888888888888888888888888888888888888888888888888888888888888888
//			//ofstream newAngle_DataFile;
//			//newAngle_DataFile.open("newAngle_DataFile" + smi + ".csv");
//			//newAngle_DataFile << newAngle << "\n";
//			//newAngle_DataFile.close();
//			//////8888888888888888888888888888888888888888888888888888888888888888888888888888888
//
//			//Bin_Analysis.close();
//			//////999999999999999999999999999999999999999999999999999999
//			//ofstream ContainerFile;
//			//ContainerFile.open("ContainerFile_" + smi + ".txt");
//			for (int i = 0; i < container.size(); i++)
//			{
//				//ContainerFile << "container[" << i << "].bin= " << container[i].bin << "\n";
//				//ContainerFile << "container[" << i << "].i= " << container[i].i << "\n";
//				//ContainerFile << "container[" << i << "].j= " << container[i].j << "\n";
//				//ContainerFile << "container[" << i << "].angle= " << container[i].angle << "\n";
//				//ContainerFile << "container[" << i << "].value= " << container[i].value << "\n";
//				//ContainerFile << "\n";
//				//ContainerFile << "\n";
//			}
//			//ContainerFile.close();
//			//////999999999999999999999999999999999999999999999999999999
//			int maxCount = 0;
//			struct maxCountStruct {
//				int bin;
//				int angle;
//				int size;
//			};
//			vector<maxCountStruct> maxCountContainer;
//			int temp = 0;
//			struct MaxElementStruct {
//				int bin = 0;
//				int angle = 0;
//				int size = 0;
//			};
//			MaxElementStruct mes;
//
//			//ofstream KeepingTrackOfContainers_DataFile;
//			//KeepingTrackOfContainers_DataFile.open("KeepingTrackOfContainers_DataFile.csv");
//
//			/*ComponentsLoop << "			For every element in container (total size = " << container.size() << ")" << "\n";*/
//			/// Grouping bin values together and counting them
//			/// For every element in container
//			for (size_t l = 0; l < container.size(); l++)
//			{
//				/*ComponentsLoop << "			container[l] = " << l << "\n";*/
//				/// If new container is empty (at start)
//				if (maxCountContainer.empty())
//				{
//				/*	ComponentsLoop << "				Initial element (new container empty) in new container[l] = " << l << "\n";*/
//					maxCountContainer.push_back(maxCountStruct());		/// Initialise new container
//					maxCountContainer[l].bin = container[l].bin;		/// Store container bin value in new container bin field
//					maxCountContainer[l].angle = container[l].angle;	/// Store container angle value in new container angle field
//					maxCountContainer[l].size += 1;						/// Increment new container size field (counting elements with similair bin values)
//				}
//				else  /// If not at start (new container contains first element)
//				{
//					/// For every element in new container (new container contains at least one element thus far) & its elements will increase with every loop
//					for (size_t m = 0; m < maxCountContainer.size(); m++)
//					{
//						/*ComponentsLoop << "				For every element in new container[m] (as it's filling up ) = " << m << "\n";*/
//
//						//KeepingTrackOfContainers_DataFile << "container iterator (l): " << l << "\n";
//						//KeepingTrackOfContainers_DataFile << "maxCountContainer iterator (m)= " << m << "\n";
//						/// 
//						if (maxCountContainer[m].bin == container[l].bin)
//						{
//							//ComponentsLoop << "					When container element [l] already exist in new container [m] (don't re-add it, just update it's count ) = " << l << ", " << m << "\n";
//							maxCountContainer[m].size += 1;
//							break;
//						}
//						else if (m == maxCountContainer.size() - 1)
//						{
//							maxCountContainer.push_back(maxCountStruct());
//							maxCountContainer[maxCountContainer.size() - 1].bin = container[l].bin;
//							maxCountContainer[maxCountContainer.size() - 1].angle = container[l].angle;
//							maxCountContainer[maxCountContainer.size() - 1].size += 1;
//							break;
//						}
//
//						//////666666666666666666666666666666666666666666666666666666666666666666666666666666
//						//ofstream maxCountContainer_DataFile;
//						//maxCountContainer_DataFile.open("maxCountContainer_DataFile_" + smi + ".csv");
//						//for (int i = 0; i < maxCountContainer.size(); i++)
//						//{
//						//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].i= " << i << "\n";
//						//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].bin= " << maxCountContainer[i].bin << "\n";
//						//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].angle= " << maxCountContainer[i].angle << "\n";
//						//	maxCountContainer_DataFile << "maxCountContainer[" << i << "].size= " << maxCountContainer[i].size << "\n";
//						//	maxCountContainer_DataFile << "\n";
//						//	maxCountContainer_DataFile << "\n";
//						//}
//						//maxCountContainer_DataFile.close();
//						//maxCountContainer_DataFile.close();
//						//////666666666666666666666666666666666666666666666666666666666666666666666666666666
//
//				/*		ComponentsLoop << "		Adjusting element with the highest frequency" << "\n";*/
//						if (maxCountContainer[m].size > temp)	///Find bin with the most elements
//						{
//							temp = maxCountContainer[m].size;
//							mes.bin = (int)maxCountContainer[m].bin;	///Bin with most elements (bin ID)
//							mes.angle = (int)maxCountContainer[m].angle;
//							mes.size = (int)maxCountContainer[m].size;
//						}
//					/*	ComponentsLoop << "		Element with highest frequency (" << mes.size << "= " << mes.angle << "\n";*/
//					}
//				}
//			}
//			//KeepingTrackOfContainers_DataFile.close();
//			AngleTest_DataFile.open("AngleTest_DataFile.txt");
//			if (!all_output) { AngleTest_DataFile << mes.angle << endl; }
//			//if (!all_output) { AngleTest_DataFile << "The biggest number is: " << mes.size << " at bin " << mes.bin << endl; }
//			//if (!all_output){ AngleTest_DataFile << "Angle (mes)- " << smi << "= " << mes.angle << "\n";}
//			//ComponentAngle << "Angle (mes)- " << smi << "\n";
//			Mat tempGraySrc = GrayImg;
//			for (size_t n = 0; n < container.size(); n++)
//			{
//				if (container[n].bin == mes.bin)
//				{
//					tempGraySrc.at<uchar>(container[n].i, container[n].j) = 255;
//				}
//			}
//
//			//if (all_output){ imshow("tempGraySrc", tempGraySrc);
//			//if (all_output){ imwrite("tempGraySrc.bmp", tempGraySrc);
//
//			Mat tempSrc2 = imread("/images/20140612_Minegarden_Survey_SIDESCAN_Renavigated_R1.jpg", CV_LOAD_IMAGE_UNCHANGED);
//			//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//			//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//			//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01.jpg");
//#pragma region Bounding Box
//			vector<double> lengths(4);
//			double rectSize_b;
//			size_t imgCount = 0;
//			////if (all_output){ cout << "maskImages.size()= " << maskImages.size() << "\n";}
//			//for (imgCount; imgCount < maskImages.size(); imgCount++)
//			//{
//			Mat tempPoints;
//			findNonZero(imread("mask_i" + smi + ".bmp", CV_LOAD_IMAGE_UNCHANGED), tempPoints);
//			//if (!all_output) { cout << "tempPoints = " << tempPoints << "\n"; }
//			Points.push_back(tempPoints);
//			//}
//			Point2f vtx[4];
//			RotatedRect box = minAreaRect(Points[mi-1]); //only the first Mat Points
//			//if (!all_output){ AngleTest_DataFile << "RotatedRect box = minAreaRect(Points[mi])" << box.angle << "\n";}
//			//if (!all_output) { AngleTest_DataFile << "###########################################################" << "\n"; }
//			AngleTest_DataFile.close();
//			box.points(vtx);
//			/*Mat tempSrc1 = imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);*/
//			for (int i = 0; i < 4; i++)
//			{
//				line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
//				line(tempSrc2, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
//				lengths.push_back(norm((vtx[(i + 1) % 4]) - (vtx[i])));
//			}
//			if (all_output) { imshow("Bounding Box", tempSrc1); }
//			//if (all_output){ imwrite("Bounding_Box" + smi + ".bmp", tempSrc1);
//			////if (all_output){ cout << "minAreaRect Angle - "<<smi<<"= " << box.angle + 180 << "\n";}
//			//if (all_output){ cout << "minAreaRect width= " << box.size.width << "\n";}
//			//if (all_output){ cout << "minAreaRect height= " << box.size.height << "\n";}
//#pragma endregion
//			Mat plotImage = src;
//			circle(plotImage, maskCentroid.at(mi), 1, Scalar(0, 255, 0), 1, 8, 0);
//			circle(tempSrc2, maskCentroid.at(mi), 1, Scalar(0, 255, 0), 1, 8, 0);
//
//#pragma region walk in edge angle direction
//			Point2f u, u2, u22, v;
//			Point2f w1, w2;
//			//if (all_output){ cout << "cos((mes.angle)* CV_PI / 180.0)= " << cos((mes.angle)* CV_PI / 180.0) << "\n";}
//			//if (all_output){ cout << "sin((mes.angle)* CV_PI / 180.0)= " << sin((mes.angle)* CV_PI / 180.0) << "\n";}
//			u = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
//			u2 = u;
//			rectSize_b = *max_element(lengths.begin(), lengths.end());
//			double d = 0.1*rectSize_b;
//			double normU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
//			////if (all_output){ cout << "normU= " << normU << "\n";}
//			v = Point2f(u.x / normU, u.y / normU);
//			//Mat tempSrcW1 = src, tempSrcW2 = src;
//			for (size_t i = 0; i < 10; i++)
//			{
//				if (i == 0)
//				{	// starting point = center of mask
//					w1.x = maskCentroid.at(mi).x + v.x*d;	//one side
//					w1.y = maskCentroid.at(mi).y + v.y*d;
//
//					w2.x = maskCentroid.at(mi).x - v.x*d;	//other side
//					w2.y = maskCentroid.at(mi).y - v.y*d;
//				}
//				else
//				{	// points on either-side of mask center point
//					w1.x = u2.x + v.x*d;		//one side
//					w1.y = u2.y + v.y*d;
//
//					w2.x = u22.x - v.x*d;		//other side
//					w2.y = u22.y - v.y*d;
//				}
//				////if (all_output){ cout << "i - " << i << "2-Plot here= " << w1 << ", " << w2 << "\n";}
//				//circle(plotImage, w1, 1, Scalar(0, 0, 255), 1, 8, 0);
//				//circle(plotImage, w2, 1, Scalar(255, 0, 0), 1, 8, 0);
//
//				circle(tempSrc2, w1, 1, Scalar(55, 55, 55), 1, 8, 0);
//				circle(tempSrc2, w2, 1, Scalar(55, 55, 55), 1, 8, 0);
//				//circle(tempSrcW1, w1, 1, Scalar(0, 0, 0), 1, 8, 0);
//				//circle(tempSrcW2, w2, 1, Scalar(255, 255, 255), 1, 8, 0);
//				u2 = w1;
//				u22 = w2;
//			}
//			if (all_output) { imshow("walk in edge angle direction - component nr." + smi, tempSrc2); }
//			//rectangle(tempSrcW1, cv::Point2f(10, 10), cv::Point2f(src.size().width - 10, src.size().height - 10), cv::Scalar(255, 0, 0));
//			//if (all_output){ imshow("tempSrcW1- " + smi, tempSrcW1);
//			//if (all_output){ imwrite("tempSrcW1_" + smi + ".bmp", tempSrcW1);
//
//			//rectangle(tempSrcW2, cv::Point2f(10, 10), cv::Point2f(src.size().width - 10, src.size().height - 10), cv::Scalar(0, 255, 0));
//			//if (all_output){ imshow("tempSrcW2- " + smi, tempSrcW2);
//			//if (all_output){ imwrite("tempSrcW2_" + smi + ".bmp", tempSrcW2);
//#pragma endregion
//
//			struct buffer {
//				std::vector<double> pixValues;
//				Point2f startPoint;
//				Point2f endPoint;
//				//int j;
//				//int angle;
//				//int value;
//			};
//			vector<buffer> Profiles;
//			int ProfilesCount = 0;
//
//#pragma region walk perpendicular in edge angle direction
//			Point2f uu, uu2, uu22, vv, ep11, ep12, ep21, ep22;
//			Point2f ww1, ww2;
//			uu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
//			uu2 = uu;
//			rectSize_b = *max_element(lengths.begin(), lengths.end());
//			//double dd = 0.1*rectSize_b;
//			double normUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
//			vv = Point2f(uu.x / normUU, uu.y / normUU);
//			u = vv;
//			//rotate and swap
//			double tempXX = vv.x;
//			vv.x = -vv.y;
//			vv.y = tempXX;
//			int e = 20;
//
//			Mat tempGraySrc3;
//			cv::cvtColor(src, tempGraySrc3, cv::COLOR_BGR2GRAY);
//			Mat tempSrc3 = tempGraySrc;//imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//			//Mat linePixelsTempGraySrc3 = tempGraySrc3;
//			int i = -1;
//			/*std::string ii = std::to_string(i);*/
//			Mat imageArray[10];
//			bool beginning = true;
//
//			for (i = -1; i < box.size.width / 2; i++)
//			{
//				std::string ii = std::to_string(i);
//
//				if (i == -1)
//				{
//					ww1.x = maskCentroid.at(mi).x;// +vv.x*d;	//one side
//					ww1.y = maskCentroid.at(mi).y;// +vv.y*d;
//
//					ww2.x = maskCentroid.at(mi).x;// -vv.x*d;	//other side
//					ww2.y = maskCentroid.at(mi).y;// -vv.y*d;
//
//											   //end points of profile
//					ep11.x = ww1.x - ((box.size.width + e) / 2) * uu.x;
//					ep11.y = ww1.y - ((box.size.width + e) / 2) * uu.y;
//					ep12.x = ww1.x + ((box.size.width + e) / 2) * uu.x;
//					ep12.y = ww1.y + ((box.size.width + e) / 2) * uu.y;
//
//					ep21.x = ww2.x - ((box.size.width + e) / 2) * uu.x;
//					ep21.y = ww2.y - ((box.size.width + e) / 2) * uu.y;
//					ep22.x = ww2.x + ((box.size.width + e) / 2) * uu.x;
//					ep22.y = ww2.y + ((box.size.width + e) / 2) * uu.y;
//				}
//				else if (i == 0)
//				{	// starting point = center of mask
//					ww1.x = maskCentroid.at(mi).x + vv.x*d;	//one side
//					ww1.y = maskCentroid.at(mi).y + vv.y*d;
//
//					ww2.x = maskCentroid.at(mi).x - vv.x*d;	//other side
//					ww2.y = maskCentroid.at(mi).y - vv.y*d;
//
//					//end points of profile
//					ep11.x = ww1.x - ((box.size.width + e) / 2) * uu.x;
//					ep11.y = ww1.y - ((box.size.width + e) / 2) * uu.y;
//					ep12.x = ww1.x + ((box.size.width + e) / 2) * uu.x;
//					ep12.y = ww1.y + ((box.size.width + e) / 2) * uu.y;
//
//					ep21.x = ww2.x - ((box.size.width + e) / 2) * uu.x;
//					ep21.y = ww2.y - ((box.size.width + e) / 2) * uu.y;
//					ep22.x = ww2.x + ((box.size.width + e) / 2) * uu.x;
//					ep22.y = ww2.y + ((box.size.width + e) / 2) * uu.y;
//
//					beginning = false;
//				}
//				else
//				{	// points on either-side of mask center point
//					ww1.x = uu2.x + vv.x*d;		//one side
//					ww1.y = uu2.y + vv.y*d;
//
//					ww2.x = uu22.x - vv.x*d;	//other side
//					ww2.y = uu22.y - vv.y*d;
//
//					//end points of profile
//					ep11.x = ww1.x - ((box.size.width + e) / 2) * uu.x;
//					ep11.y = ww1.y - ((box.size.width + e) / 2) * uu.y;
//					ep12.x = ww1.x + ((box.size.width + e) / 2) * uu.x;
//					ep12.y = ww1.y + ((box.size.width + e) / 2) * uu.y;
//
//					ep21.x = ww2.x - ((box.size.width + e) / 2) * uu.x;
//					ep21.y = ww2.y - ((box.size.width + e) / 2) * uu.y;
//					ep22.x = ww2.x + ((box.size.width + e) / 2) * uu.x;
//					ep22.y = ww2.y + ((box.size.width + e) / 2) * uu.y;
//				}
//				circle(tempSrc2, ww2, 1, Scalar(255, 0, 0), 1, 8, 0); //turqoise
//				circle(tempSrc2, ww1, 1, Scalar(55, 0, 0), 1, 8, 0); //
//																		 //circle(tempSrc2, ww2, 1, Scalar(255, 255, 10), 1, 8, 0); //
//				circle(tempSrc2, ep11, 1, Scalar(0, 0, 255), 1, 8, 0); //
//				circle(tempSrc2, ep12, 1, Scalar(0, 0, 255), 1, 8, 0); //
//				circle(tempSrc2, ep21, 1, Scalar(0, 0, 255), 1, 8, 0); //
//				circle(tempSrc2, ep22, 1, Scalar(0, 0, 255), 1, 8, 0); //
//
//				uu2 = ww1;
//				uu22 = ww2;
//
//#pragma region DrawLines
//				int thickness = 0.2;
//				int lineType = 8;
//				line(tempGraySrc3,
//					Point(ep11.x, ep11.y),
//					Point(ep12.x, ep12.y),
//					Scalar(255, 0, 0),
//					thickness,
//					lineType);
//
//				profileLinesCount += 1;
//				std::string profileLinesCount1 = std::to_string(profileLinesCount);
//				if (all_output) { imshow("Profile Line-" + profileLinesCount1, tempGraySrc3); }
//
//				line(tempGraySrc3,
//					Point(ep21.x, ep21.y),
//					Point(ep22.x, ep22.y),
//					Scalar(255, 0, 0),
//					thickness,
//					lineType);
//
//				profileLinesCount += 1;
//				std::string profileLinesCount2 = std::to_string(profileLinesCount);
//				if (all_output) { imshow("Profile Line-" + profileLinesCount2, tempGraySrc3); }
//#pragma endregion
//
//#pragma region LinePixels
//
//				// grabs pixels along the line (pt1, pt2)
//				// from 8-bit 3-channel image to the buffer
//				LineIterator it1B(tempGraySrc3, Point(ep11), Point(ep12), 8);		// Lines before centroid
//				LineIterator it1A(tempGraySrc3, Point(ep21), Point(ep22), 8);		// Lines after centroid
//				//LineIterator it2(tempSrc3, Point(ep21), Point(ep22), 8);
//				LineIterator it11B = it1B;
//				LineIterator it11A = it1A;
//				//LineIterator it22 = it2;
//				//vector<Vec3b> buf(it.count);
//
//				//ofstream file;
//				vector<float> pixelsOnLineB;
//				vector<float> pixelsOnLineA;
//				vector<vector<float>> pixels;
//
//				Mat linePixel;
//				//float pixelsUnderLine[it1.count];
//
//				// Record pixels under line B (Before centroid)
//				for (int l = 0; l < it1B.count; l++, ++it1B)
//				{
//					Profiles.push_back(buffer());
//					Profiles[ProfilesCount].startPoint = ep11;
//					Profiles[ProfilesCount].endPoint = ep12;
//					double valB = (double)linePixelsTempGraySrc3.at<uchar>(it1B.pos());
//					pixelsOnLineB.push_back(valB);
//					linePixel.push_back(valB);
//					//5555555555555555555555555555555555555555555555555555555555555555
//		/*			ofstream tempGraySrc3DataFile;
//					tempGraySrc3DataFile.open("linePixelsTempGraySrc3" + profileLinesCount1 + "B.csv");
//					tempGraySrc3DataFile << linePixelsTempGraySrc3 << "\n";
//					tempGraySrc3DataFile << "\n";
//					tempGraySrc3DataFile << "\n";
//					tempGraySrc3DataFile.close();*/
//					//55555555555555555555555555555555555555555555555555555555555555555555
//
//					////if (all_output){ cout << "Point(ep11.x, ep11.y), Point(ep12.x, ep12.y) = " << Point(ep11.x, ep11.y) << ", "<< Point(ep12.x, ep12.y) << "\n";
//					////if (all_output){ cout << "it1.pos() = " << Point(ep11.x, ep11.y) << ", " << it1B.pos() << "\n";
//					////if (all_output){ cout << "(double)tempGraySrc3.at<uchar>(it1.pos()) = " << (double)linePixelsTempGraySrc3.at<uchar>(it1B.pos()) << "\n";}
//					Profiles[ProfilesCount].pixValues.push_back(valB);// (double)tempSrc3.at<uchar>(it1.pos());
//
//					//double val = (double)src_gray.at<uchar>(it.pos());
//					//buf[i] = val;
//
//					//std::string L = std::to_string(l);
//					//file.open("buf_" + format("(%d,%d)", Profiles[ProfilesCount].startPoint, Profiles[ProfilesCount].endPoint) + ".csv", ios::app);
//					//file.open("buf_" + L + ".csv", ios::app);
//					//file << Profiles[ProfilesCount].startPoint << "\n";
//					//file << Profiles[ProfilesCount].endPoint << "\n";
//					//file << Mat(Profiles[ProfilesCount].pixValues) << "\n";
//					//file.close();
//					ProfilesCount += 1;
//				}
//
//				// Record pixels under line A (After centroid)
//				for (int l = 0; l < it1A.count; l++, ++it1A)
//				{
//					Profiles.push_back(buffer());
//					Profiles[ProfilesCount].startPoint = ep21;
//					Profiles[ProfilesCount].endPoint = ep22;
//					double valA = (double)linePixelsTempGraySrc3.at<uchar>(it1A.pos());
//					pixelsOnLineA.push_back(valA);
//					linePixel.push_back(valA);
//					//5555555555555555555555555555555555555555555555555555555555555555
//		/*			ofstream tempGraySrc3DataFile;
//					tempGraySrc3DataFile.open("linePixelsTempGraySrc3" + profileLinesCount2 + "A.csv");
//					tempGraySrc3DataFile << linePixelsTempGraySrc3 << "\n";
//					tempGraySrc3DataFile << "\n";
//					tempGraySrc3DataFile << "\n";
//					tempGraySrc3DataFile.close();*/
//					//55555555555555555555555555555555555555555555555555555555555555555555
//					////if (all_output){ cout << "Point(ep11.x, ep11.y), Point(ep12.x, ep12.y) = " << Point(ep21.x, ep21.y) << ", " << Point(ep22.x, ep22.y) << "\n";}
//					////if (all_output){ cout << "it1.pos() = " << Point(ep21.x, ep21.y) << ", " << it1A.pos() << "\n";}
//					////if (all_output){ cout << "(double)tempGraySrc3.at<uchar>(it1.pos()) = " << (double)linePixelsTempGraySrc3.at<uchar>(it1A.pos()) << "\n";}
//					Profiles[ProfilesCount].pixValues.push_back(valA);// (double)tempSrc3.at<uchar>(it1.pos());
//
//																	  //double val = (double)src_gray.at<uchar>(it.pos());
//																	  //buf[i] = val;
//
//					//std::string L = std::to_string(l);
//					//file.open("buf_" + format("(%d,%d)", Profiles[ProfilesCount].startPoint, Profiles[ProfilesCount].endPoint) + ".csv", ios::app);
//					//file.open("buf_" + L + ".csv", ios::app);
//					//file << Profiles[ProfilesCount].startPoint << "\n";
//					//file << Profiles[ProfilesCount].endPoint << "\n";
//					//file << Mat(Profiles[ProfilesCount].pixValues) << "\n";
//					//file.close();
//					ProfilesCount += 1;
//				}
//#pragma endregion
//
//				//4444444444444444444444444444444444444444444444444444444444444444444
//
//				if (pixelsOnLineB.empty())
//				{
//					////if (all_output){ cout << "pixelsOnLine is empty" << "\n";}
//				}
//				else
//				{
//					//ofstream PixelsOnLineBFile;
//					//PixelsOnLineBFile.open("PiixelsOnLineB" + profileLinesCount1 + ".csv");
//					//PixelsOnLineBFile << Mat(pixelsOnLineB) << "\n";;
//					//PixelsOnLineBFile << "\n";
//					//PixelsOnLineBFile << "\n";
//					//PixelsOnLineBFile.close();
//				}
//
//				if (pixelsOnLineA.empty())
//				{
//					//if (all_output){ cout << "pixelsOnLineA is empty" << "\n";}
//				}
//				else
//				{
//			/*		ofstream PixelsOnLineAFile;
//					PixelsOnLineAFile.open("PixelsOnLineA" + profileLinesCount2 + ".csv");
//					PixelsOnLineAFile << Mat(pixelsOnLineA) << "\n";;
//					PixelsOnLineAFile << "\n";
//					PixelsOnLineAFile << "\n";
//					PixelsOnLineAFile.close();*/
//				}
//				//4444444444444444444444444444444444444444444444444444444444444444444
//			}
//			if (all_output) { imshow("walk perpendicular in edge angle direction - component nr." + smi, tempSrc2); }
//#pragma endregion  
//
//#pragma region EPs
//			Point2f uuu, uuu2, uuu22, vvvv, ep1,ep2;
//			Point2f www1, www2;
//			uuu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
//			uuu2 = uuu;
//			rectSize_b = *max_element(lengths.begin(), lengths.end());
//			//double dd = 0.1*rectSize_b;
//			double normUUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
//			vvvv = Point2f(uuu.x / normUUU, uuu.y / normUUU);
//			//rotate and swap
//			Point2d vvvvv;
//			double tempXXX = vvvv.x;
//			vvvvv.x = -vvvv.y;
//			vvvvv.y = tempXXX;
//			for (size_t i = 0; i < 10; i++)
//			{
//				if (i == 0)
//				{
//					www1.x = maskCentroid.at(mi).x + vvvv.x*d;
//					www1.y = maskCentroid.at(mi).y + vvvv.y*d;
//					//ep1.x = www1.x-(box.size.width/2)*(vvvv.x*d);
//					//ep1.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
//					//ep2.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
//					//ep2.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
//
//					www2.x = maskCentroid.at(mi).x - vvvvv.x*d;
//					www2.y = maskCentroid.at(mi).y - vvvvv.y*d;
//					ep1.x = www2.x - (box.size.width / 2)*(vvvv.x);
//					ep1.y = www2.y - (box.size.width / 2)*(vvvv.y);
//					ep2.x = www2.x + (box.size.width / 2)*(vvvv.x);
//					ep2.y = www2.y + (box.size.width / 2)*(vvvv.y);
//				}
//				else
//				{
//					www1.x = uuu2.x + vvvv.x*d;
//					www1.y = uuu2.y + vvvv.y*d;
//					//ep1.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
//					//ep1.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
//					//ep2.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
//					//ep2.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
//					////if (all_output){ cout << "minAreaRect Angle= " << box.angle + 180 << "\n";}
//					www2.x = uuu22.x - vvvv.x*d;
//					www2.y = uuu22.y - vvvv.y*d;
//					ep1.x = www2.x - (box.size.width / 2)*(vvvv.x);
//					ep1.y = www2.y - (box.size.width / 2)*(vvvv.y);
//					ep2.x = www2.x + (box.size.width / 2)*(vvvv.x);
//					ep2.y = www2.y + (box.size.width / 2)*(vvvv.y);
//					////if (all_output){ cout << "ww2= " << ww2 << "\n";}
//				}
//				////if (all_output){ cout << "i - " << i << "1-Plot here= " << ww1 << ", " << ww2 << "\n";}
//				//circle(src, ww1, 1, Scalar(0, 255, 255), 1, 8, 0); //yellow
//				//circle(plotImage, ww2, 1, Scalar(255, 255, 0), 1, 8, 0); //turqoise
//				//circle(tempSrc2, www1, 1, Scalar(255, 255, 10), 1, 8, 0); //
//				circle(tempSrc2, www2, 1, Scalar(255, 255, 100), 1, 8, 0); //
//				circle(tempSrc2, ep1, 1, Scalar(255, 255, 200), 1, 8, 0);
//				circle(tempSrc2, ep2, 1, Scalar(255, 255, 10), 1, 8, 0);
//				uuu2 = www1;
//				uuu22 = www2;
//			}
//			//if (all_output){ imshow("Profile Lines", tempSrc2);
//#pragma endregion
//
//
//
//		//if (all_output){ imshow("Plot Image", plotImage);
//		//if (all_output){ imshow("Source - component nr." + smi, tempSrc2); 
//		//if (all_output){ imwrite("Source_component_nr_" + smi + ".bmp", tempSrc2);
//		//if (all_output){ imshow("Grayscale- component nr." + smi, tempGraySrc3);
//		//if (all_output){ imwrite("Grayscale- component nr." + smi + ".bmp", tempGraySrc3);
//		//} 
//		}
//	}
//#pragma endregion
//	if (all_output){ cout << "15 - RETRIEVING COMPONENTS (ONE-BY-ONE) FROM: 'maskImages'" << "\n";}
//
//	if (all_output){ cout << "16 - EACH COMPONENT STORED IN: 'tempComponent'" << "\n";}
//
//	if (all_output){ cout << "17 - CALCULATING EDGES/GRADIENT PER COMPONENT USING: Sobel" << "\n";}
//	if (all_output){ cout << "18 -  -- STORING CALCULATED EDGES RESULTS: 'grad_x' AND 'grad_y'" << "\n";}
//	if (all_output){ cout << "19 -  -- CALCULATING MAGNITUDE AND ANGLE MATRICES : 'Mag' AND 'Angle'" << "\n";}
//	if (all_output){ cout << "20 -  -- WRITING CALCULATED ANGLE RESULTS TO FILE: 'Angle_DataFile_i.csv'" << "\n";}
//	if (all_output){ cout << "21 -  -- CALCULATING CANNY EDGE: 'cannyEdge'" << "\n";}
//	if (all_output){ cout << "22 -  -- WRITING CALCULATED CANNY EDGE RESULTS TO FILE: 'cannyEdge_DataFile_I.csv'" << "\n";}
//	if (all_output){ cout << "23 -  -- -- FOR EACH CANNY EDGE PIXEL" << "\n";}
//	if (all_output){ cout << "24 -  -- -- FIND EACH NON-ZERO CANNY EDGE PIXEL" << "\n";}
//	if (all_output){ cout << "25 -  -- -- -- STORE EACH PIXEL IN NEW MATRIX: 'newAngle'" << "\n";}
//	if (all_output){ cout << "26 -  -- -- -- STORE BIN #, COORDINATES, ANGLE VALUE AND PIXEL VALUE IN: 'Container'" << "\n";}
//
//	if (all_output){ cout << "27 -  -- SORT 'Container' DATA AND WRITE RESULTS TO FILE: 'KeepingTrackOfContainers_DataFile_i.csv' AND 'ContainerFile_i.csv'" << "\n";}
//
//	if (all_output){ cout << "28 -  -- STORE SORTED DATA IN: 'maxCountContainer'" << "\n";}
//
//	if (all_output){ cout << "29 -  -- BIN'S WITH THEIR FREQUENCY ON UNIQUE ELEMENTS STORED IN: 'maxCountContainer_DataFile_i.csv'" << "\n";}
//
//	if (all_output){ cout << "30 -  -- BIN (BIN-VALUE) WITH HIGHEST FREQUENCY (WITH ITS CORRESPONDING ANGLE-VALUE AND BIN-SIZE) CALCULATED AND STORED IN: 'mes'" << "\n";}
//
//	if (all_output){ cout << "31 -  -- CALCULATING BOUNDING BOX" << "\n";}
//
//	if (all_output){ cout << "32 -  -- CALCULATING VECTOR IN DIRECTION OF PREVIOUSLY CALCULATED ANGLE (mes.angle): 'v'" << "\n";}
//	if (all_output){ cout << "33 -  -- -- CALCULATED VECTOR POINTS STORED IN: 'w1-one_side of cetroid point AND w2-otherside of cetroid point'" << "\n";}
//	if (all_output){ cout << "34 -  -- -- PLOTTING CALCULATING VECTOR ON: 'tempSrc2'" << "\n";}
//
//
//	if (all_output){ cout << "35 -  -- CALCULATING VECTOR IN DIRECTION PERPENDICULAR TO 'v': 'vv'" << "\n";}
//	if (all_output){ cout << "36 -  -- -- CALCULATED VECTOR POINTS STORED IN: 'ww1-one_side of cetroid point AND ww2-otherside of cetroid point'" << "\n";}
//	if (all_output){ cout << "37 -  -- -- PLOTTING CALCULATED VECTOR ON: 'tempSrc2'" << "\n";}
//
//	if (all_output){ cout << "38 -  -- CALCULATING VECTOR IN DIRECTION PERPENDICULAR TO 'v': 'vv'" << "\n";}
//	if (all_output){ cout << "39 -  -- -- CALCULATED END-POINTS STORED IN: Point on one side (x,y)-'ep11, ep12' AND Point on otherside (x,y)-'ep21, ep22'" << "\n";}
//	if (all_output){ cout << "40 -  -- -- DRAWING LINES BETWEEN POINTS (ep11,ep12) AND (ep21,ep22)" << "\n";}
//	if (all_output){ cout << "41 -  -- -- PLOTTING CALCULATING LINES ON: 'tempSrc2'" << "\n";}
//
//	if (all_output){ cout << "42 -  -- CALCULATING PIXEL PIXEL VALUES UNDER LINES" << "\n";}
//	if (all_output){ cout << "43 -  -- -- XXX" << "\n";}
//	if (all_output){ cout << "44 -  -- -- YYY" << "\n";}
//	if (all_output){ cout << "45 -  -- -- ZZZ" << "\n";}
//	//ComponentsLoop.close();
//	//ComponentAngle.close();
//	//if (all_output){ imshow("Plot Image", src);
//	//if (all_output){ imshow("Bounding Box", tempSrc1);
//	waitKey(0);
//	return 0;
//}