//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "iostream"
//#include "fstream"
//
//#include <opencv2/opencv.hpp>
//#include "opencv2/imgcodecs.hpp"
//#include <vector>
//#include <stdlib.h>
//#include <array>
//#include <math.h>
//
////#include <opencv2/legacy/compat.hpp>
//
//using std::vector;
//
//#define _USE_MATH_DEFINES
//using namespace cv;
//using namespace std;
//
//Mat src_gray;
//Mat dst, detected_edges, angleHist, origHist,grayHist, pixelsBin;
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
////---------------------------
//ofstream PixelsInBinFile;
//std::array<std::vector<int>, 2> vvv{ {} };
////string filename = "PixelsInBinFile";
////PixelsInBinFile.open("PixelsInBinFile.txt");
//int mouseX, mouseY;
//Point pt1, pt2;
//int clickCounter = 0;
//
////LineIterator it;
////LineIterator it2;
////vector<Vec3b> buf;
////-----------------------------
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
//int threshval = 60;
////---------------------------
//int CalcAngle(Mat src)
//{
//	ofstream myEdgeDetectorFile;
//	myEdgeDetectorFile.open("myEdgeDetectorFile.txt");
//
//	//![load]
//	//src = imread(argv[1], IMREAD_COLOR); // Load an image
//	//src = imread("contour_1.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
//															//IplImage* img = cvLoadImage("20161215 02.33_368L.jpg");
//	imshow("CalcAngle Contour Image", src);
//	ofstream CalcAngle_ContourMatrixFile;
//	CalcAngle_ContourMatrixFile.open("CalcAngle_ContourMatrixFile.csv");
//	CalcAngle_ContourMatrixFile << src << "\n";
//	CalcAngle_ContourMatrixFile.close();
//
//	if (src.empty())
//	{
//		//return -1;
//	}
//	//![load]
//
//	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
//	//![convert_to_gray]
//	//cvtColor(src, src_gray, COLOR_BGR2GRAY);
//	//![convert_to_gray]
//	src_gray = src;
//	myEdgeDetectorFile << "src_gray= " << src_gray << "\n";
//	myEdgeDetectorFile << "src_gray size= " << src_gray.size() << "\n";
//	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
//	//grayHist = getHistogram(src_gray,"GrayScaleImageHist");
//	imshow("CalcAngle_ContourGrayMatrixFile Image", src_gray);
//	ofstream CalcAngle_ContourGrayMatrixFile;
//	CalcAngle_ContourGrayMatrixFile.open("CalcAngle_ContourGrayMatrixFile.csv");
//	CalcAngle_ContourGrayMatrixFile << src << "\n";
//	CalcAngle_ContourGrayMatrixFile.close();
//	//set the callback function for any mouse event
//	//setMouseCallback("GrayScale Image", CallBackFunc, NULL);
//	//![reduce_noise]
//	//Reduce noise with a kernel 3x3
//	//blur(src_gray, detected_edges, Size(3, 3));
//	detected_edges = src_gray;
//	//![reduce_noise]
//	//cout << "src_gray = " << (double)src_gray.at<uchar>(2, 24) << "\n";
//	//imshow("Blurred Image", detected_edges);
//	//![canny]
//
//	//Canny detector
//	//Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
//	//![canny]
//	imshow("CalcAngle_ContourGrayCannyMatrixFile", detected_edges);
//	ofstream CalcAngle_ContourGrayCannyMatrixFile;
//	CalcAngle_ContourGrayCannyMatrixFile.open("CalcAngle_ContourGrayCannyMatrixFile.csv");
//	CalcAngle_ContourGrayCannyMatrixFile << detected_edges << "\n";
//	CalcAngle_ContourGrayCannyMatrixFile.close();
//	/// Gradient X
//	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//	Sobel(detected_edges, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//	ofstream GradXMatrixFile;
//	GradXMatrixFile.open("GradXMatrixFile.csv");
//	GradXMatrixFile << grad_x << "\n";
//	GradXMatrixFile.close();
//	/// Gradient Y
//	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//	Sobel(detected_edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//	ofstream GradYMatrixFile;
//	GradYMatrixFile.open("GradYMatrixFile.csv");
//	GradYMatrixFile << grad_y << "\n";
//	GradYMatrixFile.close();
//	//![sobel]
//	//convertScaleAbs(grad_x, abs_grad_x);
//	//convertScaleAbs(grad_y, abs_grad_y);
//	//cv::Mat angle(src.size(), CV_64F);
//
//	Mat Mag(src.size(), CV_32FC1);
//	Mat Angle(src.size(), CV_32FC1);
//
//	//cv::Mat angle(image.size(), CV_64F)
//	cout << "grad_x size= " << grad_x.size() << "\n";
//	cout << "grad_x depth= " << grad_y.depth() << "\n";
//	cout << "grad_y size= " << grad_y.size() << "\n";
//	cout << "grad_y depth= " << grad_y.depth() << "\n";
//	cout << "Angle size= " << Angle.size() << "\n";
//	cout << "Angle depth= " << Angle.depth() << "\n";
//	cout << "detected_edges size= " << detected_edges.size() << "\n";
//	cout << "detected_edges depth= " << detected_edges.depth() << "\n";
//	ofstream GradsAndAtansMatrixFile;
//			GradsAndAtansMatrixFile.open("GradsAndAtansMatrixFile.txt");
//	//for (size_t i = 0; i < detected_edges.rows; i++)
//	//{
//	////	//const float* xRow_i = grad_x.ptr<float>(i);
//	////	//const float* yRow_i = grad_y.ptr<float>(i);
//
//	//	for (size_t j = 0; j < detected_edges.cols; j++)
//	//	{	
//
//	//		GradsAndAtansMatrixFile << "grad_y.at<float>(" << i << "," << j << ") = " << grad_y.at<float>(i, j) << "\n";
//	//		GradsAndAtansMatrixFile << "grad_x.at<float>(" << i << "," << j << ") = "<< grad_x.at<float>(i, j) << "\n";
//	//		//Angle[i, j] = atan(yRow_i/ xRow_i);// (double)detected_edges.at<uchar>(i, j);
//	//		//if (grad_x.at<float>(i, j)!=0)
//	//		//{
//	//			Angle.at<float>(i, j) = (180/ 3.14159)*(atan((grad_y.at<float>(i, j))/(grad_x.at<float>(i, j))));
//	//		//}
//	//		
//	//	}
//
//	//}
//	//	GradsAndAtansMatrixFile.close();
//	//for each(i, j) such that contours[i, j] > 0
//	//{
//	//	angle[i, j] = atan2(dy[i, j], dx[i, j])
//	//}
//	//convertScaleAbs(grad_x, abs_grad_x);
//	//convertScaleAbs(grad_y, abs_grad_y);
//	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
//	//imshow("Sobel Demo - Simple Edge Detector", grad);
//	//Mat orientation;
//	////Mat orientation = Mat::zeros(abs_grad_x.rows, abs_grad_y.cols, CV_32F); //to store the gradients
//	////grad_x.convertTo(grad_x,CV_32F);
//	////grad_y.convertTo(grad_y, CV_32F);
//	//phase(grad_x, grad_y, orientation,true);
//	//cv::normalize(orientation, orientation, 0x00, 0xFF, cv::NORM_MINMAX, CV_8U);
//	//namedWindow("Orientation", CV_WINDOW_AUTOSIZE);
//	//imshow("Orientation", orientation);
//	////myEdgeDetectorFile << "grad_x= " << grad_x << "\n";
//	////myEdgeDetectorFile << "\n";
//	////myEdgeDetectorFile << "grad_y= " << grad_y << "\n";
//	////myEdgeDetectorFile << "\n";
//	//myEdgeDetectorFile << "orientation size= " << orientation.size() << "\n";
//	//myEdgeDetectorFile << "orientation= " << orientation << "\n";
//	////myEdgeDetectorFile << "\n";
//
//	double minM, maxM, minA, maxA;
//
//	//phase(grad_x, grad_y, Angle, true);
//	cartToPolar(grad_x, grad_y, Mag, Angle, true);
//
//	//angleHist = getHistogram(Angle, "AngleMatrixHist");
//	////cv::minMaxLoc(Mag, &minM, &maxM);
//	////cv::minMaxLoc(Angle, &minA, &maxA);
//	//myEdgeDetectorFile << "angleHist size(w,h)= " << angleHist.size().width << ", " << angleHist.size().height << "\n";
//	//myEdgeDetectorFile << "Original Image size= " << src.size() << "\n";
//	//myEdgeDetectorFile << "Angle size= " << Angle.size() << "\n";
//	//myEdgeDetectorFile << "\n";
//	//myEdgeDetectorFile << "angleHist= " << angleHist << "\n";
//
//	ofstream AngleMatrixFile;
//	AngleMatrixFile.open("Calculated AngleMatrixFile.csv");
//	AngleMatrixFile << Angle << "\n";
//	AngleMatrixFile.close();
//
//	ofstream GrayScaleMatrixFile;
//	GrayScaleMatrixFile.open("GrayScaleMatrixFile.csv");
//	GrayScaleMatrixFile << src_gray << "\n";
//	GrayScaleMatrixFile.close();
//
//	std::array<std::vector<int>, 72> vvv{ {} };
//	//cout << "gray scale image type= " << type2str(src_gray.type()) << "\n";
//	//for (size_t i = 0; i < Angle.rows; i++)
//	//{
//	//	//myHistogramFile << i << "\n";
//	//	const float* Row_i = Angle.ptr<float>(i);
//	//	//Row_i[j]
//	//	//const float* RowSG_i = src_gray.ptr<float>(i);
//	//	//float RowSG2_i = src_gray.at<float>(i);
//	//	//cout << "i" << i << "\n";
//
//	//	for (size_t j = 0; j < Angle.cols; j++)
//	//	{
//	//		/// Establish the number of bins
//	//		//int intensity = src_gray.at<int>(j,i);
//	//		int histSize = 256;
//	//		myEdgeDetectorFile << "Angle(i,j)= (" << i << ", " << j << ")= " << Row_i[j] << "\n";
//	//		//myEdgeDetectorFile << "(i,j)= (" << i << ", " << j << ")= " << Row_i[j] << "\n";
//	//		myEdgeDetectorFile << "Row_i[j]/binSize= (" << Row_i[j] << ", " << binSize << ")= " << int(Row_i[j] / binSize) << "\n";
//	//		//myEdgeDetectorFile << "src_gray[i=" << i << ",j=" <<  j << "]=" << RowSG_i[j] << "\n";
//	//		myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << (double)src_gray.at<uchar>(i, j) << "\n";
//
//	//		//vvv[i].at(j) = RowSG_i[j];
//	//		//vvv[int(Row_i[j] / binSize)].push_back(RowSG_i[j]);
//	//		vvv[int(Row_i[j] / binSize)].push_back((double)src_gray.at<uchar>(i, j));
//	//		//if (int(Row_i[j] / binSize)==53)
//	//		//{
//	//		//	myEdgeDetectorFile << "----------------------------------"<<"\n";
//	//		//	myEdgeDetectorFile << "Row_i[j]/binSize= (" << Row_i[j] << "/ " << binSize << ")= " << int(Row_i[j] / binSize) << "\n";
//	//		//	//myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << RowSG_i[j] << "\n";
//	//		//	myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << (double)src_gray.at<uchar>(i,j) << "\n";
//	//		//}
//	//		myEdgeDetectorFile << "\n";
//	//		myEdgeDetectorFile << "\n";
//	//		myEdgeDetectorFile << "\n";
//	//		//myEdgeDetectorFile << Row_i[j] << "\n";
//
//	//		//if (i==6 && j==274)
//	//		//{
//	//		//	cout << "src_gray[i=" << i <<",j=" << j <<"]="<<RowSG_i[j]<<"\n";
//	//		//}
//	//		//for (size_t k = 1; k < angleHist.rows; k++)
//	//		//{
//	//		//	binID = angleHist[k];
//	//		//	pixelsBin = PixelsInBin(src_gray, angleHist, i,j,k);
//	//		//}
//	//	}
//	//}
//
//	//ofstream ArrayVectorFile;
//	//ArrayVectorFile.open("ArrayVectorFile.txt");
//	//ArrayVectorFile << "vvv size= " << vvv.size() << "\n";
//	//int mySum, myMean, mySD;
//	//for (int i = 0; i<vvv.size(); i++)
//	//{
//	//	mySum = 0;
//	//	//std::ArrayVectorFile << "v.size()= " << vvv.size() << "\n";
//	//	ArrayVectorFile << "vvv[" << i << "].size()" << vvv[i].size() << "\n";
//	//	for (int j = 0; j<vvv[i].size(); j++)
//	//	{
//	//		//ArrayVectorFile << "vvv size= " << vvv.size() << "\n";
//	//		//ArrayVectorFile << "vvv[" << i<< "].size()" << vvv[i].size() << "\n";
//	//		//if (!vvv[i].empty())
//	//		//{
//	//		ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
//	//		mySum += vvv[i].at(j);
//	//		//myMean = mean(vvv[i],NULL);
//	//		//}
//	//		//if (i==53)
//	//		//{
//	//		//	ArrayVectorFile << "\n";
//	//		//	ArrayVectorFile << "\n";
//	//		//	ArrayVectorFile << "\n";
//	//		//	ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
//	//		//}
//	//		//std::cout << "2- i= "<<i<<"\n";
//	//	}
//	//	ArrayVectorFile << "sum= " << mySum << "\n";
//	//	//cout << "i= " << i << ", j= " << j << ", value= " << v[i].at(j)<< "\n";
//	//	//std::cout << "3- i= "<<i<<"\n";
//	//}
//	//ArrayVectorFile.close();
//	//myEdgeDetectorFile << "vvv= " << vvv << "\n";
//	myEdgeDetectorFile.close();
//
//	waitKey(0);
//	return 0;
//}
//
//void WriteToFile(String filename, Mat image)
//{
//	ofstream outputFile;
//	outputFile.open(filename+".csv");
//
//	outputFile << image << "\n";
//	// Iterate through pixels.
//	//for (int r = 0; r < image.rows; r++)
//	//{
//	//	for (int c = 0; c < image.cols; c++)
//	//	{
//	//		int pixel = image.at<uchar>(r, c);
//
//	//		outputFile << pixel;
//	//	}
//	//	outputFile << endl;
//	//}
//	outputFile.close();
//}
//int main()
//{		
//	Mat image;
//	Mat gray;				
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	//namedWindow("Display window", CV_WINDOW_AUTOSIZE);
//	ofstream ContourMatrixFile;
//	for (int j = 1; j < 2/*773*/; j++)
//	{
//
//		std::string s = std::to_string(j);
//		image = imread("mask_i_" + s + ".png",1);
//		cout << "mask image type= " << image.type() << "\n";
//		cout << "mask image depth= " << image.depth() << "\n";
//		cout << "mask image channels= " << image.channels() << "\n";
//		//WriteToFile("findContour_Mask_i_MatrixFile",image);
//		ofstream outputFile;
//		outputFile.open("findContour_Mask_i_MatrixFile.csv");
//		outputFile << image << "\n";
//		outputFile.close();
//		//image = imread("connectedComponentImage01.jpg", 1);
//		//---------------------------------------
//		//is this gray or binary?
//		//cast to binary since original connectCJPG suppose to be binary
//
//		//use neighbourhopod of original image to calculate angle from binary
//		//walk through binary get equivalent neihbourhood form orignal and claculate Angle
//
//		//read image, filter (denoise), connectedComponent (binary), contouring (single pixel think contour = edge), walk along edge get neigh from orihinal and calculate angle, mag of edge point
//
//		imshow("mask_i Original Image", image);
//		//cout << "image file name= " << "mask_i_" + s + ".jpg" << "\n";
//		gray = image;
//		//cvtColor(image, gray, CV_BGR2GRAY);
//		//Canny(gray, gray, 100, 200, 3);
//		WriteToFile("findContours_GrayImage",gray);
//
//		/// Find contours   
//		//cv::Mat binaryImg = threshval < 128 ? (gray < threshval) : (gray > threshval);
//		//cv::Mat binaryImg(gray.size(), gray.type());
//		//cv::Mat binaryImg(image.size(), image.type());
//		//cv::threshold(gray, binaryImg, 100, 255, cv::THRESH_BINARY);
//		cv::Mat binaryImg = image;
//		imshow("Binary Image", binaryImg);
//		WriteToFile("findContours_BinaryMaskMatrixFile",binaryImg);
//		//ofstream findContours_BinaryMaskMatrixFile;
//		//findContours_BinaryMaskMatrixFile.open("findContours_BinaryMaskMatrixFile.csv");
//		//findContours_BinaryMaskMatrixFile << binaryImg << "\n";
//		//findContours_BinaryMaskMatrixFile.close();
//		
//		RNG rng(12345);
//		findContours(binaryImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
//		/// Draw contours
//		Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
//		for (int i = 0; i < contours.size(); i++)
//		{
//			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//			drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
//		}
//
//		imwrite("contour_" + s + ".jpg", drawing);
//		imshow("mask_i Contour Image", drawing);
//
//		if (j==1)
//		{
//		ofstream ContourMatrixFile;	
//		ContourMatrixFile.open("ContourMatrixFile_" + s + ".csv");
//		ContourMatrixFile << drawing << "\n";
//		ContourMatrixFile.close();
//
//		int temp = CalcAngle(drawing);
//		}
//
//
//		//cout << "Processing: " << "ContourMatrixFile_" + s << "\n";
//
//		//imshow("Result window", drawing);
//		//imshow("Display window", image);
//	}
//
//		
//
//	waitKey(0);
//	return 0;
//}
//
//
//
//
//
//
//
////#include "opencv2/core/core.hpp"
////#include "opencv2/highgui/highgui.hpp""
////#include "opencv2/imgproc/imgproc.hpp"
////#include "iostream"
////using namespace cv;
////using namespace std;
////int main()
////{
////	Mat src;
////	src = imread("mask_i.jpg", CV_LOAD_IMAGE_COLOR);
////	Mat gray;
////	cvtColor(src, gray, CV_BGR2GRAY);
////	threshold(gray, gray, 200, 255, THRESH_BINARY_INV); //Threshold the gray
////	imshow("gray", gray); int largest_area = 0;
////	int largest_contour_index = 0;
////	Rect bounding_rect;
////	vector<vector<Point>> contours; // Vector for storing contour
////	vector<Vec4i> hierarchy;
////	findContours(gray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
////	// iterate through each contour.
////	for (int i = 0; i< contours.size(); i++)
////	{
////		//  Find the area of contour
////		double a = contourArea(contours[i], false);
////		std::string s = std::to_string(i);
////		string name = "contour-" + s;
////		cout << "name= " << name << "\n";
////		int myR = 255 - i;
////		int myG = i + i;
////		int myB = 10 * i;
////		Scalar color(myB, myG, myR);  // color of the contour in the
////		drawContours(src, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy);
////		rectangle(src, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
////		//namedWindow(name, CV_WINDOW_AUTOSIZE);
////		imshow(name, src);
////		//imwrite(name + ".jpg", src);
////		if (a>largest_area) {
////			largest_area = a; cout << i << " area  " << a << endl;
////			// Store the index of largest contour
////			largest_contour_index = i;
////			// Find the bounding rectangle for biggest contour
////			bounding_rect = boundingRect(contours[i]);
////
////		}
////	}
////	//Scalar color(255, 255, 255);  // color of the contour in the
////	//							  //Draw the contour and rectangle
////	//drawContours(src, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy);
////	//rectangle(src, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
////	//namedWindow("Display window", CV_WINDOW_AUTOSIZE);
////	//imshow("Display window", src);
////	waitKey(0);
////	return 0;
////}