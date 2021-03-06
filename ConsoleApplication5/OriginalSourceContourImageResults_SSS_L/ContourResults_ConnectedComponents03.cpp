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
//#include <opencv2/legacy/compat.hpp>
using std::vector;

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
Mat detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;
//int edgeThresh = 1;
//int lowThreshold;
//int const max_lowThreshold = 100;
//int ratio = 3;
//int kernel_size = 3;
//const char* window_name = "Edge Map";
int ddepth = CV_32FC1;// CV_16S;
int scale = 1;
int delta = 0;

int CalcAngle(Mat orig, Mat src, int componentCount)
{
	std::string count = std::to_string(componentCount);

	//imshow("Original Image", orig);
	ofstream myEdgeDetectorFile;
	myEdgeDetectorFile.open("myEdgeDetectorFile.txt");

	//![load]
	//src = imread(argv[1], IMREAD_COLOR); // Load an image
	//src = imread("contour_1.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	//IplImage* img = cvLoadImage("20161215 02.33_368L.jpg");
	//imshow("CalcAngleSourceImage_"+ componentCount, src);
	ofstream CalcAngle_ContourMatrixFile;
	//CalcAngle_ContourMatrixFile.open("CalcAngle_"+ count +"_SourceCannyMatrixFile.csv");
	CalcAngle_ContourMatrixFile.open("CalcAngle_" + count + "_SourceMatrixFile.csv");
	CalcAngle_ContourMatrixFile << src << "\n";
	CalcAngle_ContourMatrixFile.close();

	if (src.empty())
	{
		//return -1;
	}
	//![load]

	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
	//![convert_to_gray]
	//cvtColor(src, src_gray, COLOR_BGR2GRAY);
	//![convert_to_gray]
	angle_src_gray = src;
	myEdgeDetectorFile << "CalcAngle_src_gray_"+ count +"= " << angle_src_gray << "\n";
	myEdgeDetectorFile << "CalcAngle_src_gray_"+ count+" size= " << angle_src_gray.size() << "\n";
	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
	//grayHist = getHistogram(src_gray,"GrayScaleImageHist");
	//imshow("CalcAngleGrayImage_"+ count, angle_src_gray);
	ofstream CalcAngle_ContourGrayMatrixFile;
	//CalcAngle_ContourGrayMatrixFile.open("CalcAngle_"+ count+"_ContourGrayMatrixFile.csv");
	CalcAngle_ContourGrayMatrixFile.open("CalcAngle_" + count + "_GrayMatrixFile.csv");
	CalcAngle_ContourGrayMatrixFile << angle_src_gray << "\n";
	CalcAngle_ContourGrayMatrixFile.close();
	//set the callback function for any mouse event
	//setMouseCallback("GrayScale Image", CallBackFunc, NULL);
	//![reduce_noise]
	//Reduce noise with a kernel 3x3
	//blur(angle_src_gray, detected_edges, Size(3, 3));
	detected_edges = angle_src_gray;
	//![reduce_noise]
	//cout << "src_gray = " << (double)src_gray.at<uchar>(2, 24) << "\n";
	//imshow("Blurred Image", detected_edges);
	//![canny]

	//Canny detector
	//Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	//![canny]
	//imshow("CalcAngle_"+count+"_DetectedEdgesGrayCannyMatrixFile", detected_edges);
	//ofstream CalcAngle_ContourGrayCannyMatrixFile;
	//CalcAngle_ContourGrayCannyMatrixFile.open("CalcAngle_"+ count+"_DetectedEdgesGrayCannyMatrixFile.csv");
	//CalcAngle_ContourGrayCannyMatrixFile << detected_edges << "\n";
	//CalcAngle_ContourGrayCannyMatrixFile.close();	
	
	Mat orig_gray;
	cvtColor(orig, orig_gray, CV_BGR2GRAY);

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(detected_edges, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	ofstream GradXMatrixFile;
	GradXMatrixFile.open("GradXMatrixFile_"+ count+".csv");
	GradXMatrixFile << grad_x << "\n";
	GradXMatrixFile.close();
	convertScaleAbs(grad_x, abs_grad_x);


	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(detected_edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	ofstream GradYMatrixFile;
	GradYMatrixFile.open("GradYMatrixFile_"+ count+".csv");
	GradYMatrixFile << grad_y << "\n";
	GradYMatrixFile.close();
	convertScaleAbs(grad_y, abs_grad_y);
	//![sobel]
	//convertScaleAbs(grad_x, abs_grad_x);
	//convertScaleAbs(grad_y, abs_grad_y);
	//cv::Mat angle(src.size(), CV_64F);

	Mat Mag(src.size(), CV_32FC1);
	Mat Angle(src.size(), CV_32FC1);

	//cv::Mat angle(image.size(), CV_64F)
	cout << "grad_x_"+ count+ "size= " << grad_x.size() << "\n";
	cout << "grad_x_"+ count+" depth= " << grad_y.depth() << "\n";
	cout << "grad_y_"+ count+" size= " << grad_y.size() << "\n";
	cout << "grad_y_"+ count+" depth= " << grad_y.depth() << "\n";
	cout << "Angle_"+ count+" size= " << Angle.size() << "\n";
	cout << "Angle_"+ count+" depth= " << Angle.depth() << "\n";
	cout << "detected_edges_"+ count+" size= " << detected_edges.size() << "\n";
	cout << "detected_edges_"+ count+" depth= " << detected_edges.depth() << "\n";
	ofstream GradsAndAtansMatrixFile;
	GradsAndAtansMatrixFile.open("GradsAndAtans+"+ count+"_MatrixFile.txt");
	//for (size_t i = 0; i < detected_edges.rows; i++)
	//{
	////	//const float* xRow_i = grad_x.ptr<float>(i);
	////	//const float* yRow_i = grad_y.ptr<float>(i);

	//	for (size_t j = 0; j < detected_edges.cols; j++)
	//	{	

	//		GradsAndAtansMatrixFile << "grad_y.at<float>(" << i << "," << j << ") = " << grad_y.at<float>(i, j) << "\n";
	//		GradsAndAtansMatrixFile << "grad_x.at<float>(" << i << "," << j << ") = "<< grad_x.at<float>(i, j) << "\n";
	//		//Angle[i, j] = atan(yRow_i/ xRow_i);// (double)detected_edges.at<uchar>(i, j);
	//		//if (grad_x.at<float>(i, j)!=0)
	//		//{
	//			Angle.at<float>(i, j) = (180/ 3.14159)*(atan((grad_y.at<float>(i, j))/(grad_x.at<float>(i, j))));
	//		//}
	//		
	//	}

	//}
	//	GradsAndAtansMatrixFile.close();
	//for each(i, j) such that contours[i, j] > 0
	//{
	//	angle[i, j] = atan2(dy[i, j], dx[i, j])
	//}
	//convertScaleAbs(grad_x, abs_grad_x);
	//convertScaleAbs(grad_y, abs_grad_y);
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	//imshow("Sobel Demo - Simple Edge Detector", grad);
	//Mat orientation;
	////Mat orientation = Mat::zeros(abs_grad_x.rows, abs_grad_y.cols, CV_32F); //to store the gradients
	////grad_x.convertTo(grad_x,CV_32F);
	////grad_y.convertTo(grad_y, CV_32F);
	//phase(grad_x, grad_y, orientation,true);
	//cv::normalize(orientation, orientation, 0x00, 0xFF, cv::NORM_MINMAX, CV_8U);
	//namedWindow("Orientation", CV_WINDOW_AUTOSIZE);
	//imshow("Orientation", orientation);
	////myEdgeDetectorFile << "grad_x= " << grad_x << "\n";
	////myEdgeDetectorFile << "\n";
	////myEdgeDetectorFile << "grad_y= " << grad_y << "\n";
	////myEdgeDetectorFile << "\n";
	//myEdgeDetectorFile << "orientation size= " << orientation.size() << "\n";
	//myEdgeDetectorFile << "orientation= " << orientation << "\n";
	////myEdgeDetectorFile << "\n";

	double minM, maxM, minA, maxA;

	//phase(grad_x, grad_y, Angle, true);
	cartToPolar(grad_x, grad_y, Mag, Angle, true);

	//angleHist = getHistogram(Angle, "AngleMatrixHist");
	////cv::minMaxLoc(Mag, &minM, &maxM);
	////cv::minMaxLoc(Angle, &minA, &maxA);
	//myEdgeDetectorFile << "angleHist size(w,h)= " << angleHist.size().width << ", " << angleHist.size().height << "\n";
	//myEdgeDetectorFile << "Original Image size= " << src.size() << "\n";
	//myEdgeDetectorFile << "Angle size= " << Angle.size() << "\n";
	//myEdgeDetectorFile << "\n";
	//myEdgeDetectorFile << "angleHist= " << angleHist << "\n";

	ofstream AngleMatrixFile;
	AngleMatrixFile.open("Calculated Angle_"+ count+"_MatrixFile.csv");
	AngleMatrixFile << Angle << "\n";
	AngleMatrixFile.close();

	ofstream GrayScaleMatrixFile;
	GrayScaleMatrixFile.open("GrayScale_"+ count+"_MatrixFile.csv");
	GrayScaleMatrixFile << angle_src_gray << "\n";
	GrayScaleMatrixFile.close();

	std::array<std::vector<int>, 72> vvv{ {} };
	//cout << "gray scale image type= " << type2str(src_gray.type()) << "\n";
	//for (size_t i = 0; i < Angle.rows; i++)
	//{
	//	//myHistogramFile << i << "\n";
	//	const float* Row_i = Angle.ptr<float>(i);
	//	//Row_i[j]
	//	//const float* RowSG_i = src_gray.ptr<float>(i);
	//	//float RowSG2_i = src_gray.at<float>(i);
	//	//cout << "i" << i << "\n";

	//	for (size_t j = 0; j < Angle.cols; j++)
	//	{
	//		/// Establish the number of bins
	//		//int intensity = src_gray.at<int>(j,i);
	//		int histSize = 256;
	//		myEdgeDetectorFile << "Angle(i,j)= (" << i << ", " << j << ")= " << Row_i[j] << "\n";
	//		//myEdgeDetectorFile << "(i,j)= (" << i << ", " << j << ")= " << Row_i[j] << "\n";
	//		myEdgeDetectorFile << "Row_i[j]/binSize= (" << Row_i[j] << ", " << binSize << ")= " << int(Row_i[j] / binSize) << "\n";
	//		//myEdgeDetectorFile << "src_gray[i=" << i << ",j=" <<  j << "]=" << RowSG_i[j] << "\n";
	//		myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << (double)src_gray.at<uchar>(i, j) << "\n";

	//		//vvv[i].at(j) = RowSG_i[j];
	//		//vvv[int(Row_i[j] / binSize)].push_back(RowSG_i[j]);
	//		vvv[int(Row_i[j] / binSize)].push_back((double)src_gray.at<uchar>(i, j));
	//		//if (int(Row_i[j] / binSize)==53)
	//		//{
	//		//	myEdgeDetectorFile << "----------------------------------"<<"\n";
	//		//	myEdgeDetectorFile << "Row_i[j]/binSize= (" << Row_i[j] << "/ " << binSize << ")= " << int(Row_i[j] / binSize) << "\n";
	//		//	//myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << RowSG_i[j] << "\n";
	//		//	myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << (double)src_gray.at<uchar>(i,j) << "\n";
	//		//}
	//		myEdgeDetectorFile << "\n";
	//		myEdgeDetectorFile << "\n";
	//		myEdgeDetectorFile << "\n";
	//		//myEdgeDetectorFile << Row_i[j] << "\n";

	//		//if (i==6 && j==274)
	//		//{
	//		//	cout << "src_gray[i=" << i <<",j=" << j <<"]="<<RowSG_i[j]<<"\n";
	//		//}
	//		//for (size_t k = 1; k < angleHist.rows; k++)
	//		//{
	//		//	binID = angleHist[k];
	//		//	pixelsBin = PixelsInBin(src_gray, angleHist, i,j,k);
	//		//}
	//	}
	//}

	//ofstream ArrayVectorFile;
	//ArrayVectorFile.open("ArrayVectorFile.txt");
	//ArrayVectorFile << "vvv size= " << vvv.size() << "\n";
	//int mySum, myMean, mySD;
	//for (int i = 0; i<vvv.size(); i++)
	//{
	//	mySum = 0;
	//	//std::ArrayVectorFile << "v.size()= " << vvv.size() << "\n";
	//	ArrayVectorFile << "vvv[" << i << "].size()" << vvv[i].size() << "\n";
	//	for (int j = 0; j<vvv[i].size(); j++)
	//	{
	//		//ArrayVectorFile << "vvv size= " << vvv.size() << "\n";
	//		//ArrayVectorFile << "vvv[" << i<< "].size()" << vvv[i].size() << "\n";
	//		//if (!vvv[i].empty())
	//		//{
	//		ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
	//		mySum += vvv[i].at(j);
	//		//myMean = mean(vvv[i],NULL);
	//		//}
	//		//if (i==53)
	//		//{
	//		//	ArrayVectorFile << "\n";
	//		//	ArrayVectorFile << "\n";
	//		//	ArrayVectorFile << "\n";
	//		//	ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
	//		//}
	//		//std::cout << "2- i= "<<i<<"\n";
	//	}
	//	ArrayVectorFile << "sum= " << mySum << "\n";
	//	//cout << "i= " << i << ", j= " << j << ", value= " << v[i].at(j)<< "\n";
	//	//std::cout << "3- i= "<<i<<"\n";
	//}
	//ArrayVectorFile.close();
	//myEdgeDetectorFile << "vvv= " << vvv << "\n";
	myEdgeDetectorFile.close();

	//waitKey(0);
	return 0;
}


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
	srcImg = cv::imread("20161215 02.33_368L.jpg");
	if (srcImg.empty()) {
		return -1;
	}

	string filter;		
	Mat fltrImg;
	cv::Mat fltrGrayImg;
	//for (int i = 0; i < 2; i++)
	//{
	//	myConnectedComponents03file << "i= " << i << std::endl;
	//	if (i==0)
	//	{
			fltrImg = srcImg;
			filter = "Original";

	//	}
	//	else if (i==1)
	//	{
			////Apply median filter
			//blur(srcImg, fltrImg, Size(5, 5), Point(-1, -1));
			//filter = "median filter result";
	//	}
	//	else if (i==2)
	//	{
	//		//Apply bilateral filter
	//		bilateralFilter(srcImg, fltrImg, 15, 80, 80);
	//		filter = "bilateralFilter result";
	//	}
	//	else
	//	{
	//		//fltrImg = srcImg;
	//		//filter = "Original";
	//	}
		//fltrImg = srcImg;
		//filter = "src image";
		imshow(filter, fltrImg);
		imwrite(filter+".bmp", fltrImg);
		cv::cvtColor(fltrImg, fltrGrayImg, cv::COLOR_BGR2GRAY);
		//fltrGrayImg = srcImg;
		imshow("gray-scale image", fltrGrayImg);
		//Mat FltrBinaryImg;
		//cv::threshold(fltrGrayImg, FltrBinaryImg, 100, 255, cv::THRESH_BINARY_INV);
		cv::Mat FltrBinaryImg = threshval < 128 ? (fltrGrayImg < threshval) : (fltrGrayImg > threshval);
		imshow("binary image", FltrBinaryImg);
		cv::Mat FltrLabelImage;
		cv::Mat FltrStats, FltrCentroids;
		int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
		std::string nFltrLabelsString = std::to_string(nFltrLabels);
		//normalize(nFltrLabels, FltrLabelImage, 0, 255, NORM_MINMAX, CV_8U);
		cv::Mat FltrLabelImage2;
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
		for (int FltrLabel = 1; FltrLabel < 5/*nFltrLabels*/; ++FltrLabel) {
			FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
			//myConnectedComponents03file << "Component " << FltrLabel << std::endl;
			//myConnectedComponents03file << "CC_STAT_LEFT   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
			//myConnectedComponents03file << "CC_STAT_TOP    = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
			//myConnectedComponents03file << "CC_STAT_WIDTH  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
			//myConnectedComponents03file << "CC_STAT_HEIGHT = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
			//myConnectedComponents03file << "CC_STAT_AREA   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
			//myConnectedComponents03file << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;

			// Get the mask for the i-th contour
			Mat mask_i = FltrLabelImage == FltrLabel;
			string name = "mask_i_";
			std::string s = std::to_string(FltrLabel);
			//strcat("mask_i_", std::to_string(i));
			//if (FltrLabel==1)
			//{
			//imshow("mask_i_" + s, mask_i);
			imwrite("mask_i_" + s + ".bmp", mask_i);
			ofstream Mask_MatrixFile;
			Mask_MatrixFile.open("Mask_" + s + "_MatrixFile.csv");
			Mask_MatrixFile << mask_i << "\n";
			Mask_MatrixFile.close();
			//}

			//=======================================================================
			 ///apply your filter
			Canny(mask_i, mask_i, 100, 200);
			//if (FltrLabel == 1)
			//{
				//imshow("Canny mask_i_" + s, mask_i);			
				
			imwrite("Cannymask_i_" + s + ".bmp", mask_i);
			ofstream CannyMask_MatrixFile;
			CannyMask_MatrixFile.open("CannyMask_" + s + "_MatrixFile.csv");
			//CannyMask_MatrixFile.open("Mask_" + s + "_MatrixFile.csv");
			CannyMask_MatrixFile << mask_i << "\n";
			CannyMask_MatrixFile.close();
			//}

			/////walk the edge
			//ofstream WalkCannyMask_MatrixFile;
			//WalkCannyMask_MatrixFile.open("WalkCannyMask_" + s + "_MatrixFile.txt");
			////WalkCannyMask_MatrixFile << mask_i << "\n";
			//for (int r = 0; r < mask_i.rows; ++r) {
			//	for (int c = 0; c < mask_i.cols; ++c) {
			//		double value = (double)mask_i.at<uchar>(r, c);
			//		if (value==255)
			//		{
			//			WalkCannyMask_MatrixFile << "(r=" << r << ", c=" << c << ")= " << (double)mask_i.at<uchar>(r, c)<<" HERE" << "\n";
			//			WalkCannyMask_MatrixFile << "(r=" << r << ", c=" << c << ")= " << (double)srcImg.at<uchar>(r, c) << "\n";
			//			
			//			WalkCannyMask_MatrixFile << "(r=" << r << ", c=" << c << ")= " << (double)fltrGrayImg.at<uchar>(r, c) << "\n";
			//		}
			//		/*WalkCannyMask_MatrixFile << "(r="<<r<<", c="<<c<<")= "<<(double)mask_i.at<uchar>(r,c) << "\n";*/

			//	}
			//}
			//WalkCannyMask_MatrixFile.close();

			/*int temp = CalcAngle(srcImg, mask_i, FltrLabel);*/

			findContours(mask_i.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			//
			////cout << "hierarchy size= " << hierarchy.size() << "\n";

			/// Draw contours
			RNG rng(12345);
			Mat drawing = Mat::zeros(fltrGrayImg.size(), CV_8UC3);

			///Find the rotated rectangles and ellipses for each contour
			//vector<RotatedRect> minRect(contours.size());
			//vector<RotatedRect> minEllipse(contours.size());
			//ofstream ContourMinEllipseMatrixFile;
			//ContourMinEllipseMatrixFile.open("Contour_" + s + "MinEllipseMatrixFile.csv");
			//cout << "contours size= " << contours.size() << "\n";
			for (int i = 0; i < contours.size(); i++)
			{
				//random colour for contour
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				// contour
				drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());

				int temp = CalcAngle(srcImg, drawing, FltrLabel);
				//if (FltrLabel == 1)
				//{
					//imshow("mask_i Contour_" + s + " Image", drawing);
					imwrite("mask_i Contour_" + s + " Image" + ".bmp", drawing);
					ofstream mask_iContourMatrixFile;
					mask_iContourMatrixFile.open("mask_i Contour_" + s + " Image"+".csv");
					mask_iContourMatrixFile << drawing << "\n";
					mask_iContourMatrixFile.close();
				//}
				//imshow("mask_i Contour_" + s + " Image", drawing);
				//Mat test = Mat(contours[i]);
				//minEllipse[i] = minAreaRect(test);

				//ContourMinEllipseMatrixFile << minEllipse[i].angle << "\n";
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
			//ContourMinEllipseMatrixFile.close();
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

			//imwrite("contour_" + s + ".bmp", drawing);

			//(x, y), (MA, ma), angle = cv2.fitEllipse(cnt);

			//if (FltrLabel==1)
			//{
			//	imshow("mask_i Contour_"+s+" Image", drawing);
			//	ofstream ContourMatrixFile;
			//	ContourMatrixFile.open("ContourMatrixFile_" + s + ".csv");
			//	ContourMatrixFile << drawing << "\n";
			//	ContourMatrixFile.close();
			//	//cout << "hierarchy size= " << hierarchy.size() << "\n";
			//	//ofstream ContourHierarchyMatrixFile;
			//	//ContourHierarchyMatrixFile.open("Contour_"+s+"HierarchyMatrixFile.txt");
			//	//ContourHierarchyMatrixFile << "[Next, Previous, First_Child, Parent]" << endl;
			//	//for (int k = 0; k<hierarchy.size(); k++)
			//	//{
			//	//	ContourHierarchyMatrixFile << hierarchy[k] << endl;

			//	//}
			//	//ContourHierarchyMatrixFile.close();
			//}



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
		imshow(nFltrLabelsString+"-Connected Components", FltrDst);
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