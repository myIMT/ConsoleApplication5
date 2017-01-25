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
//
//using std::vector;
//
//using namespace cv;
//using namespace std;
//
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
//Mat detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;
////int edgeThresh = 1;
////int lowThreshold;
////int const max_lowThreshold = 100;
////int ratio = 3;
////int kernel_size = 3;
////const char* window_name = "Edge Map";
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
//int channels[1];
//int binID;
////----------------------------------------------------
//
//string type2str(int type) {
//	string r;
//
//	uchar depth = type & CV_MAT_DEPTH_MASK;
//	uchar chans = 1 + (type >> CV_CN_SHIFT);
//
//	switch (depth) {
//	case CV_8U:  r = "8U"; break;
//	case CV_8S:  r = "8S"; break;
//	case CV_16U: r = "16U"; break;
//	case CV_16S: r = "16S"; break;
//	case CV_32S: r = "32S"; break;
//	case CV_32F: r = "32F"; break;
//	case CV_64F: r = "64F"; break;
//	default:     r = "User"; break;
//	}
//
//	r += "C";
//	r += (chans + '0');
//
//	return r;
//}
//
/////Histogram Analysis
//cv::MatND getHistogram(const cv::Mat &data, string file) {
//	cv::MatND hist;
//
//	ofstream myAngleHistFile;
//	myAngleHistFile.open(file + ".txt");
//
//	// Compute histogram
//	//calcHist(&data, 1, 0, Mat(), angle_hist, 1, &histSize, &histRange, uniform, myAccumulate);
//	calcHist(&data,
//		1, // histogram from 1 image only
//		channels, // the channel used
//		cv::Mat(), // no mask is used
//		hist, // the resulting histogram
//		1, // it is a 1D histogram
//		&histSize, // number of bins
//		&histRange, // pixel value range
//		uniform,
//		myAccumulate
//	);
//
//
//	// Plot the histogram
//	int hist_w = 512; int hist_h = 400;   myAngleHistFile << "hist_w= " << hist_w << "\n";    myAngleHistFile << "hist_h= " << hist_h << "\n";
//	int bin_w = cvRound((double)hist_w / histSize);   myAngleHistFile << "bin_w= " << bin_w << "\n";
//
//	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));   myAngleHistFile << "Black Image of size(h,w)= (" << hist_h << "," << hist_w << ")" << "\n";
//	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//
//	myAngleHistFile << "AngleLimit= " << AngleLimit << "\n";
//	myAngleHistFile << "binSize= " << binSize << "\n";
//	myAngleHistFile << "(number of bins) histSize= " << histSize << "\n";
//	myAngleHistFile << "hist size= " << hist.size() << "\n";
//
//	myAngleHistFile << "\n";    myAngleHistFile << "\n";
//
//	for (int i = 1; i < histSize; i++)
//	{
//		myAngleHistFile << "histSize(" << i - 1 << ")= " << hist.at<float>(i - 1) << "   ------   ";
//		myAngleHistFile << "Line(" << i - 1 << ")= (" << bin_w*(i - 1) << "," << hist_h - cvRound(hist.at<float>(i - 1)) << ") TO (" << bin_w*(i) << "," << hist_h - cvRound(hist.at<float>(i)) << ")" << "\n";
//
//
//
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
//			Scalar(255, 0, 0), 2, 8, 0);
//	}
//
//	myAngleHistFile << "\n";    myAngleHistFile << "\n";
//	myAngleHistFile << "hist= " << hist << "\n";
//	myAngleHistFile << "\n";    myAngleHistFile << "\n";
//	myAngleHistFile << "histImage= " << histImage << "\n";
//	namedWindow(file, 1);    imshow(file, histImage);
//	myAngleHistFile.close();
//	//waitKey(0);
//	return hist;
//}
//
//Mat PlotHistogram(Mat adata, int nimages, int channel, int dims, int binSize, int rangeMin, int rangeMax)
//{
//	/// Establish the number of bins
//	int ahistSize = rangeMax / binSize;
//
//	/// Set the range
//	float range[] = { rangeMin, rangeMax };
//	const float* ahistRange = { range };
//	int mychannels[] = { channel };
//	///We want our bins to have the same size (uniform) and to clear the histograms in the beginning, so:
//	bool auniform = true; bool aaccumulate = false;
//
//	Mat ahist;
//
//	/// Compute the histograms:
//	//calcHist(&adata, 1, channels, Mat(), ahist, 1, &ahistSize, &ahistRange, auniform, aaccumulate);
//	calcHist(&adata,
//		1, // histogram from 1 image only
//		channels, // the channel used
//		cv::Mat(), // no mask is used
//		ahist, // the resulting histogram
//		1, // it is a 1D histogram
//		&histSize, // number of bins
//		&histRange, // pixel value range
//		uniform,
//		myAccumulate
//	);
//	// Draw the histograms for B, G and R
//	int hist_w = 512; int hist_h = 400;
//	int bin_w = cvRound((double)hist_w / histSize);
//
//	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
//
//	/// Normalize the result to [ 0, histImage.rows ]
//	normalize(ahist, ahist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//
//	/// Draw for each channel
//	for (int i = 1; i < histSize; i++)
//	{
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(ahist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(ahist.at<float>(i))),
//			Scalar(255, 0, 0), 2, 8, 0);
//	}
//
//	/// Display
//	namedWindow("calcHist", CV_WINDOW_AUTOSIZE);
//	imshow("calcHist", histImage);
//
//	return ahist;
//}
//
//int CalcAngle(Mat orig, Mat mask, Mat cannyResult,int componentCount)
//{
//#pragma region For Testing Purposes ONLY
//	ofstream CalcAngle_OrigMatrixFile;
//	CalcAngle_OrigMatrixFile.open("CalcAngle_OrigMatrixFile.csv");
//	CalcAngle_OrigMatrixFile << orig << "\n";
//	CalcAngle_OrigMatrixFile.close();
//
//	ofstream CalcAngle_MaskMatrixFile;
//	CalcAngle_MaskMatrixFile.open("CalcAngle_MaskMatrixFile.csv");
//	CalcAngle_MaskMatrixFile << mask << "\n";
//	CalcAngle_MaskMatrixFile.close();
//
//	ofstream CalcAngle_CannyResultMatrixFile;
//	CalcAngle_CannyResultMatrixFile.open("CalcAngle_CannyResultMatrixFile.csv");
//	CalcAngle_CannyResultMatrixFile << cannyResult << "\n";
//	CalcAngle_CannyResultMatrixFile.close();
//#pragma endregion
//
//
//	std::string count = std::to_string(componentCount);
//
//	//imshow("Original Image", orig);
//	ofstream myEdgeDetectorFile;
//	myEdgeDetectorFile.open("myEdgeDetectorFile.txt");
//
//#pragma region For Testing Purposes ONLY
//	//![load]
//	//src = imread(argv[1], IMREAD_COLOR); // Load an image
//	//src = imread("contour_1.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
//	//IplImage* img = cvLoadImage("20161215 02.33_368L.jpg");
//	//imshow("CalcAngleSourceImage_"+ componentCount, src);
//	//ofstream CalcAngle_ContourMatrixFile;
//	//CalcAngle_ContourMatrixFile.open("CalcAngle_" + count + "_SourceMatrixFile.csv");
//	//CalcAngle_ContourMatrixFile << src << "\n";
//	//CalcAngle_ContourMatrixFile.close();
//
//	//if (src.empty())
//	//{
//	//	//return -1;
//	//}
//	//![load]
//
//	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
//
//	//angle_src_gray = mask;
//	myEdgeDetectorFile << "CalcAngle_src_gray_"+ count +"= " << angle_src_gray << "\n";
//	myEdgeDetectorFile << "CalcAngle_src_gray_"+ count+" size= " << angle_src_gray.size() << "\n";
//	myEdgeDetectorFile << "\n";    myEdgeDetectorFile << "\n";
//
//	//detected_edges = angle_src_gray;
//#pragma endregion
//
//#pragma region Original GrayScale
//	Mat orig_gray;
//	cvtColor(orig, orig_gray, CV_BGR2GRAY);
//#pragma endregion
//
//#pragma region Sobel Dx
//	/// Gradient X
//	Sobel(mask, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//#pragma endregion
//#pragma region Write Dx To File
//	ofstream GradXMatrixFile;
//	GradXMatrixFile.open("GradXMatrixFile_" + count + ".csv");
//	GradXMatrixFile << grad_x << "\n";
//	GradXMatrixFile.close();
//	convertScaleAbs(grad_x, abs_grad_x);
//#pragma endregion
//
//
//#pragma region Sobel Dy
//	/// Gradient Y
//	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//
//	Sobel(mask, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//#pragma endregion
//#pragma region Write Dy To File
//	ofstream GradYMatrixFile;
//	GradYMatrixFile.open("GradYMatrixFile_"+ count+".csv");
//	GradYMatrixFile << grad_y << "\n";
//	GradYMatrixFile.close();
//	convertScaleAbs(grad_y, abs_grad_y);
//
//	//![sobel]
//	//convertScaleAbs(grad_x, abs_grad_x);
//	//convertScaleAbs(grad_y, abs_grad_y);
//	//cv::Mat angle(src.size(), CV_64F);
//#pragma endregion
//
//	Mat Mag(mask.size(), CV_32FC1);
//	Mat Angle(mask.size(), CV_32FC1);
//
//	//cv::Mat angle(image.size(), CV_64F)
//	cout << "grad_x_"+ count+ "size= " << grad_x.size() << "\n";
//	cout << "grad_x_"+ count+" depth= " << grad_y.depth() << "\n";
//	cout << "grad_y_"+ count+" size= " << grad_y.size() << "\n";
//	cout << "grad_y_"+ count+" depth= " << grad_y.depth() << "\n";
//	cout << "Angle_"+ count+" size= " << Angle.size() << "\n";
//	cout << "Angle_"+ count+" depth= " << Angle.depth() << "\n";
//	cout << "detected_edges_"+ count+" size= " << detected_edges.size() << "\n";
//	cout << "detected_edges_"+ count+" depth= " << detected_edges.depth() << "\n";
//	ofstream GradsAndAtansMatrixFile;
//	GradsAndAtansMatrixFile.open("GradsAndAtans+"+ count+"_MatrixFile.txt");
//
//	double minM, maxM, minA, maxA;
//
//#pragma region Mag and Angle
//	//phase(grad_x, grad_y, Angle, true);
//	cartToPolar(grad_x, grad_y, Mag, Angle, true);
//#pragma endregion
//
//
//	myEdgeDetectorFile << "src size= " << mask.size() << "\n";
//	myEdgeDetectorFile << "Angle size= " << Angle.size() << "\n";
//
//	std::array<std::vector<int>, 72> vvv{ {} };
//
//#pragma region For Testing Purposes ONLY
//	ofstream CannyMagGradOrig_0M_MatrixFile;
//	CannyMagGradOrig_0M_MatrixFile.open("CannyMagGradOrig_0M_" + count + "MatrixFile.csv");
//	ofstream CannyMagGradOrig_0A_MatrixFile;
//	CannyMagGradOrig_0A_MatrixFile.open("CannyMagGradOrig_0A_" + count + "MatrixFile.csv");
//	ofstream CannyMagGradOrig_0C_MatrixFile;
//	CannyMagGradOrig_0C_MatrixFile.open("CannyMagGradOrig_0C_" + count + "MatrixFile.csv");
//
//	ofstream CannyMagGradOrig_Mag_MatrixFile;
//	CannyMagGradOrig_Mag_MatrixFile.open("CannyMagGradOrig_Mag_" + count + "MatrixFile.csv");
//	ofstream CannyMagGradOrig_Angle_MatrixFile;
//	CannyMagGradOrig_Angle_MatrixFile.open("CannyMagGradOrig_Angle_" + count + "MatrixFile.csv");
//	ofstream CannyMagGradOrig_Canny_MatrixFile;
//	CannyMagGradOrig_Canny_MatrixFile.open("CannyMagGradOrig_Canny_" + count + "MatrixFile.csv");
//
//	ofstream CannyMagGradOrig_newAngle_MatrixFile;
//	CannyMagGradOrig_newAngle_MatrixFile.open("CannyMagGradOrig_newAngle_" + count + "MatrixFile.csv");
//
//	ofstream CalcAngle_Container_MatrixFile;
//	CalcAngle_Container_MatrixFile.open("CalcAngle_Container_" + count + "_MatrixFile.txt");
//
//	ofstream CalcAngle_Histogram_MatrixFile;
//	CalcAngle_Histogram_MatrixFile.open("CalcAngle_Histogram_" + count + "_MatrixFile.csv");
//#pragma endregion
//
//#pragma region Extract Edge angles
//	Mat newAngle = Mat(cannyResult.size().width, cannyResult.size().height, CV_64F, 0.0);;
//	struct element {
//		int bin;
//		int i;
//		int j;
//		int angle;
//		int value;
//	};
//	vector<element> container;
//	int containerCount = 0;
//
//	///Walk Canny (edge)
//	for (size_t i = 0; i < cannyResult.rows; i++)
//	{
//		const float* aRow_i = Angle.ptr<float>(i);
//		const float* mRow_i = Mag.ptr<float>(i);
//		//		const double* cRow_i = (double)cannyResult.at<uchar>(i);
//
//		for (size_t j = 0; j < cannyResult.cols; j++)
//		{
//			/// Establish the number of bins
//			int histSize = 256;
//			myEdgeDetectorFile << "Angle(i,j)= (" << i << ", " << j << ")= " << aRow_i[j] << "\n";
//
//			myEdgeDetectorFile << "Row_i[j]/binSize= (" << aRow_i[j] << ", " << binSize << ")= " << int(aRow_i[j] / binSize) << "\n";
//
//			myEdgeDetectorFile << "src_gray[i=" << i << ",j=" << j << "]=" << aRow_i[j] << "\n";
//
//			//vvv[int(aRow_i[j] / binSize)].push_back(aRow_i[j]);
//
//			myEdgeDetectorFile << "\n";
//			myEdgeDetectorFile << "\n";
//			myEdgeDetectorFile << "\n";
//			/// For all non-zero edge pixels
//			if ((int)cannyResult.at<uchar>(i, j) != 0)
//			{
//				///Get pixel from Angle at Canny-corresponding-coordinates
//				newAngle.at<double>(i, j) = (double)aRow_i[j];
//				///Storing angle information ([x,y],bin,angle-value,Canny-value)
//				CalcAngle_Histogram_MatrixFile << "newAngle(i,j)= (" << i << ", " << j << ")= " << newAngle.at<double>(i, j) << "\n";
//				container.push_back(element());
//				container[containerCount].bin = int(newAngle.at<double>(i, j) / binSize);
//				container[containerCount].i = i;
//				container[containerCount].j = j;
//				container[containerCount].angle = newAngle.at<double>(i, j);
//				container[containerCount].value = (int)cannyResult.at<uchar>(i, j);
//				CalcAngle_Container_MatrixFile << containerCount << "," << container[containerCount].bin << "," << container[containerCount].i << "," << container[containerCount].j << "," << container[containerCount].angle << "," << container[containerCount].value << "\n";
//				containerCount++; ///To index into container
//				vvv[int(newAngle.at<double>(i, j) / binSize)].push_back(newAngle.at<double>(i, j));
//				cout << "vvv["<< int(newAngle.at<double>(i, j) / binSize) <<"] = "<< (int)newAngle.at<double>(i, j)<<"\n";
//				CannyMagGradOrig_Canny_MatrixFile << "(i;j)= (" << i << ";" << j << ")= " << "," << (int)cannyResult.at<uchar>(i, j) << "\n";
//				CannyMagGradOrig_Angle_MatrixFile << "(i;j)= (" << i << ";" << j << ")= " << "," << aRow_i[j] << "\n";
//				CannyMagGradOrig_Mag_MatrixFile << "(i;j)= (" << i << ";" << j << ")= " << "," << mRow_i[j] << "\n";
//			}
//			else
//			{
//				CannyMagGradOrig_0C_MatrixFile << "(i;j)= (" << i << ";" << j << ")= " << "," << (int)cannyResult.at<uchar>(i, j) << "\n";
//				CannyMagGradOrig_0A_MatrixFile << "(i;j)= (" << i << ";" << j << ")= " << "," << aRow_i[j] << "\n";
//				CannyMagGradOrig_0M_MatrixFile << "(i;j)= (" << i << ";" << j << ")= " << "," << mRow_i[j] << "\n";
//			}
//		}
//	}
//#pragma endregion
//
//#pragma region For Testing Purposes ONLY
//	CannyMagGradOrig_newAngle_MatrixFile << newAngle << "\n";
//	CannyMagGradOrig_Canny_MatrixFile.close();
//	CannyMagGradOrig_newAngle_MatrixFile.close();
//	CannyMagGradOrig_Mag_MatrixFile.close();
//	CannyMagGradOrig_Angle_MatrixFile.close();
//	CannyMagGradOrig_0M_MatrixFile.close();
//	CannyMagGradOrig_0A_MatrixFile.close();
//	CannyMagGradOrig_0C_MatrixFile.close();
//	CalcAngle_Container_MatrixFile.close();
//	CalcAngle_Histogram_MatrixFile.close();
//
//	cout << "newAngle.size()= " << newAngle.size() << "\n";
//	cout << "Angle.size()= " << Angle.size() << "\n";
//	cout << "Canny.size()= " << cannyResult.size() << "\n";
//	cout << "src.size()= " << orig.size() << "\n";
//	cout << "Mask.size()= " << mask.size() << "\n";
//
//	//ofstream CalcAngle_Histogram_MatrixFile;
//	//CalcAngle_Histogram_MatrixFile.open("CalcAngle_Histogram_" + count + "_MatrixFile.csv");
//
//	//ofstream CalcAngle_Container_MatrixFile;
//	//CalcAngle_Container_MatrixFile.open("CalcAngle_Container_" + count + "_MatrixFile.txt");
//#pragma endregion
//
//	//struct element {
//	//	int bin;
//	//	int i;
//	//	int j;
//	//	int angle;
//	//	int value;
//	//};
//	//vector<element> container;
//	//int containerCount = 0;
//	/////Sort Angle according to histogram
//	//for (size_t i = 0; i < newAngle.rows; i++)
//	//{
//	//	for (size_t j = 0; j < newAngle.cols; j++)
//	//	{				
//	//		double newAngle2Row_ij = newAngle.at<double>(i, j);
//	//		if (newAngle2Row_ij !=0)
//	//		{
//	//			CalcAngle_Histogram_MatrixFile << "newAngle(i,j)= (" << i << ", " << j << ")= " << newAngle2Row_ij << "\n";
//	//			container.push_back(element());
//	//			container[containerCount].bin = int(newAngle2Row_ij / binSize);
//	//			container[containerCount].i = i;
//	//			container[containerCount].j = j;
//	//			container[containerCount].angle = newAngle2Row_ij;
//	//			container[containerCount].value = (int)cannyResult.at<uchar>(i, j);
//	//			
//	//			CalcAngle_Container_MatrixFile << containerCount<<","<< container[containerCount].bin <<","  << container[containerCount].i << "," << container[containerCount].j << "," << container[containerCount].angle << "," << container[containerCount].value<<"\n";
//	//			containerCount++;
//
//	//			vvv[int(newAngle2Row_ij / binSize)].push_back(newAngle2Row_ij);
//	//		}
//	//	}
//	//}
//	//std::copy(container.begin(), container.end(), std::ostream_iterator<element>(CalcAngle_Container_MatrixFile<< " "));
//	//CalcAngle_Container_MatrixFile << (Mat)container << "\n";
//	//CalcAngle_Container_MatrixFile.close();
//	//CalcAngle_Histogram_MatrixFile.close();
//
//#pragma region Store all angle frequencies
//	ofstream ArrayVectorFile;
//	ArrayVectorFile.open("ArrayVector" + count + "File.txt");
//	std::array<std::vector<int>, 72> vvvCounts{ {} };
//	ofstream CalcAngle_vvvCounts_MatrixFile;
//	CalcAngle_vvvCounts_MatrixFile.open("CalcAngle_vvvCounts_" + count + "MatrixFile.csv");
//	int temp = 0;
//	struct MaxElement {
//		int bin;
//		int value = 0;
//	};
//	MaxElement me;
//	for (int i = 0; i < vvv.size(); i++)
//	{
//		for (int j = 0; j < vvv[i].size(); j++)
//		{
//			ArrayVectorFile << "i= " << i << ", j= " << j << ", value= " << vvv[i].at(j) << "\n";
//			if (vvvCounts[i].empty())	///If it's the first one
//			{	///Store angle frequency
//				vvvCounts[i].push_back(vvv[i].size());	///vvv[i].size() = Number angle of occurences			
//				if (vvvCounts[i].at(0) > temp)	///Find bin with the most elements
//				{
//					temp = vvvCounts[i].at(0);
//					me.bin = i;	///Bin with most elements (bin ID)
//					me.value = vvvCounts[i].at(0);	///Frequency count
//				}
//				CalcAngle_vvvCounts_MatrixFile << "vvvCounts[" << i << "].at[" << j << "]= " << vvvCounts[i].at(j) << "\n";
//			}
//		}
//	}
//	CalcAngle_vvvCounts_MatrixFile.close();
//	ArrayVectorFile.close();
//	cout << "The biggest number is: " << me.value << " at bin " << me.bin << endl;
//#pragma endregion
//
//Mat GraySrcImg;
//cv::cvtColor(orig, GraySrcImg, cv::COLOR_BGR2GRAY);
//Mat temp2Orig = GraySrcImg;
//#pragma region Plot Back to Canny
//	Mat tempCannyResult = cannyResult;
//	for (size_t i = 0; i < cannyResult.rows; i++)
//	{
//		for (size_t j = 0; j < cannyResult.cols; j++)
//		{
//			if ((int)cannyResult.at<uchar>(i, j) != 0)
//			{
//				//for (size_t cc = 0; cc < container.size(); cc++)
//				//{
//				//	if (container[cc].i == i && container[cc].j == j)
//				//	{
//						//tempCannyResult.at<uchar>(i, j) = 100;
//						temp2Orig.at<uchar>(i, j) = 100;
//				//	}
//				//}
//
//			}
//		}
//	}
//	//imshow("tempCannyResult", tempCannyResult);
//	//imwrite("tempCannyResult.bmp", tempCannyResult);
//	imshow("temp2Orig", temp2Orig);
//	imwrite("temp2Orig.bmp", temp2Orig);
//#pragma endregion
//
//
//#pragma region Plot Back to Mask
//	Mat tempMask = mask;
//	int similar = 0;
//	for (size_t i = 0; i < mask.rows; i++)
//	{
//		for (size_t j = 0; j < mask.cols; j++)
//		{
//			if ((int)mask.at<uchar>(i, j) != 0)
//			{
//				for (size_t cc = 0; cc < container.size(); cc++)
//				{
//					if (container[cc].i == i && container[cc].j == j)
//					{
//						tempMask.at<uchar>(i, j) = 100;
//						similar++;
//					}
//					else
//					{
//						similar--;
//						//cout << "Not similar at i= " << i << ", j= " << j << "\n";
//					}
//				}
//
//			}
//		}
//	}
//	imshow("tempMask", tempMask);
//	imwrite("tempMask.bmp", tempMask);
//#pragma endregion
//
//
//#pragma region Plot Back onto Original
//	Mat tempOrig = GraySrcImg;
//	for (size_t i = 0; i < GraySrcImg.rows; i++)
//	{
//		for (size_t j = 0; j < GraySrcImg.cols; j++)
//		{
//			if ((int)GraySrcImg.at<uchar>(i, j) != 0)
//			{
//				for (size_t cc = 0; cc < container.size(); cc++)
//				{
//					
//					if (container[cc].i == i && container[cc].j == j)
//					{
//						//cout << "cc= " << cc << "\n";
//						//cout << "(i= " << i << " ,j= " << j << ")" << "\n";
//						tempOrig.at<uchar>(i, j) = 255;
//					}
//					else
//					{
//
//					}
//				}
//
//			}
//		}
//	}
//	imshow("tempOrig", tempOrig);
//	imwrite("tempOrig.jpg", tempOrig);
//#pragma endregion
//
//
//	myEdgeDetectorFile.close();
//
//	//waitKey(0);
//	return 0;
//}
//
//int main(int argc, char *argv[])
//{
//	ofstream myConnectedComponents03file;
//	myConnectedComponents03file.open("myConnectedComponents03file.txt");
//
//#pragma region Could Be Deleted
//	cv::CommandLineParser parser(argc, argv, keys);
//	parser.about("");
//	if (parser.has("help")) {
//		parser.printMessage();
//		return 0;
//	}
//#pragma endregion
//
//	cv::Mat srcImg;
//
//#pragma region Could Be Deleted
//	std::string file = parser.get<std::string>(0);
//	if (file == "j") {
//		file = parser.get<std::string>("j");
//	}
//	else if (file == "contours") {
//		file = parser.get<std::string>("contours");
//	}
//#pragma endregion
//
//	srcImg = cv::imread("20161215 02.33_368L.jpg");
//	if (srcImg.empty()) {
//		return -1;
//	}
//
//	string filter;		
//	Mat fltrImg;
//	cv::Mat fltrGrayImg;
//	///Filtering code commented out
//	//for (int i = 0; i < 2; i++)
//	//{
//	//	myConnectedComponents03file << "i= " << i << std::endl;
//	//	if (i==0)
//	//	{
//			fltrImg = srcImg;
//			filter = "Original";
//
//	//	}
//	//	else if (i==1)
//	//	{
//			////Apply median filter
//			//blur(srcImg, fltrImg, Size(5, 5), Point(-1, -1));
//			//filter = "median filter result";
//	//	}
//	//	else if (i==2)
//	//	{
//	//		//Apply bilateral filter
//	//		bilateralFilter(srcImg, fltrImg, 15, 80, 80);
//	//		filter = "bilateralFilter result";
//	//	}
//	//	else
//	//	{
//	//		//fltrImg = srcImg;
//	//		//filter = "Original";
//	//	}
//		//fltrImg = srcImg;
//		//filter = "src image";
//		//imshow(filter, fltrImg);
//		imwrite(filter+".bmp", fltrImg);
//#pragma region GrayScale Image
//		cv::cvtColor(fltrImg, fltrGrayImg, cv::COLOR_BGR2GRAY);
//#pragma endregion
//		//fltrGrayImg = srcImg;
//		//imshow("gray-scale image", fltrGrayImg);
//		//Mat FltrBinaryImg;
//		//cv::threshold(fltrGrayImg, FltrBinaryImg, 100, 255, cv::THRESH_BINARY_INV);
//#pragma region Binary Image
//		cv::Mat FltrBinaryImg = threshval < 128 ? (fltrGrayImg < threshval) : (fltrGrayImg > threshval);
//#pragma endregion
//
//		//imshow("binary image", FltrBinaryImg);
//		cv::Mat FltrLabelImage;
//		cv::Mat FltrStats, FltrCentroids;
//#pragma region Connected Components
//		int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
//#pragma endregion
//		std::string nFltrLabelsString = std::to_string(nFltrLabels);
//		//normalize(nFltrLabels, FltrLabelImage, 0, 255, NORM_MINMAX, CV_8U);
//		cv::Mat FltrLabelImage2;
//#pragma region Normalize Components
//		normalize(FltrLabelImage, FltrLabelImage2, 0, 255, NORM_MINMAX, CV_8U);
//#pragma endregion
//		//imshow("Labels", FltrLabelImage);
//		//myConnectedComponents03file << "nFltrLabels= " << nFltrLabels << std::endl;
//		//myConnectedComponents03file << "size of original image= " << fltrGrayImg.size() << std::endl;
//		//myConnectedComponents03file << "size of FltrLabelImage= " << FltrLabelImage.size() << std::endl;
//		//imshow("FltrLabelImage2", FltrLabelImage2);
//		std::vector<cv::Vec3b> FltrColors(nFltrLabels);
//		FltrColors[0] = cv::Vec3b(0, 0, 0);
//		myConnectedComponents03file << "(Filter) Number of connected components = " << nFltrLabels << std::endl << std::endl;
//		vector<vector<Point>> contours;
//		//vector<Vec4i> hierarchy;
//		for (int FltrLabel = 1; FltrLabel < 2/*nFltrLabels*/; ++FltrLabel) {
//			FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
//			//myConnectedComponents03file << "Component " << FltrLabel << std::endl;
//			//myConnectedComponents03file << "CC_STAT_LEFT   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_LEFT) << std::endl;
//			//myConnectedComponents03file << "CC_STAT_TOP    = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_TOP) << std::endl;
//			//myConnectedComponents03file << "CC_STAT_WIDTH  = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_WIDTH) << std::endl;
//			//myConnectedComponents03file << "CC_STAT_HEIGHT = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_HEIGHT) << std::endl;
//			//myConnectedComponents03file << "CC_STAT_AREA   = " << FltrStats.at<int>(FltrLabel, cv::CC_STAT_AREA) << std::endl;
//			//myConnectedComponents03file << "CENTER   = (" << FltrCentroids.at<double>(FltrLabel, 0) << "," << FltrCentroids.at<double>(FltrLabel, 1) << ")" << std::endl << std::endl;
//
//			// Get the mask for the i-th contour
//#pragma region Extract each Component (Mask)
//			Mat mask_i = FltrLabelImage == FltrLabel;
//#pragma endregion
//			string name = "mask_i_";
//			std::string s = std::to_string(FltrLabel);
//			//strcat("mask_i_", std::to_string(i));
//			//if (FltrLabel==1)
//			//{
//			//imshow("mask_i_" + s, mask_i);
//			imwrite("mask_i_" + s + ".bmp", mask_i);
//			ofstream Mask_MatrixFile;
//			Mask_MatrixFile.open("Mask_" + s + "_MatrixFile.csv");
//			Mask_MatrixFile << mask_i << "\n";
//			Mask_MatrixFile.close();
//			//}
//#pragma region Test Plotting over grayScale Source Image
//			//				//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//			//Mat tempSrcImg = fltrGrayImg;
//			//Mat tempMask = mask_i;
//			////cv::cvtColor(mask_i, tempMask, cv::COLOR_BGR2GRAY);
//			//for (size_t i = 0; i < mask_i.rows; i++)
//			//{
//			//	for (size_t j = 0; j < mask_i.cols; j++)
//			//	{
//			//		//myEdgeDetectorFile << "Angle(i,j)= (" << i << ", " << j << ")= " << aRow_i[j] << "\n";
//			//		if (mask_i.at<uchar>(i, j) != 0)
//			//		{
//			//			//container[containerCount].value = (int)cannyResult.at<uchar>(i, j);
//			//			//if (srcImg[cc].i == i && container[cc].j == j)
//			//			//{
//			//			tempSrcImg.at<uchar>(i, j) = 100;
//			//			tempMask.at<uchar>(i, j) = 100;
//			//			//}
//			//		}
//			//	}
//			//}
//			//imshow("tempSrcImg", tempSrcImg);
//			//imwrite("tempSrcImg.jpg", tempSrcImg);
//			//imshow("tempMask", tempMask);
//			//imwrite("tempMask.jpg", tempMask);
//			////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  
//#pragma endregion
//
//			//=======================================================================
//#pragma region Canny Edge Detection
//			 ///apply your filter
//			Mat CannyMaskResult;
//			Canny(mask_i, CannyMaskResult, 100, 200);
//#pragma endregion
//			imwrite("Cannymask_i_" + s + ".bmp", CannyMaskResult);
//			ofstream CannyMask_MatrixFile;
//			CannyMask_MatrixFile.open("CannyMask_" + s + "_MatrixFile.csv");
//			//CannyMask_MatrixFile.open("Mask_" + s + "_MatrixFile.csv");
//			CannyMask_MatrixFile << CannyMaskResult << "\n";
//			CannyMask_MatrixFile.close();
//			//}
//
//			/////walk the edge
//			//ofstream WalkCannyMask_MatrixFile;
//			//WalkCannyMask_MatrixFile.open("WalkCannyMask_" + s + "_MatrixFile.txt");
//			////WalkCannyMask_MatrixFile << mask_i << "\n";
//			//for (int r = 0; r < mask_i.rows; ++r) {
//			//	for (int c = 0; c < mask_i.cols; ++c) {
//			//		double value = (double)mask_i.at<uchar>(r, c);
//			//		if (value==255)
//			//		{
//			//			WalkCannyMask_MatrixFile << "(r=" << r << ", c=" << c << ")= " << (double)mask_i.at<uchar>(r, c)<<" HERE" << "\n";
//			//			WalkCannyMask_MatrixFile << "(r=" << r << ", c=" << c << ")= " << (double)srcImg.at<uchar>(r, c) << "\n";
//			//			
//			//			WalkCannyMask_MatrixFile << "(r=" << r << ", c=" << c << ")= " << (double)fltrGrayImg.at<uchar>(r, c) << "\n";
//			//		}
//			//		/*WalkCannyMask_MatrixFile << "(r="<<r<<", c="<<c<<")= "<<(double)mask_i.at<uchar>(r,c) << "\n";*/
//
//			//	}
//			//}
//			//WalkCannyMask_MatrixFile.close();
//#pragma region Angle Histogram Analysis
//			int temp = CalcAngle(srcImg, mask_i, CannyMaskResult, FltrLabel);
//#pragma endregion
//			/*findContours(mask_i.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//			Draw contours
//			RNG rng(12345);
//			Mat drawing = Mat::zeros(fltrGrayImg.size(), CV_8UC3);
//			for (int i = 0; i < contours.size(); i++)
//			{
//				//random colour for contour
//				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//				// contour
//				drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
//			}*/
//
//			//std::array<std::vector<int>, 72> vvv{ {} };
//			//for (size_t i = 0; i < vvv.size(); i++)
//			//{
//			//	vvv[i].push_back(0);
//			//	//cout << "vvv[" << i << "] = " << vvv[i].at(0) << "\n";
//			//}
//
//			//ofstream Test1MatrixFile;
//			//Test1MatrixFile.open("Test1MatrixFile.csv");
//			//for (size_t j = 0; j < 361; j++)
//			//{
//			//	Test1MatrixFile << "j= " << j << "\n";
//			//	vvv[(int)j / 5].pop_back();
//			//	vvv[(int)j / 5].push_back(j);
//			//}
//			//Test1MatrixFile.close();
//
//			//ofstream Test2MatrixFile;
//			//Test2MatrixFile.open("Test2MatrixFile.csv");
//			//for (size_t j = 0; j < vvv.size(); j++)
//			//{
//			//	Test2MatrixFile << "vvv[ " << (int)j / 5 << "] = " << vvv[(int)j / 5].at(0) << "\n";
//			//}
//			//Test2MatrixFile.close();
//
//			myConnectedComponents03file << "mask_i_" + s << "\n";
//			//imwrite("mask_i_"+s+".png", mask_i);
//			//myConnectedComponents03file << "i= " << i << std::endl;
//			//myConnectedComponents03file << "FltrLabelImage.size= " << FltrLabelImage.size() << std::endl;
//			//myConnectedComponents03file << "mask_i= " << mask_i.size() << std::endl;
//			// Compute the contour
//			//findContours(mask_i, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//			//findContours(mask_i, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//		}
//#pragma region Colour Calculated Components
//		cv::Mat FltrDst(fltrGrayImg.size(), CV_8UC3);
//		//cv::imshow("filterDisplayName", FltrDst);
//		for (int r = 0; r < FltrDst.rows; ++r) {
//			for (int c = 0; c < FltrDst.cols; ++c) {
//				int FltrLabel = FltrLabelImage.at<int>(r, c);
//				cv::Vec3b &FltrPixel = FltrDst.at<cv::Vec3b>(r, c);
//				FltrPixel = FltrColors[FltrLabel];
//			}
//		}
//		imshow(nFltrLabelsString+"-Connected Components", FltrDst);
//		imwrite("Connected Components.bmp", FltrDst);
//#pragma endregion
//
//	myConnectedComponents03file.close();
//
//	waitKey(0);
//
//
//	return 0;
//}