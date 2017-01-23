//// original code by http://stackoverflow.com/users/951860/mevatron
//// see http://stackoverflow.com/a/11157426/15485
//// http://stackoverflow.com/users/15485/uvts-cvs added the code for saving x and y gradient component 
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <iostream>
//#include <vector>
//
//#include <fstream>
//
//using namespace cv;
//using namespace std;
//
//Mat mat2gray(const cv::Mat& src)
//{
//	Mat dst;
//	normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
//
//	return dst;
//}
//
//Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0)
//{
//	Mat oriMap = Mat::zeros(ori.size(), CV_8UC3);
//	Vec3b red(0, 0, 255);
//	Vec3b cyan(255, 255, 0);
//	Vec3b green(0, 255, 0);
//	Vec3b yellow(0, 255, 255);
//
//	ofstream MatrixFile;
//	MatrixFile.open("orientationMapMatrixFile.txt");
//	//MatrixFile << ori << "\n";
//	MatrixFile << "mag.rows*mag.cols= "<< mag.rows*mag.cols << "\n";
//
//	for (int i = 0; i < mag.rows*mag.cols; i++)
//	{
//		//MatrixFile << "i= "<<i << "\n";
//		float* magPixel = reinterpret_cast<float*>(mag.data + i * sizeof(float));
//		//MatrixFile << "magPixel= " << magPixel << "\n";
//		//MatrixFile << "\n";
//		//MatrixFile << "\n";
//		//MatrixFile << "\n";
//		if (*magPixel > thresh)
//		{
//			//MatrixFile << "i= " << i << "\n";
//			float* oriPixel = reinterpret_cast<float*>(ori.data + i * sizeof(float));
//			//MatrixFile << "ori.data= " << ori.data << "\n";
//			//MatrixFile << "i * sizeof(float)= " << i * sizeof(float) << "\n";
//			//MatrixFile << "oriPixel= " << oriPixel << "\n";
//			//MatrixFile << "\n";
//			Vec3b* mapPixel = reinterpret_cast<Vec3b*>(oriMap.data + i * 3 * sizeof(char));
//			//MatrixFile << "oriMap.data= " << oriMap.data << "\n";
//			//MatrixFile << "i  * 3* sizeof(float)= " << i * 3 * sizeof(float) << "\n";
//			MatrixFile << "oriPixel= " << *oriPixel << "\n";
//
//			if (*oriPixel < 90.0)
//				*mapPixel = red; 
//			else if (*oriPixel >= 90.0 && *oriPixel < 180.0)
//				*mapPixel = cyan;
//			else if (*oriPixel >= 180.0 && *oriPixel < 270.0)
//				*mapPixel = green;
//			else if (*oriPixel >= 270.0 && *oriPixel < 360.0)
//				*mapPixel = yellow;
//		}
//	}
//	MatrixFile.close();
//	return oriMap;
//}
//
//int main(int argc, char* argv[])
//{
//	Mat image = Mat::zeros(Size(320, 240), CV_8UC1);
//	circle(image, Point(160, 120), 80, Scalar(255, 255, 255), -1, CV_AA);
//
//	imshow("original", image);
//
//	Mat Sx;
//	Sobel(image, Sx, CV_32F, 1, 0, 3);
//
//	Mat Sy;
//	Sobel(image, Sy, CV_32F, 0, 1, 3);
//
//	Mat mag, ori;
//	//magnitude(Sx, Sy, mag);
//	//phase(Sx, Sy, ori, true);
//	cartToPolar(Sx, Sy, mag, ori,true);
//	Mat oriMap = orientationMap(mag, ori, 1.0);
//
//	ofstream MatrixFile;
//	MatrixFile.open("oriMatrixFile.csv");
//	MatrixFile << ori << "\n";
//	MatrixFile.close();
//
//	ofstream magMatrixFile;
//	magMatrixFile.open("magMatrixFile.csv");
//	magMatrixFile << mag << "\n";
//	magMatrixFile.close();
//
//	imshow("x", mat2gray(Sx));
//	imshow("y", mat2gray(Sy));
//
//	imwrite("hor.png", mat2gray(Sx));
//	imwrite("ver.png", mat2gray(Sy));
//
//	imshow("magnitude", mat2gray(mag));
//	imshow("orientation", mat2gray(ori));
//	imshow("orientation map", oriMap);
//	waitKey();
//
//	return 0;
//}