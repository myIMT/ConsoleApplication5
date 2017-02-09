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
//Mat src,srcImg, GrayImg, hist, cannyEdge, detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;
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
//vector<Mat> maskImages;
//vector<Point> maskCentroid;
////----------------------------------------------------
//Mat GetConnectedComponent(Mat GrayScaleSrcImg)
//{
//	cv::Mat FltrBinaryImg = threshval < 128 ? (GrayScaleSrcImg < threshval) : (GrayScaleSrcImg > threshval);
//
//	cv::Mat FltrLabelImage;
//	cv::Mat FltrStats, FltrCentroids;
//
//	int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
//	
//	std::string nFltrLabelsString = std::to_string(nFltrLabels);
//
//	cv::Mat FltrLabelImage2;
//
//	normalize(FltrLabelImage, FltrLabelImage2, 0, 255, NORM_MINMAX, CV_8U);
//	//imshow("FltrLabelImage2", FltrLabelImage2);
//
//	std::vector<cv::Vec3b> FltrColors(nFltrLabels);
//	FltrColors[0] = cv::Vec3b(0, 0, 0);
//
//	for (int FltrLabel = 1; FltrLabel < nFltrLabels; ++FltrLabel) {
//		//FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
//		FltrColors[FltrLabel] = cv::Vec3b((255), (255), (255));
//		Mat mask_i = FltrLabelImage == FltrLabel;
//		if (mask_i.empty())      // please, *always check* resource-loading.
//		{
//			cerr << "mask_i is empty - can't be loaded!" << endl;
//			continue;
//		}
//		maskImages.push_back(mask_i);
//		maskCentroid.push_back(Point(FltrCentroids.at<double>(FltrLabel, 0), FltrCentroids.at<double>(FltrLabel, 1)));
//	}
//
//	cv::Mat FltrDst(GrayScaleSrcImg.size(), CV_8UC3);
//	for (int r = 0; r < FltrDst.rows; ++r) {
//		for (int c = 0; c < FltrDst.cols; ++c) {
//			int FltrLabel = FltrLabelImage.at<int>(r, c);
//			cv::Vec3b &FltrPixel = FltrDst.at<cv::Vec3b>(r, c);
//			FltrPixel = FltrColors[FltrLabel];
//		}
//	}
//	//imshow(nFltrLabelsString + "-Connected Components", FltrDst);
//	//imwrite("Connected Components.bmp", FltrDst);
//
//	return FltrDst;
//}
//
//int main(int argc, char *argv[])
//{
//	src = cv::imread("20161215 02.33_368L2.jpg");
//	imshow("src", src);
//	//srcImg = src;
//		//bilateralFilter(src, srcImg, 15, 80, 80);
//	blur(src, srcImg, Size(5, 5), Point(-1, -1));
//	imshow("srcImg", srcImg);
//	cv::cvtColor(srcImg, GrayImg, cv::COLOR_BGR2GRAY);
//	imshow("GrayImg", GrayImg);
//
//	Mat components = GetConnectedComponent(GrayImg);
//	imshow("components", components);
//
//	Mat GrayComponents;
//	cvtColor(components, GrayComponents, COLOR_BGR2GRAY);
//	imshow("GrayComponents", GrayComponents);
//
//	Sobel(GrayComponents, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//	Sobel(GrayComponents, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//
//	Mat Mag(GrayComponents.size(), CV_32FC1);
//	Mat Angle(GrayComponents.size(), CV_32FC1);
//	cartToPolar(grad_x, grad_y, Mag, Angle, true);
//
//	std::array<std::vector<int>, 72> vvv{ {} };
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
//	Canny(GrayComponents, cannyEdge, 100, 200);
//	imshow("cannyEdge", cannyEdge);
//
//	Mat newAngle = Mat(Angle.size().height, Angle.size().width, Angle.type(), Scalar(0, 0, 0));
//
//	for (size_t i = 0; i < cannyEdge.rows; i++)
//	{
//		for (size_t j = 0; j < cannyEdge.cols; j++)
//		{
//			if ((int)cannyEdge.at<uchar>(i, j) != 0)
//			{
//				newAngle.ptr<float>(i)[j] = Angle.ptr<float>(i)[j];
//				container.push_back(element());
//				container[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);
//				container[containerCount].i = i;
//				container[containerCount].j = j;
//				container[containerCount].angle = newAngle.ptr<float>(i)[j];
//				container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);
//				containerCount++;
//			}
//		}
//	}
//	//////999999999999999999999999999999999999999999999999999999
//	ofstream ContainerFile;
//	ContainerFile.open("ContainerFile.txt");
//	for (int i = 0; i < container.size(); i++)
//	{
//		ContainerFile << "container[" << i << "].bin= " << container[i].bin << "\n";
//		ContainerFile << "container[" << i << "].i= " << container[i].i << "\n";
//		ContainerFile << "container[" << i << "].j= " << container[i].j << "\n";
//		ContainerFile << "container[" << i << "].angle= " << container[i].angle << "\n";
//		ContainerFile << "container[" << i << "].value= " << container[i].value << "\n";
//		ContainerFile << "\n";
//		ContainerFile << "\n";
//	}
//	ContainerFile.close();
//	//////999999999999999999999999999999999999999999999999999999
//	int maxCount = 0;
//	struct maxCountStruct {
//		int bin;
//		int angle;
//		int size;
//	};
//	vector<maxCountStruct> maxCountContainer;
//	int temp = 0;
//	struct MaxElementStruct {
//		int bin;
//		int angle;
//		int size;
//	};
//	MaxElementStruct mes;
//	for (size_t l = 0; l < container.size(); l++)
//	{
//		if (maxCountContainer.empty())
//		{
//			maxCountContainer.push_back(maxCountStruct());
//			maxCountContainer[l].bin = container[l].bin;
//			maxCountContainer[l].angle = container[l].angle;
//			maxCountContainer[l].size += 1;
//		}
//		else
//		{
//			for (size_t m = 0; m < maxCountContainer.size(); m++)
//			{
//				if (maxCountContainer[m].bin == container[l].bin)
//				{
//					maxCountContainer[m].size += 1;
//					break;
//				}
//				else if (m == maxCountContainer.size() - 1)
//				{
//					maxCountContainer.push_back(maxCountStruct());
//					maxCountContainer[maxCountContainer.size() - 1].bin = container[l].bin;
//					maxCountContainer[maxCountContainer.size() - 1].angle = container[l].angle;
//					maxCountContainer[maxCountContainer.size() - 1].size += 1;
//					break;
//				}
//
//				if (maxCountContainer[m].size > temp)	///Find bin with the most elements
//				{
//					temp = maxCountContainer[m].size;
//					mes.bin = (int)maxCountContainer[m].bin;	///Bin with most elements (bin ID)
//					mes.angle = (int)maxCountContainer[m].angle;
//					mes.size = (int)maxCountContainer[m].size;
//				}
//			}
//		}
//	}
//	cout << "The biggest number is: " << mes.size << " at bin " << mes.bin << endl;
//	cout << "Angle= " << mes.angle << "\n";
//	Mat tempGraySrc = GrayImg;
//	for (size_t n = 0; n < container.size(); n++)
//	{
//		if (container[n].bin == mes.bin)
//		{
//			tempGraySrc.at<uchar>(container[n].i, container[n].j) = 255;
//		}
//	}
//	imshow("tempGraySrc", tempGraySrc);
//	imwrite("tempGraySrc.bmp", tempGraySrc);
//
//#pragma region Bounding Box
//	vector<double> lengths(4);
//	double rectSize_b;
//	size_t imgCount = 0;
//	cout << "maskImages.size()= " << maskImages.size() << "\n";
//	for (imgCount; imgCount < maskImages.size(); imgCount++)
//	{
//		Mat tempPoints;
//		findNonZero(maskImages[imgCount], tempPoints);
//		Points.push_back(tempPoints);
//	}
//	Point2f vtx[4];
//	RotatedRect box = minAreaRect(Points[0]); //only the first Mat Points
//	box.points(vtx);
//	Mat tempSrc1 = imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//	for (int i = 0; i < 4; i++)
//	{
//		line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
//		lengths.push_back(norm((vtx[(i + 1) % 4]) - (vtx[i])));
//	}
//	imshow("Bounding Box", tempSrc1);
//	cout << "minAreaRect Angle= " << box.angle + 180 << "\n";
//#pragma endregion
//	Mat plotImage = src;
//	circle(plotImage, maskCentroid[0], 1, Scalar(0, 255, 0), 1, 8, 0);
//
//#pragma region walk in edge angle direction
//	Point2f u, u2, u22, v;
//	Point2f w1, w2;
//	u = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
//	u2 = u;
//	rectSize_b = *max_element(lengths.begin(), lengths.end());
//	double d = 0.1*rectSize_b;
//	double normU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
//	v = Point2f(u.x / normU, u.y / normU);
//
//	for (size_t i = 0; i < 10; i++)
//	{
//		if (i == 0)
//		{
//			w1.x = maskCentroid[0].x + v.x*d;
//			w1.y = maskCentroid[0].y + v.y*d;
//
//			w2.x = maskCentroid[0].x - v.x*d;
//			w2.y = maskCentroid[0].y - v.y*d;
//		}
//		else
//		{
//			w1.x = u2.x + v.x*d;
//			w1.y = u2.y + v.y*d;
//
//			w2.x = u22.x - v.x*d;
//			w2.y = u22.y - v.y*d;
//		}
//		cout << "i - " << i << "2-Plot here= " << w1 << ", " << w2 << "\n";
//		circle(plotImage, w1, 1, Scalar(0, 0, 255), 1, 8, 0);
//		//circle(plotImage, w2, 1, Scalar(255, 0, 0), 1, 8, 0);
//		u2 = w1;
//		u22 = w2;
//	}
//#pragma endregion
//
//#pragma region walk perpendicular in edge angle direction
//		Point2f uu, uu2, uu22, vv;
//		Point2f ww1, ww2;
//		uu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
//		uu2 = uu;
//		rectSize_b = *max_element(lengths.begin(), lengths.end());
//		//double dd = 0.1*rectSize_b;
//		double normUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
//		vv = Point2f(uu.x / normUU, uu.y / normUU);
//		//rotate and swap
//		double tempXX = vv.x;
//		vv.x = -vv.y;
//		vv.y = tempXX;
//		for (size_t i = 0; i < 10; i++)
//		{
//			if (i == 0)
//			{
//				ww1.x = maskCentroid[0].x + vv.x*d;
//				ww1.y = maskCentroid[0].y + vv.y*d;
//
//				ww2.x = maskCentroid[0].x - vv.x*d;
//				ww2.y = maskCentroid[0].y - vv.y*d;
//			}
//			else
//			{
//				ww1.x = uu2.x + vv.x*d;
//				ww1.y = uu2.y + vv.y*d;
//				//cout << "minAreaRect Angle= " << box.angle + 180 << "\n";
//				ww2.x = uu22.x - vv.x*d;
//				ww2.y = uu22.y - vv.y*d;
//				cout << "ww2= " << ww2 << "\n";
//			}
//			cout << "i - " << i << "1-Plot here= " << ww1 << ", " << ww2 << "\n";
//			//circle(src, ww1, 1, Scalar(0, 255, 255), 1, 8, 0); //yellow
//			//circle(plotImage, ww2, 1, Scalar(255, 255, 0), 1, 8, 0); //turqoise
//			uu2 = ww1;
//			uu22 = ww2;
//		}
//#pragma endregion
//
//		imshow("Plot Image", src);
//
//		waitKey(0);
//		return 0;
//}