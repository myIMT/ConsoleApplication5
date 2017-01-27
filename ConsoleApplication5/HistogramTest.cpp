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
Mat src,srcImg, GrayImg, hist, cannyEdge, detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;

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
//----------------------------------------------------
Mat GetConnectedComponent(Mat GrayScaleSrcImg)
{
	cv::Mat FltrBinaryImg = threshval < 128 ? (GrayScaleSrcImg < threshval) : (GrayScaleSrcImg > threshval);

	cv::Mat FltrLabelImage;
	cv::Mat FltrStats, FltrCentroids;

	int nFltrLabels = cv::connectedComponentsWithStats(FltrBinaryImg, FltrLabelImage, FltrStats, FltrCentroids, 8, CV_32S);
	
	std::string nFltrLabelsString = std::to_string(nFltrLabels);

	cv::Mat FltrLabelImage2;

	normalize(FltrLabelImage, FltrLabelImage2, 0, 255, NORM_MINMAX, CV_8U);
	//imshow("FltrLabelImage2", FltrLabelImage2);

	std::vector<cv::Vec3b> FltrColors(nFltrLabels);
	FltrColors[0] = cv::Vec3b(0, 0, 0);

	for (int FltrLabel = 1; FltrLabel < nFltrLabels; ++FltrLabel) {
		//FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
		FltrColors[FltrLabel] = cv::Vec3b((255), (255), (255));
		Mat mask_i = FltrLabelImage == FltrLabel;

		
	}

	cv::Mat FltrDst(GrayScaleSrcImg.size(), CV_8UC3);
	for (int r = 0; r < FltrDst.rows; ++r) {
		for (int c = 0; c < FltrDst.cols; ++c) {
			int FltrLabel = FltrLabelImage.at<int>(r, c);
			cv::Vec3b &FltrPixel = FltrDst.at<cv::Vec3b>(r, c);
			FltrPixel = FltrColors[FltrLabel];
		}
	}
	//imshow(nFltrLabelsString + "-Connected Components", FltrDst);
	//imwrite("Connected Components.bmp", FltrDst);

	return FltrDst;
}

int main(int argc, char *argv[])
{
	src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg");
	imshow("src", src);
	//srcImg = src;
		//bilateralFilter(src, srcImg, 15, 80, 80);
	blur(src, srcImg, Size(5, 5), Point(-1, -1));
		imshow("srcImg", srcImg);
	//	imwrite("BlurFilSrcImg.bmp", srcImg);
	cv::cvtColor(srcImg, GrayImg, cv::COLOR_BGR2GRAY);
	imshow("GrayImg", GrayImg);
	//imwrite("GrayImg.bmp", GrayImg);

	Mat components = GetConnectedComponent(GrayImg);
	imshow("components", components);
	//imwrite("components.bmp", components);
	//ofstream ComponentsFile;
	//ComponentsFile.open("ComponentsFile.csv");
	//ComponentsFile << components;
	//ComponentsFile.close();

	Mat GrayComponents;
	cvtColor(components, GrayComponents, COLOR_BGR2GRAY);
	imshow("GrayComponents", GrayComponents);
	//imwrite("GrayComponents.bmp", GrayComponents);
	//ofstream GrayComponentsFile;
	//GrayComponentsFile.open("GrayComponentsFile.csv");
	//GrayComponentsFile << GrayComponents;
	//GrayComponentsFile.close();

	Sobel(GrayComponents, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//imwrite("grad_x.bmp", grad_x);
	Sobel(GrayComponents, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	//imwrite("grad_y.bmp", grad_y);

	Mat Mag(GrayComponents.size(), CV_32FC1);
	Mat Angle(GrayComponents.size(), CV_32FC1);
	cartToPolar(grad_x, grad_y, Mag, Angle, true);
	//imwrite("Mag.bmp", Mag);
	//imwrite("Angle.bmp", Angle);
	//ofstream AngleFile;
	//AngleFile.open("AngleFile.csv");
	//AngleFile << Angle;
	//AngleFile.close();

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
	imshow("cannyEdge", cannyEdge);
	//imwrite("cannyEdge.bmp", cannyEdge);
	//ofstream CannyFile;
	//CannyFile.open("CannyFile.csv");
	//CannyFile << cannyEdge;
	//CannyFile.close();

	Mat newAngle = Mat(Angle.size().height, Angle.size().width,Angle.type(), Scalar(0, 0, 0));
	//
	//ofstream TestFile;
	//TestFile.open("TestFile.csv");
	for (size_t i = 0; i < cannyEdge.rows; i++)
	{
		//const float* aRow_i = Angle.ptr<float>(i);

		for (size_t j = 0; j < cannyEdge.cols; j++)
		{
			if ((int)cannyEdge.at<uchar>(i, j) != 0)
			{
				//newAngle.at<double>(i, j) = (double)aRow_i[j];
				//newAngle.ptr<float>(i)[j] = (double)aRow_i[j];
				newAngle.ptr<float>(i)[j] = Angle.ptr<float>(i)[j];
				//newAngle.at<double>(i, j) = Angle.at<double>(i, j);
				//TestFile << "Angle.at<double>(i, j)= " << Angle.at<double>(i, j) << "\n";
				container.push_back(element());
				container[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);
				container[containerCount].i = i;
				container[containerCount].j = j;
				container[containerCount].angle = newAngle.ptr<float>(i)[j];
				container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);
				containerCount++;
			}
		}
	}
	//TestFile.close();

	//imwrite("newAngle.bmp", newAngle);
	//ofstream NewAngleFile;
	//NewAngleFile.open("NewAngleFile.csv");
	//NewAngleFile << newAngle;
	//NewAngleFile.close();
	//////999999999999999999999999999999999999999999999999999999
	ofstream ContainerFile;
	ContainerFile.open("ContainerFile.txt");
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
		int bin;
		int angle;
		int size;
	};
	MaxElementStruct mes;
	for (size_t l = 0; l < container.size(); l++)
	{
		if (maxCountContainer.empty())
		{
			maxCountContainer.push_back(maxCountStruct());
			maxCountContainer[l].bin = container[l].bin;
			maxCountContainer[l].angle = container[l].angle;
			maxCountContainer[l].size += 1;
		}
		else
		{
			for (size_t m = 0; m < maxCountContainer.size(); m++)
			{
				if (maxCountContainer[m].bin == container[l].bin)
				{
					maxCountContainer[m].size += 1;
					break;
				}
				else if (m== maxCountContainer.size() - 1)
				{
					maxCountContainer.push_back(maxCountStruct());
					maxCountContainer[maxCountContainer.size()-1].bin = container[l].bin;
					maxCountContainer[maxCountContainer.size() - 1].angle = container[l].angle;
					maxCountContainer[maxCountContainer.size() - 1].size += 1;
					break;
				}

				if (maxCountContainer[m].size > temp)	///Find bin with the most elements
				{
					temp = maxCountContainer[m].size;
					mes.bin = (int)maxCountContainer[m].bin;	///Bin with most elements (bin ID)
					mes.angle = (int)maxCountContainer[m].angle;
					mes.size = (int)maxCountContainer[m].size;
					//me.value = vvvCounts[i].at(0);	///Frequency count
				}
			}
		}
	}
	cout << "The biggest number is: " << mes.size << " at bin " << mes.bin << endl;

	Mat tempGraySrc = GrayImg;
	for (size_t n = 0; n < container.size(); n++)
	{
		if (container[n].bin == mes.bin)
		{
			tempGraySrc.at<uchar>(container[n].i, container[n].j) = 255;
		}
	}
	imshow("tempGraySrc", tempGraySrc);
	imwrite("tempGraySrc.bmp", tempGraySrc);

	//ofstream maxCountContainerFile;
	//maxCountContainerFile.open("maxCountContainerFile.txt");
	//for (int i = 0; i < maxCountContainer.size(); i++)
	//{
	//	maxCountContainerFile << "maxCountContainer[" << i << "].bin= " << maxCountContainer[i].bin << "\n";
	//	maxCountContainerFile << "maxCountContainer[" << i << "].angle= " << maxCountContainer[i].angle << "\n";
	//	maxCountContainerFile << "maxCountContainer[" << i << "].size= " << maxCountContainer[i].size << "\n";
	//	maxCountContainerFile << "\n";
	//	maxCountContainerFile << "\n";
	//}
	////maxCountContainerFile << m << "," << maxCountContainer[m].bin << "," << maxCountContainer[m].angle << "," << maxCountContainer[m].size << "\n";
	//maxCountContainerFile.close();
	////Mat tempGray4Plot = GrayImg;
	////for (size_t k = 0; k < container.size(); k++)
	////{
	////	tempGray4Plot.at<uchar>(container[k].i, container[k].j) = 255;
	////}
	////imshow("tempGray4Plot", tempGray4Plot);
	////imwrite("tempGray4Plot.jpg", tempGray4Plot);

	waitKey(0);
	return 0;
}