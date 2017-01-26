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
Mat srcImg, GrayImg, hist, cannyEdge, detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;

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

	std::vector<cv::Vec3b> FltrColors(nFltrLabels);
	FltrColors[0] = cv::Vec3b(0, 0, 0);

	for (int FltrLabel = 1; FltrLabel < 2/*nFltrLabels*/; ++FltrLabel) {
		FltrColors[FltrLabel] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));

		Mat mask_i = FltrLabelImage == FltrLabel;

		return mask_i;
	}

	cv::Mat FltrDst(GrayScaleSrcImg.size(), CV_8UC3);
	for (int r = 0; r < FltrDst.rows; ++r) {
		for (int c = 0; c < FltrDst.cols; ++c) {
			int FltrLabel = FltrLabelImage.at<int>(r, c);
			cv::Vec3b &FltrPixel = FltrDst.at<cv::Vec3b>(r, c);
			FltrPixel = FltrColors[FltrLabel];
		}
	}
	imshow(nFltrLabelsString + "-Connected Components", FltrDst);
	imwrite("Connected Components.bmp", FltrDst);
}

int main(int argc, char *argv[])
{
	srcImg = cv::imread("20161215 02.33_368L.jpg");

	cv::cvtColor(srcImg, GrayImg, cv::COLOR_BGR2GRAY);

	Mat component = GetConnectedComponent(GrayImg);
	imshow("component", component);
	ofstream ComponentFile;
	ComponentFile.open("ComponentFile.csv");
	ComponentFile << component;
	ComponentFile.close();

	Sobel(component, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(component, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	Mat Mag(component.size(), CV_32FC1);
	Mat Angle(component.size(), CV_32FC1);
	cartToPolar(grad_x, grad_y, Mag, Angle, true);
	ofstream AngleFile;
	AngleFile.open("AngleFile.csv");
	AngleFile << Angle;
	AngleFile.close();

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

	Canny(component, cannyEdge, 100, 200);
	ofstream CannyFile;
	CannyFile.open("CannyFile.csv");
	CannyFile << cannyEdge;
	CannyFile.close();

	Mat newAngle = Mat(cannyEdge.size().width, cannyEdge.size().height, CV_64F, 0.0);;
	
	for (size_t i = 0; i < cannyEdge.rows; i++)
	{
		const float* aRow_i = Angle.ptr<float>(i);

		for (size_t j = 0; j < cannyEdge.cols; j++)
		{
			if ((int)cannyEdge.at<uchar>(i, j) != 0)
			{
				newAngle.at<double>(i, j) = (double)aRow_i[j];

				container.push_back(element());
				container[containerCount].bin = int(newAngle.at<double>(i, j) / binSize);
				container[containerCount].i = i;
				container[containerCount].j = j;
				container[containerCount].angle = newAngle.at<double>(i, j);
				container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);
				containerCount++;
			}
		}
	}
	ofstream NewAngleFile;
	NewAngleFile.open("NewAngleFile.csv");
	NewAngleFile << newAngle;
	NewAngleFile.close();
	//999999999999999999999999999999999999999999999999999999
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
	//999999999999999999999999999999999999999999999999999999
	Mat tempGray4Plot = GrayImg;
	for (size_t k = 0; k < container.size(); k++)
	{
		tempGray4Plot.at<uchar>(container[k].i, container[k].j) = 255;
	}
	imshow("tempGray4Plot", tempGray4Plot);
	imwrite("tempGray4Plot.jpg", tempGray4Plot);

	waitKey(0);
	return 0;
}