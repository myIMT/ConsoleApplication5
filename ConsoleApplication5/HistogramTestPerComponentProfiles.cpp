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
Mat src, srcImg, GrayImg, hist, cannyEdge, detected_edges, angle_src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y;

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

vector<Mat> Points;
vector<Mat> maskImages;
vector<Point> maskCentroid;
//----------------------------------------------------

/// <summary>                                                            
/// Calculates components                           
/// </summary>                                                           
/// <param name="GrayScaleSrcImg">Grayscale image (Mat) of original image.</param> 
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
		if (mask_i.empty())      // please, *always check* resource-loading.
		{
			cerr << "mask_i is empty - can't be loaded!" << endl;
			continue;
		}
		maskImages.push_back(mask_i);
		maskCentroid.push_back(Point(FltrCentroids.at<double>(FltrLabel, 0), FltrCentroids.at<double>(FltrLabel, 1)));
	}

	//
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

/// <summary>                                                            
/// Operations performed on each component.                             
/// </summary>                                                           
/// <param name="">xxx.</param> 
int main(int argc, char *argv[])
{
	//src = cv::imread("20161215 02.33_368L2.jpg");
	//src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg");
	src = cv::imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg");
	//imshow("src", src);
	//srcImg = src;
	//bilateralFilter(src, srcImg, 15, 80, 80);
	blur(src, srcImg, Size(5, 5), Point(-1, -1));
	//imshow("srcImg", srcImg);`
	cv::cvtColor(srcImg, GrayImg, cv::COLOR_BGR2GRAY);
	//imshow("GrayImg", GrayImg);

	Mat components = GetConnectedComponent(GrayImg);
	//imshow("components", components);

	//imshow("maskImages[0]", Mat(maskImages[0]));
	//Mat tempSrc1 = imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat tempSrc1 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg", CV_LOAD_IMAGE_UNCHANGED);
#pragma region MyRegion
	
	ofstream ComponentsLoop;
	ComponentsLoop.open("ComponentsLoop.txt");

	ofstream Bin_Analysis;
	Bin_Analysis.open("Bin_Analysis.csv");

	///Loop through each component
	for (size_t mi = 0; mi < 1/*maskImages.size()*/; mi++)
	{
		ComponentsLoop << "		Component Nr. = " << mi << "\n"; 
		//if (mi== 17 ||mi == 16 ||mi == 13 || mi == 9)
		//{
			std::string smi = std::to_string(mi);
			Mat tempComponent = Mat(maskImages[mi]);
			//imshow("tempComponent- " + smi, tempComponent);
			Mat GrayComponents;
			//cvtColor(tempComponent, GrayComponents, COLOR_BGR2GRAY);
			//imshow("GrayComponents", GrayComponents);
			GrayComponents = tempComponent;
			Sobel(GrayComponents, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
			Sobel(GrayComponents, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

			Mat Mag(GrayComponents.size(), CV_32FC1);
			Mat Angle(GrayComponents.size(), CV_32FC1);
			cartToPolar(grad_x, grad_y, Mag, Angle, true);

			//////77777777777777777777777777777777777777777777777777777777777777777777777777777777777
			ofstream Angle_DataFile;
			Angle_DataFile.open("Angle_DataFile.csv");
			Angle_DataFile << Angle << "\n";
			Angle_DataFile.close();
			//////77777777777777777777777777777777777777777777777777777777777777777777777777777777777

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
			//imshow("cannyEdge", cannyEdge);
			//imwrite("cannyEdge_DataFile.csv",cannyEdge);
			//////8888888888888888888888888888888888888888888888888888888888888888888888888888888
			ofstream cannyEdge_DataFile;
			cannyEdge_DataFile.open("cannyEdge_DataFile.csv");
			cannyEdge_DataFile << cannyEdge << "\n";
			cannyEdge_DataFile.close();
			//////8888888888888888888888888888888888888888888888888888888888888888888888888888888

			Mat newAngle = Mat(Angle.size().height, Angle.size().width, Angle.type(), Scalar(0, 0, 0));

			///Walk along cannyEdge rows
			for (size_t i = 0; i < cannyEdge.rows; i++)
			{
				///Walk along cannyEdge rows & columns
				for (size_t j = 0; j < cannyEdge.cols; j++)
				{
					ComponentsLoop << "			Walk along Canny Edge (coordinates - (x,y) ) = " << i <<", " << j << "\n";
					Bin_Analysis << "Canny Edge (coordinates - (x,y) ) = " << i << ", " << j << "\n";
					///if cannyEdge pixel intensity id non-zero
					if ((int)cannyEdge.at<uchar>(i, j) != 0)
					{
						ComponentsLoop << "				Non-Zero Canny pixel value (coordinates - (x,y) ) = " << i << ", " << j << "\n";
						ComponentsLoop << "				Storing Non-Zero Canny pixel value in container" << "\n";

						Bin_Analysis << "Canny Edge Non-zero pixel (coordinates - (x,y) ) = " << i << ", " << j << "\n";

						newAngle.ptr<float>(i)[j] = Angle.ptr<float>(i)[j];							///Create new Angle matrix
						container.push_back(element());												///Initialise container
						container[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);	///Store bin (newAngle/5)
						container[containerCount].i = i;											///Store row position
						container[containerCount].j = j;											///Store column position
						container[containerCount].angle = newAngle.ptr<float>(i)[j];				///Store new Angle value
						container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);			///Store canny pixel intensity
						containerCount++;
					}
				}
			}
			ComponentsLoop << "Finished walking on Canny Edge" << "\n";
			ComponentsLoop << "\n";
			ComponentsLoop << "\n";

			Bin_Analysis.close();
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
				int bin = 0;
				int angle = 0;
				int size = 0;
			};
			MaxElementStruct mes;

			ofstream KeepingTrackOfContainers_DataFile;
			KeepingTrackOfContainers_DataFile.open("KeepingTrackOfContainers_DataFile.csv");

			ComponentsLoop << "			For every element in container (total size = " << container.size() << ")" << "\n";
			/// Grouping bin values together and counting them
			/// For every element in container
			for (size_t l = 0; l < container.size(); l++)
			{
				ComponentsLoop << "			container[l] = " << l << "\n";
				/// If new container is empty (at start)
				if (maxCountContainer.empty())
				{
					ComponentsLoop << "				Initial element (new container empty) in new container[l] = " << l << "\n";
					maxCountContainer.push_back(maxCountStruct());		/// Initialise new container
					maxCountContainer[l].bin = container[l].bin;		/// Store container bin value in new container bin field
					maxCountContainer[l].angle = container[l].angle;	/// Store container angle value in new container angle field
					maxCountContainer[l].size += 1;						/// Increment new container size field (counting elements with similair bin values)
				}
				else  /// If not at start (new container contains first element)
				{
					/// For every element in new container (new container contains at least one element thus far) & its elements will increase with every loop
					for (size_t m = 0; m < maxCountContainer.size(); m++)
					{
						ComponentsLoop << "				For every element in new container[m] (as it's filling up ) = " << m << "\n";

						KeepingTrackOfContainers_DataFile << "container iterator (l): " << l << "\n";
						KeepingTrackOfContainers_DataFile << "maxCountContainer iterator (m)= " << m << "\n";
						/// 
						if (maxCountContainer[m].bin == container[l].bin)
						{
							ComponentsLoop << "					When container element [l] already exist in new container [m] (don't re-add it, just update it's count ) = " << l << ", " << m << "\n";
							maxCountContainer[m].size += 1;
							break;
						}
						else if (m == maxCountContainer.size() - 1)
						{
							maxCountContainer.push_back(maxCountStruct());
							maxCountContainer[maxCountContainer.size() - 1].bin = container[l].bin;
							maxCountContainer[maxCountContainer.size() - 1].angle = container[l].angle;
							maxCountContainer[maxCountContainer.size() - 1].size += 1;
							break;
						}

						//////666666666666666666666666666666666666666666666666666666666666666666666666666666
						ofstream maxCountContainer_DataFile;
						maxCountContainer_DataFile.open("maxCountContainer_DataFile.csv");
						for (int i = 0; i < maxCountContainer.size(); i++)
						{
							maxCountContainer_DataFile << "maxCountContainer[" << i << "].i= " << i << "\n";
							maxCountContainer_DataFile << "maxCountContainer[" << i << "].bin= " << maxCountContainer[i].bin << "\n";
							maxCountContainer_DataFile << "maxCountContainer[" << i << "].angle= " << maxCountContainer[i].angle << "\n";
							maxCountContainer_DataFile << "maxCountContainer[" << i << "].size= " << maxCountContainer[i].size << "\n";
							maxCountContainer_DataFile << "\n";
							maxCountContainer_DataFile << "\n";
						}
						maxCountContainer_DataFile.close();
						maxCountContainer_DataFile.close();
						//////666666666666666666666666666666666666666666666666666666666666666666666666666666

						ComponentsLoop << "		Adjusting element with the highest frequency" << "\n";
						if (maxCountContainer[m].size > temp)	///Find bin with the most elements
						{
							temp = maxCountContainer[m].size;
							mes.bin = (int)maxCountContainer[m].bin;	///Bin with most elements (bin ID)
							mes.angle = (int)maxCountContainer[m].angle;
							mes.size = (int)maxCountContainer[m].size;
						}
						ComponentsLoop << "		Element with highest frequency ("<< mes.size << "= " << mes.angle << "\n";
					}
				}
			}
			KeepingTrackOfContainers_DataFile.close();

			cout << "The biggest number is: " << mes.size << " at bin " << mes.bin << endl;
			cout << "Angle (mes)- " << smi << "= " << mes.angle << "\n";
			Mat tempGraySrc = GrayImg;
			for (size_t n = 0; n < container.size(); n++)
			{
				if (container[n].bin == mes.bin)
				{
					tempGraySrc.at<uchar>(container[n].i, container[n].j) = 255;
				}
			}
			//imshow("tempGraySrc", tempGraySrc);
			imwrite("tempGraySrc.bmp", tempGraySrc);

			//Mat tempSrc2 = imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
			//Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
			Mat tempSrc2 = imread("20140612_MINEGARDEN_SURVEY_CylindricalMine01R2.jpg", CV_LOAD_IMAGE_UNCHANGED);
#pragma region Bounding Box
			vector<double> lengths(4);
			double rectSize_b;
			size_t imgCount = 0;
			//cout << "maskImages.size()= " << maskImages.size() << "\n";
			//for (imgCount; imgCount < maskImages.size(); imgCount++)
			//{
			Mat tempPoints;
			findNonZero(maskImages[mi], tempPoints);
			Points.push_back(tempPoints);
			//}
			Point2f vtx[4];
			RotatedRect box = minAreaRect(Points[mi]); //only the first Mat Points
			box.points(vtx);
			/*Mat tempSrc1 = imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);*/
			for (int i = 0; i < 4; i++)
			{
				line(tempSrc1, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
				line(tempSrc2, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
				lengths.push_back(norm((vtx[(i + 1) % 4]) - (vtx[i])));
			}
			/*imshow("Bounding Box", tempSrc1);*/
			//cout << "minAreaRect Angle - "<<smi<<"= " << box.angle + 180 << "\n";
			cout << "minAreaRect width= " << box.size.width << "\n";
			cout << "minAreaRect height= " << box.size.height << "\n";
#pragma endregion
			Mat plotImage = src;
			circle(plotImage, maskCentroid[mi], 1, Scalar(0, 255, 0), 1, 8, 0);
			circle(tempSrc2, maskCentroid[mi], 1, Scalar(0, 255, 0), 1, 8, 0);

#pragma region walk in edge angle direction
			Point2f u, u2, u22, v;
			Point2f w1, w2;
			cout << "cos((mes.angle)* CV_PI / 180.0)= " << cos((mes.angle)* CV_PI / 180.0) << "\n";
			cout << "sin((mes.angle)* CV_PI / 180.0)= " << sin((mes.angle)* CV_PI / 180.0) << "\n";
			u = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
			u2 = u;
			rectSize_b = *max_element(lengths.begin(), lengths.end());
			double d = 0.1*rectSize_b;
			double normU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
			//cout << "normU= " << normU << "\n";
			v = Point2f(u.x / normU, u.y / normU);
			Mat tempSrcW1 = src, tempSrcW2 = src;
			for (size_t i = 0; i < 10; i++)
			{
				if (i == 0)
				{	// starting point = center of mask
					w1.x = maskCentroid[mi].x + v.x*d;	//one side
					w1.y = maskCentroid[mi].y + v.y*d;

					w2.x = maskCentroid[mi].x - v.x*d;	//other side
					w2.y = maskCentroid[mi].y - v.y*d;
				}
				else
				{	// points on either-side of mask center point
					w1.x = u2.x + v.x*d;		//one side
					w1.y = u2.y + v.y*d;

					w2.x = u22.x - v.x*d;		//other side
					w2.y = u22.y - v.y*d;
				}
				//cout << "i - " << i << "2-Plot here= " << w1 << ", " << w2 << "\n";
				//circle(plotImage, w1, 1, Scalar(0, 0, 255), 1, 8, 0);
				//circle(plotImage, w2, 1, Scalar(255, 0, 0), 1, 8, 0);

				circle(tempSrc2, w1, 1, Scalar(55, 55, 55), 1, 8, 0);
				circle(tempSrc2, w2, 1, Scalar(55, 55, 55), 1, 8, 0);
				//circle(tempSrcW1, w1, 1, Scalar(0, 0, 0), 1, 8, 0);
				//circle(tempSrcW2, w2, 1, Scalar(255, 255, 255), 1, 8, 0);
				u2 = w1;
				u22 = w2;
			}
			/*		imshow("tempSrcW1- " + smi, tempSrcW1);
					imshow("tempSrcW2- " + smi, tempSrcW2)*/;
#pragma endregion

			struct buffer {
				std::vector<double> pixValues;
				Point2f startPoint;
				Point2f endPoint;
				//int j;
				//int angle;
				//int value;
			};
			vector<buffer> Profiles;
			int ProfilesCount = 0;

#pragma region walk perpendicular in edge angle direction
			Point2f uu, uu2, uu22, vv, ep11, ep12, ep21, ep22;
			Point2f ww1, ww2;
			uu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
			uu2 = uu;
			rectSize_b = *max_element(lengths.begin(), lengths.end());
			//double dd = 0.1*rectSize_b;
			double normUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
			vv = Point2f(uu.x / normUU, uu.y / normUU);
			u = vv;
			//rotate and swap
			double tempXX = vv.x;
			vv.x = -vv.y;
			vv.y = tempXX;
			int e = 5;

			Mat tempGraySrc3;
			cv::cvtColor(src, tempGraySrc3, cv::COLOR_BGR2GRAY);
			Mat tempSrc3 = tempGraySrc;//imread("20161215 02.33_368L2.jpg", CV_LOAD_IMAGE_UNCHANGED);
			for (size_t i = 0; i < 10; i++)
			{
				if (i == 0)
				{	// starting point = center of mask
					ww1.x = maskCentroid[mi].x + vv.x*d;	//one side
					ww1.y = maskCentroid[mi].y + vv.y*d;

					ww2.x = maskCentroid[mi].x - vv.x*d;	//other side
					ww2.y = maskCentroid[mi].y - vv.y*d;

					//end points of profile
					ep11.x = ww1.x - ((box.size.width+e) / 2) * uu.x;
					ep11.y = ww1.y - ((box.size.width+e) / 2) * uu.y;
					ep12.x = ww1.x + ((box.size.width+e) / 2) * uu.x;
					ep12.y = ww1.y + ((box.size.width+e) / 2) * uu.y;

					ep21.x = ww2.x - ((box.size.width+e) / 2) * uu.x;
					ep21.y = ww2.y - ((box.size.width+e) / 2) * uu.y;
					ep22.x = ww2.x + ((box.size.width+e) / 2) * uu.x;
					ep22.y = ww2.y + ((box.size.width+e) / 2) * uu.y;
				}
				else
				{	// points on either-side of mask center point
					ww1.x = uu2.x + vv.x*d;		//one side
					ww1.y = uu2.y + vv.y*d;

					ww2.x = uu22.x - vv.x*d;	//other side
					ww2.y = uu22.y - vv.y*d;

					//end points of profile
					ep11.x = ww1.x - ((box.size.width+e) / 2) * uu.x;
					ep11.y = ww1.y - ((box.size.width+e) / 2) * uu.y;
					ep12.x = ww1.x + ((box.size.width+e) / 2) * uu.x;
					ep12.y = ww1.y + ((box.size.width+e) / 2) * uu.y;

					ep21.x = ww2.x - ((box.size.width+e) / 2) * uu.x;
					ep21.y = ww2.y - ((box.size.width+e) / 2) * uu.y;
					ep22.x = ww2.x + ((box.size.width+e) / 2) * uu.x;
					ep22.y = ww2.y + ((box.size.width+e) / 2) * uu.y;
				}
				circle(tempSrc2, ww2, 1, Scalar(255, 0, 0), 1, 8, 0); //turqoise
				circle(tempSrc2, ww1, 1, Scalar(55, 0, 0), 1, 8, 0); //
																		 //circle(tempSrc2, ww2, 1, Scalar(255, 255, 10), 1, 8, 0); //
				circle(tempSrc2, ep11, 1, Scalar(0, 0, 255), 1, 8, 0); //
				circle(tempSrc2, ep12, 1, Scalar(0, 0, 255), 1, 8, 0); //
				circle(tempSrc2, ep21, 1, Scalar(0, 0, 255), 1, 8, 0); //
				circle(tempSrc2, ep22, 1, Scalar(0, 0, 255), 1, 8, 0); //

				uu2 = ww1;
				uu22 = ww2;

				#pragma region DrawLines
					int thickness = 0.2;
					int lineType = 8;
					line(tempGraySrc3,
						Point(ep11.x, ep11.y),
						Point(ep12.x, ep12.y),
						Scalar(255, 0, 0),
						thickness,
						lineType);  
				#pragma endregion


				#pragma region LinePixels
				
				// grabs pixels along the line (pt1, pt2)
				// from 8-bit 3-channel image to the buffer
				LineIterator it1(tempGraySrc3, Point(ep11), Point(ep12), 8);
				//LineIterator it2(tempSrc3, Point(ep21), Point(ep22), 8);
				LineIterator it11 = it1;
				//LineIterator it22 = it2;
				//vector<Vec3b> buf(it.count);

				ofstream file;
				
				for (int l = 0; l < it1.count; l++, ++it1)
				{
					Profiles.push_back(buffer());
					Profiles[ProfilesCount].startPoint = ep11;
					Profiles[ProfilesCount].endPoint = ep12;
					double val = (double)tempGraySrc3.at<uchar>(it1.pos());
					Profiles[ProfilesCount].pixValues.push_back(val);// (double)tempSrc3.at<uchar>(it1.pos());

					//double val = (double)src_gray.at<uchar>(it.pos());
					//buf[i] = val;

					std::string L = std::to_string(l);
					file.open("buf_"+ format("(%d,%d)", Profiles[ProfilesCount].startPoint, Profiles[ProfilesCount].endPoint)+".csv", ios::app);
					//file.open("buf_" + L + ".csv", ios::app);
					//file << Profiles[ProfilesCount].startPoint << "\n";
					//file << Profiles[ProfilesCount].endPoint << "\n";
					file << Mat(Profiles[ProfilesCount].pixValues) << "\n";
					file.close();
					ProfilesCount += 1;
				}
				#pragma endregion

				
				//// alternative way of iterating through the line
				//for (int i = 0; i < it2.count; i++, ++it2)
				//{
				//	Vec3b val = img.at<Vec3b>(it2.pos());
				//	CV_Assert(buf[i] == val);
				//}
				
				//container[containerCount].bin = int(newAngle.ptr<float>(i)[j] / binSize);
				//container[containerCount].i = i;
				//container[containerCount].j = j;
				//container[containerCount].angle = newAngle.ptr<float>(i)[j];
				//container[containerCount].value = (int)cannyEdge.at<uchar>(i, j);
				//containerCount++;
			}
#pragma endregion  

#pragma region EPs
			//Point2f uuu, uuu2, uuu22, vvvv, ep1,ep2;
			//Point2f www1, www2;
			//uuu = Point2f(cos((mes.angle)* CV_PI / 180.0), sin((mes.angle)* CV_PI / 180.0));
			//uuu2 = uuu;
			//rectSize_b = *max_element(lengths.begin(), lengths.end());
			////double dd = 0.1*rectSize_b;
			//double normUUU = sqrt(cos((mes.angle)* CV_PI / 180.0)*cos((mes.angle)* CV_PI / 180.0) + sin((mes.angle)* CV_PI / 180.0)*sin((mes.angle)* CV_PI / 180.0));
			//vvvv = Point2f(uuu.x / normUUU, uuu.y / normUUU);
			////rotate and swap
			//Point2d vvvvv;
			//double tempXXX = vvvv.x;
			//vvvvv.x = -vvvv.y;
			//vvvvv.y = tempXXX;
			//for (size_t i = 0; i < 10; i++)
			//{
			//	if (i == 0)
			//	{
			//		www1.x = maskCentroid[mi].x + vvvv.x*d;
			//		www1.y = maskCentroid[mi].y + vvvv.y*d;
			//		//ep1.x = www1.x-(box.size.width/2)*(vvvv.x*d);
			//		//ep1.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
			//		//ep2.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
			//		//ep2.y = www1.y - (box.size.width / 2)*(vvvv.y*d);

			//		www2.x = maskCentroid[mi].x - vvvvv.x*d;
			//		www2.y = maskCentroid[mi].y - vvvvv.y*d;
			//		ep1.x = www2.x - (box.size.width / 2)*(vvvv.x);
			//		ep1.y = www2.y - (box.size.width / 2)*(vvvv.y);
			//		ep2.x = www2.x + (box.size.width / 2)*(vvvv.x);
			//		ep2.y = www2.y + (box.size.width / 2)*(vvvv.y);
			//	}
			//	else
			//	{
			//		www1.x = uuu2.x + vvvv.x*d;
			//		www1.y = uuu2.y + vvvv.y*d;
			//		//ep1.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
			//		//ep1.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
			//		//ep2.x = www1.x - (box.size.width / 2)*(vvvv.x*d);
			//		//ep2.y = www1.y - (box.size.width / 2)*(vvvv.y*d);
			//		//cout << "minAreaRect Angle= " << box.angle + 180 << "\n";
			//		www2.x = uuu22.x - vvvv.x*d;
			//		www2.y = uuu22.y - vvvv.y*d;
			//		ep1.x = www2.x - (box.size.width / 2)*(vvvv.x);
			//		ep1.y = www2.y - (box.size.width / 2)*(vvvv.y);
			//		ep2.x = www2.x + (box.size.width / 2)*(vvvv.x);
			//		ep2.y = www2.y + (box.size.width / 2)*(vvvv.y);
			//		//cout << "ww2= " << ww2 << "\n";
			//	}
			//	//cout << "i - " << i << "1-Plot here= " << ww1 << ", " << ww2 << "\n";
			//	//circle(src, ww1, 1, Scalar(0, 255, 255), 1, 8, 0); //yellow
			//	//circle(plotImage, ww2, 1, Scalar(255, 255, 0), 1, 8, 0); //turqoise
			//	//circle(tempSrc2, www1, 1, Scalar(255, 255, 10), 1, 8, 0); //
			//	circle(tempSrc2, www2, 1, Scalar(255, 255, 100), 1, 8, 0); //
			//	circle(tempSrc2, ep1, 1, Scalar(255, 255, 200), 1, 8, 0);
			//	circle(tempSrc2, ep2, 1, Scalar(255, 255, 10), 1, 8, 0);
			//	uuu2 = www1;
			//	uuu22 = www2;
			//}
#pragma endregion

#pragma endregion

		//imshow("Plot Image", plotImage);
		imshow("Source - component nr." + smi, tempSrc2); 
		imshow("Grayscale- component nr." + smi, tempGraySrc3);
		//}
	}
	ComponentsLoop.close();
	//imshow("Plot Image", src);
	//imshow("Bounding Box", tempSrc1);
	waitKey(0);
	return 0;
}