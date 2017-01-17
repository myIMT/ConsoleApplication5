#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
//#include "myedgedetector.h"
#include <vector>
#include <stdlib.h>
#include <array>

//#include <opencv2/legacy/compat.hpp>

using namespace cv;
using namespace std;
using std::vector;

//![variables]
mat src, src_gray;
mat dst, detected_edges, anglehist, orighist,grayhist, pixelsbin;
int edgethresh = 1;
int lowthreshold;
int const max_lowthreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "edge map";
int ddepth = cv_32fc1;// cv_16s;
int scale = 1;
int delta = 0;
/// generate grad_x and grad_y
mat grad_x, grad_y;
mat abs_grad_x, abs_grad_y;
mat grad;
//------------------angle histogram parameters--------
int binsize = 5;
int anglelimit = 360;
/// establish the number of bins
int histsize = anglelimit / binsize;
//int histsize = 72;
/// set the ranges ( for b,g,r) )
float rangea[] = { 0, 360 };
const float* histrange = { rangea };
mat angle_hist;
bool uniform = true;
bool myaccumulate = false;
int channels[1];
int binid;

ofstream pixelsinbinfile;
std::array<std::vector<int>, 2> vvv{ {} };
//string filename = "pixelsinbinfile";
//pixelsinbinfile.open("pixelsinbinfile.txt");
int mousex, mousey;
point pt1, pt2;
int clickcounter = 0;

lineiterator it;
lineiterator it2;
vector<vec3b> buf;
![variables]


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		system("cls");
		clickCounter++;
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		if (clickCounter==1)
		{
			cout << "clickCounter= " << clickCounter << "\n";
			pt1.x = x;
			pt1.y = y;
			cout << "Left button of the mouse is clicked - position (" << pt1.x << ", " << pt1.y << ")" << endl;
		}
		else if (clickCounter==2)
		{
			cout << "clickCounter= " << clickCounter << "\n";
			pt2.x = x;
			pt2.y = y;
			cout << "Left button of the mouse is clicked - position (" << pt2.x << ", " << pt2.y << ")" << endl;

			MyLine(src_gray, pt1, pt2);

			// grabs pixels along the line (pt1, pt2)
			// from 8-bit 3-channel image to the buffer
			LineIterator it(src_gray, pt1, pt2, 8);
			LineIterator it2 = it;
			vector<double> buf(it.count);

			//for (int i = 0; i < it.count; i++, ++it)
			//{
			//	buf[i] = (const Vec3b)*it;
			//	cout << "buf[i] = " << buf[i] << "\n";
			//}
				

			// alternative way of iterating through the line
			for (int i = 0; i < it2.count; i++, ++it2)
			{
				cout << "it2.pos()= " << it2.pos() << "\n";
				double val = (double)src.at<uchar>(it2.pos());
				//Vec3b val = src.at<Vec3b>(it2.pos());
				cout << "val = " << val << "\n";
				//cout << "gray image= "<< (double)src_gray.at<uchar>(it2.pos()
				//buf[i] = val;
				buf.push_back(val);
				//cout << "buf[i] = " << buf[i] << "\n";
			}
			//cout << "buf= " << Mat(buf) << "\n";
			//cerr << Mat(buf) << endl;
		}
		else
		{
			pt1.x = 0;
			pt1.y = 0;
			pt2.x = 0;
			pt2.y = 0;
		}
		
		

	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		//cout << "position (" << x << ", " << y << ")" << "\n"<<endl;
		//cout << "intensity= " << (double)src.at<uchar>(Point(x, y)) << "\n";

	}
}