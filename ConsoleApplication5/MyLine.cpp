//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "iostream"
//#include <opencv2/opencv.hpp>
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include <iostream>
//#include <fstream>
////#include "myedgedetector.h"
//#include <vector>
//#include <stdlib.h>
//#include <array>
//
////#include <opencv2/legacy/compat.hpp>
//
//using namespace cv;
//using namespace std;
//using std::vector;
//
////![variables]
//mat src, src_gray;
//mat dst, detected_edges, anglehist, orighist,grayhist, pixelsbin;
//int edgethresh = 1;
//int lowthreshold;
//int const max_lowthreshold = 100;
//int ratio = 3;
//int kernel_size = 3;
//const char* window_name = "edge map";
//int ddepth = cv_32fc1;// cv_16s;
//int scale = 1;
//int delta = 0;
///// generate grad_x and grad_y
//mat grad_x, grad_y;
//mat abs_grad_x, abs_grad_y;
//mat grad;
////------------------angle histogram parameters--------
//int binsize = 5;
//int anglelimit = 360;
///// establish the number of bins
//int histsize = anglelimit / binsize;
////int histsize = 72;
///// set the ranges ( for b,g,r) )
//float rangea[] = { 0, 360 };
//const float* histrange = { rangea };
//mat angle_hist;
//bool uniform = true;
//bool myaccumulate = false;
//int channels[1];
//int binid;
//
//ofstream pixelsinbinfile;
//std::array<std::vector<int>, 2> vvv{ {} };
////string filename = "pixelsinbinfile";
////pixelsinbinfile.open("pixelsinbinfile.txt");
//int mousex, mousey;
//point pt1, pt2;
//int clickcounter = 0;
//
////lineiterator it;
////lineiterator it2;
////vector<vec3b> buf;
////![variables]


//void myline(mat img, point start, point end)
//{
//	
//	if (clickcounter==2)
//	{
//		clickcounter = 0;
//	}
//
//	cout << "drawing line ([" << pt1.x << ", " << pt1.y << "], [" << pt2.x << ", " << pt2.y << "])" << endl;
//
//	int thickness = 2;
//	int linetype = 8;
//	line(img,
//		point(pt1.x,pt1.y),
//		point(pt2.x, pt2.y),
//		scalar(255, 0, 0),
//		thickness,
//		linetype);
//	imshow("grayscale image", img);
//}