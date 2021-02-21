/*
 * object_CV.hpp
 *
 *  Created on: Feb. 20, 2021
 *      Author: amaldev
 */

#ifndef OBJECT_CV_HPP_
#define OBJECT_CV_HPP_

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/ml.hpp>
#include <string>
#include <dirent.h>
#include <sys/types.h>

using namespace cv;

std::vector<std::string> get_sub_directories(char*);
Mat remove_light(Mat&,Mat&,int);
Mat get_background(Mat&);
Mat thresh(Mat&,int);
void detect_obj(Mat& img,int method);
std::vector<std::vector<float>> get_features(Mat& img );//,std::vector<int>* top=NULL,std::vector<int>* left =NULL
bool read_and_extract_features(std::string,int,int,std::vector<float>&,std::vector<int>&,std::vector<float>&,std::vector<float>& );
void train_and_test(const char*,int);
static const char* keys =
	{
			"{help h usage ? | | print this message}"
			"{@path|0|}"
			"{@image || Image to process}"
			"{@background|0| Image light pattern to apply to image input}"
			"{@light_method | 0 | Method to remove background light, 0 difference, 1 div }"
			"{@thresh_method|0| method to threshold}"
			"{@seg_method | 0 | Method to segment: 1 connected Components, 2 connected components with stats, 3 find Contours }"
	};


#endif /* OBJECT_CV_HPP_ */
