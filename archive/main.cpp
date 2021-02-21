/*
 * main.cpp
 *
 *  Created on: Feb. 20, 2021
 *      Author: amaldev
 */

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

Mat remove_light(Mat&,Mat&,int);Mat img_thr;
Mat get_background(Mat&);
Mat thresh(Mat&,int);
void detect_obj(Mat& img,int method);
int main(int argc,char** argv){

	//keys recognized by command line parser
	const char* keys =
	{
			"{help h usage ? | | print this message}"
			"{@image || Image to process}"
			"{@light_pattern |0| Image light pattern to apply to image input}"
			"{@light_method | 0 | Method to remove background light, 0 difference, 1 div }"
			"{@thresh_method|0| method to threshold}"
			"{@seg_method | 1 | Method to segment: 1 connected Components, 2 connected components with stats, 3 find Contours }"
	};

	CommandLineParser parser(argc, argv, keys);
	parser.about("PhotoTool v1.0.0");
	//If requires help show
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	String img_file= parser.get<String>(0);
	String light_pattern_file= parser.get<String>(1);
	int method_light= parser.get<int>(2);
	int method_seg= parser.get<int>(4);
	int method_thresh=parser.get<int>(3);
	// Check if params are correctly parsed in variables

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	Mat image=imread(img_file);
	if(image.data ==NULL){

		std::cout<<"Error loading image"<<image<<std::endl;
		return 0;
	}

	Mat blurred_img;
	medianBlur(image,blurred_img,3);

	Mat background;
	background=get_background(blurred_img);

	Mat image_m;
	image_m= remove_light(blurred_img, background,method_light);
	Mat image_thresh;
	image_thresh=thresh(image_m,method_thresh);
	Mat thresh_bin[3];
	split(image_thresh,thresh_bin);

	detect_obj(thresh_bin[1],method_seg);
	return 0;
}
Mat remove_light(Mat& img, Mat& background, const int method)
{
	Mat aux;
	// if method is normalization
	if(method==1)
	{
		// Require change our image to 32 float for division
		Mat img32, background32;
		img.convertTo(img32, CV_32F);
		background.convertTo(background32, CV_32F);
		// Divide the image by the pattern
		aux= 1-(img32/background32);
		// Scale it to convert to 8bit format
		aux=aux*255;
		// Convert 8 bits format
		aux.convertTo(aux, CV_8U);
	}else{
		aux= background-img;
	}
	return aux;
}
Mat get_background(Mat& image){

	Mat background;
	blur(image,background,Size(image.cols/3,image.cols/3));
	return background;
}

Mat thresh(Mat& image,int method_thresh){

	Mat img_thr;
	if(method_thresh==0){
		threshold(image, img_thr, 30, 255, THRESH_BINARY);
	}else{
		threshold(image, img_thr, 140, 255, THRESH_BINARY_INV);
	}

	return img_thr;
}

void detect_obj(Mat& img,int method){

	if(method==0){
		Mat labeled_image,stats,centroids;
		int num_of_objects= connectedComponentsWithStats(img, labeled_image,stats,centroids);
		// pixels of background is set 0
		// for each of the detected objects a number representing that object is set as the pixel value for it
		// for the first object pixel value is 1, for second pixel value is 2;

		if(num_of_objects<2){

			std::cout<<"No objects detected"<<std::endl;
			return;
		}
		std::cout<<num_of_objects<<" objects detected"<<std::endl;

		RNG rng( 0xFFFFFFFF );
		Mat obj_img= Mat::zeros(img.rows,img.cols,CV_8UC3);// 3 channel image
		for( int i=1;i<num_of_objects;i++){

			Mat mask= labeled_image==i;
			Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
			obj_img.setTo(color,mask);
			std::stringstream ss;
			ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
			putText(obj_img,ss.str(),centroids.at<Point2d>(i),FONT_HERSHEY_SIMPLEX,0.4,Scalar(255,255,255));
		}

		namedWindow("img",WINDOW_NORMAL);
		imshow("img",obj_img);
		waitKey(0);
		return;
	}

	std::vector<std::vector<Point> > contours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat output= Mat::zeros(img.rows,img.cols, CV_8UC3);
	// Check the number of objects detected
	if(contours.size() == 0 ){
		std::cout << "No objects detected" << std::endl;
		return;
	}else{
		std::cout << "Number of objects detected: " << contours.size() << std::endl;
	}
	RNG rng( 0xFFFFFFFF );
	for(int i=0; i<contours.size(); i++){
		Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(output, contours, i, color);

	}

	imshow("Result", output);
	waitKey(0);


}
