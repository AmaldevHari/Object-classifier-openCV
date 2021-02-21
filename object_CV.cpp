/*
 * object_CV.cpp
 *
 *  Created on: Feb. 20, 2021
 *      Author: amaldev
 */
#include "object_CV.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> get_sub_directories(const char* path){

	std::vector<std::string> sub_directories;
	struct dirent *entry;
	DIR *dir = opendir(path);

	if (dir == NULL) {
		return sub_directories;
	}
	while ((entry = readdir(dir)) != NULL) {


			sub_directories.push_back( entry->d_name);
	}
	closedir(dir);

	for(int i=0;i<sub_directories.size();i++){
		if(sub_directories[i]==".."){
			sub_directories.erase(sub_directories.begin()+i);
		}
		if(sub_directories[i]=="."){
					sub_directories.erase(sub_directories.begin()+i);
				}
	}


	return sub_directories;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat get_background(Mat& image){

	Mat background;
	blur(image,background,Size(image.cols/3,image.cols/3));
	return background;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat thresh(Mat& image,int method_thresh){

	Mat img_thr;
	if(method_thresh==0){
		threshold(image, img_thr, 30, 255, THRESH_BINARY);
	}else{
		threshold(image, img_thr, 140, 255, THRESH_BINARY_INV);
	}

	return img_thr;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<float>> get_features(Mat& img ){//,std::vector<int>* top=NULL,std::vector<int>* left =NULL

	std::vector<std::vector<float>> features;
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	Mat img_c= img.clone();
	findContours(img_c,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);

	if(contours.size()==0){
		return features;
	}
	RNG rng(0xFFFFFFFF);

	for(int i=0;i<contours.size();i++){

		Mat mask=Mat::zeros(img.rows,img.cols,CV_8UC1);
		drawContours(mask,contours,i,Scalar(1),FILLED,LINE_8,hierarchy,1);// last argument specifies to draw the nested contours,
		//if itis 2 then it draws the nested-nested contours
		Scalar area= sum(mask);
		float area_c= area[0];
		if(area_c>500){ // 500 is the miniimum area required to be occupied by objects to be identified

			RotatedRect r=minAreaRect(contours[i]);
			float width= r.size.width;
			float height=r.size.height;
			float aspect_ratio= width<height? height/width:width/height;

			std::vector<float> feature_t;
			feature_t.push_back(area_c);
			feature_t.push_back(aspect_ratio);
			features.push_back(feature_t);

			//			if(!(left==NULL)){
			//				left->push_back(r.center.x);
			//			}if(!(top==NULL)){
			//
			//				top->push_back((r.center.y));
			//			}

		}
	}


	return features;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool read_and_extract_features(std::string path,int label,int num_of_test,std::vector<float>& training_data,std::vector<int>& response_data,std::vector<float>& test_data,std::vector<float>& test_response_data){

//	VideoCapture images(path);
//	if(!images.isOpened()){
//
//		std::cout<<"Error: Cannot open folder"<<std::endl;
//		return false;
//
//	}
	std::vector<std::string>  file_names= ::get_sub_directories(path.c_str());
	Mat frame;
	int img_index=0;
	for(int i=0;i<file_names.size();i++){

		std::stringstream ss;
		ss<<path<<"/"<<file_names[i];
		std::cout<<ss.str()<<std::endl;
		frame= imread(ss.str());
//		imshow("img",frame);
//		waitKey(0);
		Mat background;
		Mat blurred_img;
		medianBlur(frame,blurred_img,3);//smooth
		background=get_background(blurred_img);
		Mat image_m;
		image_m= remove_light(blurred_img, background,1);
		Mat image_thresh;
		image_thresh=thresh(image_m,0);
		Mat thresh_bin[3];
		split(image_thresh,thresh_bin);
		std::vector<std::vector<float>> features= get_features(thresh_bin[0]);

		for(int i=0;i<features.size();i++){

			if(img_index >= num_of_test){
				training_data.push_back(features[i][0]);
				training_data.push_back(features[i][1]);
				response_data.push_back(label);
			}else{

				test_data.push_back(features[i][0]);
				test_data.push_back(features[i][1]);
				test_response_data.push_back(label);

			}
		}
		img_index++;

	}
	return true;


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void train_and_test( const char* path,int num_of_test){

	//path is absolute path to the parent of subdirs containing training and test data
	std::vector<float> training_data;
	std::vector<int> response_data;
	std::vector<float> test_data;
	std::vector<float> test_response_data;

	std::vector<std::string> sub_dirs= get_sub_directories(path);

	for(int i=0;i<sub_dirs.size();i++){

		std::stringstream ss;
		ss<<path<<"/"<<sub_dirs[i];
		ss.str();
		read_and_extract_features(ss.str(),i,num_of_test,training_data,response_data,test_data,test_response_data);
	}
	// merge all data into matrices
	Mat training_data_mat(training_data.size()/2, 2, CV_32FC1,&training_data[0]);
	Mat response_data_mat(response_data.size(), 1, CV_32SC1,&response_data[0]);
	Mat test_data_mat(test_data.size()/2, 2, CV_32FC1, &test_data[0]);
	Mat test_response_mat(test_response_data.size(), 1, CV_32FC1,&test_response_data[0]);

	Ptr<::ml::SVM> svm;
	svm=ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::CHI2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,100, 1e-6));
	svm->train(training_data_mat,ml::ROW_SAMPLE,response_data_mat);

	if(test_response_data.size()>0){
		std::cout << "Evaluation" << std::endl;
		std::cout << "==========" << std::endl;
		// Test the ML Model
		Mat test_predict;
		svm->predict(test_data_mat, test_predict);
		std::cout << "Prediction Done" << std::endl;
		// Error calculation
		Mat error_mat= test_predict!=test_response_mat;
		float error= 100.0f * countNonZero(error_mat) / test_response_data.size();
		std::cout << "Error: " << error << "\%" << std::endl;
		//	// Plot training data with error label
		//	plotTrainData(trainingDataMat, responses, &error);
		//	}else{
		//	plotTrainData(trainingDataMat, responses);
		//	}
	}
}
