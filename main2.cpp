/*
 * main2.cpp
 *
 *  Created on: Feb. 20, 2021
 *      Author: amaldev
 */

#include "object_CV.hpp"

int main(int argc,char** argv){


	CommandLineParser parser(argc, argv, keys);
	parser.about("PhotoSegmentor v1.0.0");
	//If requires help show
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	// Check if params are correctly parsed in variables

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	std::string path0= parser.get<std::string>(0);
	const char* path= path0.c_str();
	train_and_test(path,5);
	return 0;
}


