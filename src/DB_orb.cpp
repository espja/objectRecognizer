
#include <iostream>
#include <vector>
#include <string>


#if cimg_OS==2 //Windows
#include "getopt.h"
#else
#include <unistd.h>
#include <stdlib.h>
#endif
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <memory.h>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;



void help()
{
	std::cout << "surf_opencv_two [options] imagefile\n";
	std::cout << "-u 0/1: run in rotation invariant mode. Default = " << true <<"\n";;
}


int main(int argc, char** argv)
{
	int opt;
	cv::namedWindow("points");

	std::string  outputfilename;
	int maxkeypoints = 500;
	while((opt=getopt(argc,argv,"hO:k:o"))!=-1)
	{

		switch(opt)
		{
		case 'h':
			help();
			exit(0);
			break;
		case 'o':
			outputfilename = optarg;
			break;
		case 'k':
			std::cout <<  "k"<<atoi(optarg)<<"\n";
			maxkeypoints=atoi(optarg);
			break;

		}
	}

	if(optind>=argc)
	{
		help();

		exit(0);
	}


	Mat image = imread(argv[optind] );
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	if(gray.empty() )
	{
		std::cerr <<  "Can't read  the image\n";
		return -1;
	}


	// detecting keypoints. Default OpenCV Values
	cv::Ptr< ORB > orb = cv::ORB::create(maxkeypoints);
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	clock_t t1 = clock();

	orb->detect(image, keypoints);
	orb->compute(image, keypoints, descriptors);

	clock_t t2 = clock();

	//Write keypoints and descriptors in database
	const String imgname = argv[optind];
	size_t lastindex = imgname.find_last_of(".");
	string filename = imgname.substr(0, lastindex)+".yml";

	//Save new name in database
	vector<string> files;
	try{
		//Intentamos leer el archivo aunque puede que no exista
		FileStorage fs("ORB_BD.yml",FileStorage::READ);
		fs["files"] >> files;
		fs.release();
	} catch(const std::exception& e){

	}
	//Reescribir el archivo
	files.push_back(filename);
	FileStorage fs2("ORB_BD.yml",FileStorage::WRITE);
	fs2 << "files" << files;
	fs2.release();

	FileStorage fs(filename,FileStorage::WRITE);
	fs << "keypoints" << keypoints;
	fs << "descriptors" << descriptors;
	fs << "img" << image;
	fs.release();
	std::cout << "Keypoints and descriptors saved in: " << filename << "\n";


	//Draw Keypoints

	cv::Mat output_img;
	cv::drawKeypoints(image,
			keypoints,
			output_img,
			cv::Scalar::all(-1),
			cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	clock_t t3 = clock();

	//detector(gray, mask, keypoints1, descriptors1); //2.4.11

	std::cout << (t2-t1)*1000.0/CLOCKS_PER_SEC << " ms to compute ORB\n";
	std::cout << "Detected " << keypoints.size() << "\n";
	cv::imshow("orb_mwe", output_img);
	cv::waitKey(0);




	return 0;
}







