
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
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

#define DEFAULT_THRESH_SURF_OPENCV 3000 //OpenCV Doc recommends 300--500
#define DEFAULT_OCTAVES_SURF_OPENCV 10

void help()
{
	std::cout << "surf_opencv_two [options] imagefile\n";
	std::cout << "-O #: number of octaves. Default=" << DEFAULT_OCTAVES_SURF_OPENCV <<"\n";

	std::cout << "-T #: blob response threshold.Default= "<< DEFAULT_THRESH_SURF_OPENCV <<"\n";
	std::cout << "-u 0/1: run in rotation invariant mode. Default = " << true <<"\n";;
}

int main(int argc, char** argv)
{
	int opt;
	cv::namedWindow("points");
	double hessianThreshold = DEFAULT_THRESH_SURF_OPENCV;
	int nOctaves = DEFAULT_OCTAVES_SURF_OPENCV;
	int nOctaveLayers = 2;
	bool extended = false; // if false 64 elements descriptor. If true 128 elements descriptor
	bool upright = false ; // Orientation of descriptor is estimated if false

	std::string  outputfilename;
	while((opt=getopt(argc,argv,"hO:T:u:O:"))!=-1)
	{

		switch(opt)
		{
		case 'h':
			help();
			exit(0);
			break;
		case 'O':
			nOctaves=atoi(optarg);
			break;
		case 'T':
			hessianThreshold = atof(optarg);
			break;
		case 'u':
			upright=atoi(optarg);
			break;
		case 'o':
			outputfilename = optarg;
			break;

		}
	}

	if(optind>=argc)
	{
		help();

		exit(0);
	}
	//Save new name in database
	vector<string> files,objects;
	try{
		//Intentamos leer el archivo aunque puede que no exista
		FileStorage fs("BD.yml",FileStorage::READ);
		fs["files"] >> objects;
		fs.release();
	} catch(const std::exception& e){

	}
	//Reescribir el archivo
	String imgname = argv[optind];
	size_t lastindex = imgname.find_last_of(".");
	string name =imgname.substr(0, lastindex);
	string ext = imgname.substr(lastindex, imgname.size());
	objects.push_back(name);
	FileStorage fs2("BD.yml",FileStorage::WRITE);
	fs2 << "files" << objects;
	fs2.release();
	////
	//Search for all images of an object
	bool whi=true;
	int imgnum = 1;
	while(whi){


		imgname = name +std::to_string(imgnum) + ext;
		struct stat buffer;
		if(stat (imgname.c_str(), &buffer) == 0){
			files.push_back( imgname );
			imgnum++;
		} else{
			whi=false;
		}
	}



	/////
	for(int i=0;i<files.size();i++){

		Mat image = imread(files[i] );
		if(image.empty() )
		{
			std::cerr <<  "Can't read  the image "<<files[i]<<"\n";
			return -1;
		}
		Mat gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		// detecting keypoints. Default OpenCV Values

		cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
		//SURF detector( hessianThreshold, nOctaves, nOctaveLayers, extended, upright); //2.4.11

		vector<KeyPoint> keypoints1;
		Mat descriptors1;
		//Mat mask;
		//Detect points and descriptors
		clock_t t1 = clock();
		f2d->detect( gray, keypoints1 );
		f2d->compute( gray, keypoints1, descriptors1 );
		//detector(gray, mask, keypoints1, descriptors1); //2.4.11
		clock_t t2 = clock();


		Scalar color = Scalar::all(-1);
		int flagsp =DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
		cv::Mat imgpoints1;

		//Write keypoints and descriptors in database
		size_t lastindex = files[i].find_last_of(".");
		string filename = files[i].substr(0, lastindex)+".yml";

		FileStorage fs(filename,FileStorage::WRITE);
		fs << "keypoints" << keypoints1;
		fs << "descriptors" << descriptors1;
		fs << "img" << image;
		fs.release();
		std::cout << "Keypoints and descriptors saved in: " << filename << "\n";
		std::cout << (t2-t1)*1000.0/CLOCKS_PER_SEC << " ms to compute SURF with DEFAULT_THRESH_SURF_OPENCV = "<<hessianThreshold <<" and DEFAULT_OCTAVES = " <<nOctaves<<"\n";
		std::cout << "Number of keypoints found: " << keypoints1.size() << "\n";
	}
	return 0;
}






