#include <getopt.h>


#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>

#if cimg_OS==2 //Windows
#include "getopt.h"
#else
#include <unistd.h>
#include <stdlib.h>
#endif

#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>



using namespace cv;
using namespace std;

#define DEFAULT_TH 0.04
void help()
{
	std::cout << "sift_one [options] imagefile\n";
	std::cout << " -T # : Hessian Threshold. Default: " << DEFAULT_TH << "\n";

}

int main(int argc, char** argv)
{
	int opt;
	cv::namedWindow("points");

	int nfeatures = 0;
	int nOctaveLayers = 3; //Lowe
	double contrastThreshold = 0.04;
	double edgeThreshold = 10.0;
	double sigma = 1.6;
	std::string  outputfilename;
	while((opt=getopt(argc,argv,"hT:o:"))!=-1)
	{

		switch(opt)
		{
		case 'h':
			help();
			exit(0);
			break;
		case 'T':
			contrastThreshold = atof( optarg );
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


	cv::Mat image = cv::imread(argv[optind]);
	cv::Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);

	if(gray.empty() )
	{
		std::cerr <<  "Can't read  the image\n";
		return -1;
	}


	// detecting keypoints. Default OpenCV Values
	cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create( nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
	//SIFT siftDetector( nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma); //2.4.11



	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptors1;
	//Mat mask;

	//Detect points and descriptors
	clock_t t1 = clock();
	f2d->detect( gray, keypoints1 );
	//siftDetector(gray, mask, keypoints1, descriptors1);//opencv 2.4.11
	clock_t t2 = clock();
	Scalar color = Scalar::all(-1);
	int flagsp =DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	cv::Mat imgpoints1;
	drawKeypoints(image, keypoints1, imgpoints1, color, flagsp );
	std::cout << "Detected " << keypoints1.size() << "\n";
	std::cout << "Time to detect points and compute descriptors: " << (t2-t1)*1000.0/CLOCKS_PER_SEC << " ms\n";


	if(outputfilename.size() > 0)
		imwrite( outputfilename.c_str(), imgpoints1);

	imshow("points", imgpoints1);
	waitKey(0);

	return 0;
}






