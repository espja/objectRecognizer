
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#if cimg_OS==2 //Windows
#include "getopt.h"
#else
#include <unistd.h>
#include <stdlib.h>
#endif



#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"

#define cimg_plugin1 "cvMat.h"

#include <CImg.h>
#include "hough.h"
#include <math.h>
using namespace cv;
using namespace std;
#include "upv_defaults.h"
#include <vector>
#include <string>


const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 1.0f;


void help()
{
	std::cout << "surf_opencv_two [options] imagefile1 imagefile2\n";
	std::cout << "surf_opencv_two [options] imagefile1 camera_number\n";
	std::cout << "-O #: number of octaves. Default=" << DEFAULT_OCTAVES_SURF_OPENCV <<"\n";

	std::cout << "-T #: blob response threshold.Default= "<< DEFAULT_THRESH_SURF_OPENCV <<"\n";
	std::cout << "-u 0/1: 0:Estimate point orientation. 1: Assume same orientation. Default = " << 0 <<"\n";;
}


//Get all file names included in the DB
vector<string> readfiles(){
	FileStorage fs("BD.yml",FileStorage::READ);
	vector<string> objects,files;
	fs["files"] >> objects;
	fs.release();
	if(objects.size()<1){
		std::cerr <<"Can't find SURF Data Base\n";
		exit(0);
	} else{
		int imgcounter = 0;
		for(int i = 0; i<objects.size();i++){
			bool whi=true;
			int imgnum = 1;
			while(whi){
				size_t lastindex = objects[i].find_last_of(".");
				string imgname = objects[i].substr(0, lastindex);
				imgname = objects[i] +std::to_string(imgnum) + ".yml";
				struct stat buffer;
				if(stat (imgname.c_str(), &buffer) == 0){
					files.push_back( imgname );
					imgnum++;
					imgcounter++;
				} else{
					whi=false;
				}
			}

		}
		return files;
	}
}

//Read the files with descriptors and keypoints stored for each image in DB
void readDB(int fsize, vector<string> files, vector<KeyPoint> keypoints1[fsize],Mat descriptors1[fsize],Mat img1[fsize]){
	string message = "Loading DataBase ";
	string symbol = "#####";
	int percentage = 0;
	int salto = std::ceil(100.0/double(fsize));
	std::cout << message;

	for(int i =0; i<fsize; ++i){
		//Progress bar
		message += symbol;
		percentage += salto;
		if(percentage >100){
			percentage = 100;
		}
		std::clog << "\r [" << std::setw(3) << static_cast<int>(percentage) << "%] "
				<< message << std::flush;
		//Read file stored in DB
		String filename = files.at(i);
		FileStorage fs2(filename, FileStorage::READ);
		fs2["keypoints"] >> keypoints1[i];
		fs2["descriptors"] >> descriptors1[i];
		fs2["img"] >> img1[i];
		fs2.release();
		cvtColor(img1[i], img1[i], cv::COLOR_RGB2GRAY);
	}
	std::clog << "\n\n";
}

//Print the name of the objects at the end of iteration through DB (only when cameraMode = 0)
void computeResults(vector<string> files,vector<DMatch> matches12ratio[files.size()]){
	String out;
	for(int i=0;i<files.size();i++){
		size_t lastindex = files[i].find_last_of(".");
		string name = files[i].substr(0, lastindex);
		out = out + name + " -- " + std::to_string(matches12ratio[i].size())+"\n";
	}
	out = out +"End of computeResults\n";
	std::cout <<out;
}


Mat drawRectangle(
		const Mat& img1,
		const Mat& img2,
		const std::vector<KeyPoint>& keypoints1,
		const std::vector<KeyPoint>& keypoints2,
		std::vector<DMatch>& matches,
		double scale,
		string name
)
{
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());
	//std::vector< DMatch > good_matches;
	double minDist = matches.front().distance;
	double maxDist = matches.back().distance;

	const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	std::vector<DMatch> good_matches;
	for( int i = 0; i < ptsPairs; i++ )
	{
		good_matches.push_back( matches[i] );
	}
	// drawing the results
	Mat img_matches;

	drawMatches( img1, keypoints1, img2, keypoints2,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for( size_t i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f( (float)img1.cols, 0 );
	obj_corners[2] = Point2f( (float)img1.cols, (float)img1.rows );
	obj_corners[3] = Point2f( 0, (float)img1.rows );

	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	for (int n =0; n< scene_corners.size(); n++) {
		scene_corners[n].x /= scale;
		scene_corners[n].y /= scale;
	}

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line( img2,
			scene_corners[0], scene_corners[1],
			Scalar( 0, 255, 0), 2, LINE_AA );
	line( img2,
			scene_corners[1], scene_corners[2],
			Scalar( 0, 255, 0), 2, LINE_AA );
	line( img2,
			scene_corners[2], scene_corners[3] ,
			Scalar( 0, 255, 0), 2, LINE_AA );
	line( img2,
			scene_corners[3] , scene_corners[0],
			Scalar( 0, 255, 0), 2, LINE_AA );
	int font =FONT_HERSHEY_SIMPLEX;

	//Find bottom left corner of object
	Point2f minxmaxy = Point2f(img2.cols,0);
	for(int j = 0;j<scene_corners.size();j++){
		//Find biggest y
		if(minxmaxy.y<=scene_corners[j].y ){
			minxmaxy.y = scene_corners[j].y;
			if(minxmaxy.x>=scene_corners[j].x)
				minxmaxy.x = scene_corners[j].x;
		}
	}

	float xtext = minxmaxy.x;
	float ytext = minxmaxy.y+40;
	putText(img2, name, cvPoint(xtext,ytext),
			FONT_HERSHEY_COMPLEX_SMALL,2, cvScalar(255,255,255), 2, CV_AA);
	//imshow("Good Matches & Object detection", img_matches );
	//waitKey();
	return img2;
}

//Get all object names
void getNames(vector<string> files,string names[files.size()]){
	for(int i=0;i<files.size();i++){
		size_t lastindex = files[i].find_last_of(".");
		names[i] = files[i].substr(0, lastindex-1);
	}
}
//Check if we've already found this object in DB to avoid checking with another perspective
bool alreadyIn(vector<string> Objs, string name){
	for(int i = 0; i<Objs.size();i++){
		if(!Objs.at(i).compare(name)){
			return true;
		}
	}
	return false;
}

void show_progress_bar(std::ostream& os, int time,
		std::string message, char symbol = '*')
{

	int bar_length = 70;
	// not including the percentage figure and spaces

	if (message.length() >= bar_length) {
		os << message << '\n';
		message.clear();
	} else {
		message += " ";
	}

	int progress_level = 100.0 / (bar_length - message.length());

	std::cout << message;

	for (double percentage = 0; percentage <= 100; percentage += progress_level) {
		message += symbol;
		os << "\r [" << std::setw(3) << static_cast<int>(percentage) << "%] "
				<< message << std::flush;
		std::this_thread::sleep_for(std::chrono::milliseconds(time));
	}
	os << "\n\n";
}



int main(int argc, char** argv)
{
	cimg_library::CImgDisplay cdisp;

	int flags = CV_WINDOW_AUTOSIZE;
	cv::namedWindow( "output_win", flags);
	int opt;

	double hessianThreshold = 3000;
	int nOctaves = DEFAULT_OCTAVES_SURF_OPENCV;
	int nOctaveLayers = 2;
	bool extended=false; // if false 64 elements descriptor. If true 128 elements descriptor
	bool upright=false ; // Orientation of descriptor is estimated if false
	//Resolution for camera
	float capt_width = 800;
	float capt_height = 600;

	float matchRatio = DEFAULT_MATCH_RATIO;

	bool cameraMode = false;

	int screen_width = cdisp.screen_width();
	//	int screen_height = cdisp.screen_height();


	// Process cmd line options
	while((opt=getopt(argc,argv,"hO:T:u:w"))!=-1)
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
		case 'w':
			capt_width =atoi(optarg);
			break;
		}
	}

	//Not enough args
	if(optind >= argc)
	{
		help();
		exit(0);
	}
	vector<string> files = readfiles();
	int fsize = files.size();
	vector<KeyPoint> keypoints1[fsize];

	Mat descriptors1[fsize];
	Mat img1[fsize];
	clock_t t1 = clock();

	readDB(fsize,files,keypoints1,descriptors1,img1);
	clock_t t2 = clock();
	std::cout << "DB read after =" << float(t2 - t1) / CLOCKS_PER_SEC <<"s \n";
	//Get object names
	string names[fsize];
	getNames(files,names);

	vector<KeyPoint> keypoints2;
	vector<KeyPoint> keypoints1e[fsize];
	Mat im1e[fsize];
	Mat im2e;
	Mat img2;
	VideoCapture capture;

	if( strlen (argv[optind]) > 1){ // 1 image
		img2 = imread(argv[optind]);
		if(img2.empty() ){
			std::cerr <<  "optind: " << optind << "\n";
			std::cerr <<  "Can't read  the image: " << argv[optind] << "\n";
			return -1;
		}
	}
	else { //camera input
		cameraMode = true;
		int cam = atoi(argv[optind]);
		capture.open(cam);
		if (!capture.isOpened() ) {
			std::cerr << "Can't open camera " << cam << "\n";
			exit(0);
		}
		//float capt_width = 600;
		//float capt_height = 450;
		std::cout << "capt_width " << capt_width  <<" capt_height " <<capt_height << "\n";
		capture.set(CV_CAP_PROP_FRAME_WIDTH , capt_width);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, capt_height);
		capture >> img2;
		std::cout << "img2widht " << img2.cols  <<" img2height " <<img2.rows << "\n";
		cvtColor(img2, img2, cv::COLOR_RGB2GRAY);
	}
	double scale[fsize];

	for(int i =0; i<fsize;i++){
		int total_width = img1[i].cols+img2.cols;
		// Check if screen is large enough. If not, compute scale[i] factor for output
		scale[i]= double(screen_width) / total_width;
		if (scale[i] < 1) {
			scale[i] *=  0.8;
		}
	}
	// Set up detector.
	//SURF detector( hessianThreshold, nOctaves, nOctaveLayers, extended, upright); 2.4.11

	cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

	//scale[i] points 1 for representation
	for(int i =0; i<fsize;i++){
		im1e[i]=img1[i];
		keypoints1e[i] = keypoints1[i];
		if (scale[i] < 1) {
			resize(img1[i],im1e[i], Size(), scale[i], scale[i], INTER_NEAREST );
			for (int n =0; n< keypoints1e[i].size(); n++) {
				KeyPoint & p = keypoints1e[i][n];
				p.pt.x *= scale[i];
				p.pt.y *= scale[i];
			}
		}
	}
	// Create MAtchers
	Ptr<DescriptorMatcher> dmatcher = DescriptorMatcher::create ("BruteForce");

	int distance_type = NORM_L2;
	bool crossCheck=true;
	BFMatcher bfmatcher(distance_type, crossCheck);
	//Loop images
	for(int n = 0; (n< 1) || cameraMode;  n++) {
		if(cameraMode)
			capture >> img2;

		//Detect points on image 2
		vector<KeyPoint> keypoints2;
		Mat descriptors2;

		clock_t t3 = clock();
		//detector(img2, mask, keypoints2, descriptors2);
		f2d->detect( img2, keypoints2 );
		f2d->compute( img2, keypoints2, descriptors2 );
		clock_t t4 = clock();
		vector<DMatch> matches12[fsize];
		vector<vector<DMatch> > matches12nn[fsize];
		vector<DMatch> matches12ratio[fsize];
		vector<string>objectsDetected;
		bool hayalgo = false;
		String quehay = "Lista de objetos:";
		clock_t tiempo0 = clock();
		//Iterate through all objects in DB
		for(int i =0; i<fsize;i++){

			if(!alreadyIn(objectsDetected,names[i])){
				if (keypoints1[i].size() > 20 && keypoints2.size() > 20 &&  !descriptors2.empty()) {
					//NN with crosscheck
					clock_t t5 = clock();
					bfmatcher.match( descriptors1[i], descriptors2, matches12[i] );
					clock_t t6 = clock();

					// 2-NN with ratio validation
					int knn = 2;
					clock_t t7 = clock();
					dmatcher->knnMatch( descriptors1[i], descriptors2, matches12nn[i], knn);

					//Select Matches by distance ratio and cross-check
					for(int m = 0; m< matches12nn[i].size() ; m++)
					{
						vector<DMatch> & mm = matches12nn[i][m];
						if ( mm.size() == 1 )
							matches12ratio[i].push_back( mm.front() );
						else if (mm.size() > 1 )
						{
							DMatch & candidate_match = mm.front();
							float d1 = candidate_match.distance;
							const DMatch & mm2 = mm[1];
							float d2 = mm2.distance;
							if ( d1 / d2 < matchRatio ) {
								//Now cross-check
								for(int j= 0; j < matches12[i].size(); j++) {
									DMatch & bfmatch = matches12[i][j];
									if ( (bfmatch.trainIdx == candidate_match.trainIdx) && (bfmatch.queryIdx == candidate_match.queryIdx)) {
										matches12ratio[i].push_back( mm.front() );
										break;
									}
								}
							}
						}
					}
					clock_t t8 = clock();

					if(matches12ratio[i].size() < 20)
						matches12ratio[i].clear();
					else{
						//Generate output
						if (!cameraMode) {
							std::cout <<"Object detected: "<<names[i]<<"\n";
							std::cout << "Descriptor Size =" << descriptors2.cols << " cols x " << descriptors2.rows << " rows\n";
							std::cout << "Detected in img1 " << keypoints1[i].size() << "\n";
							std::cout << "Detected in img2 " << keypoints2.size() << "\n";
							std::cout << "2NN-Ratio Matches " << matches12ratio[i].size() << "\n";

							//std::cout << float(t2 - t1 + t4 - t3) /2.0 / CLOCKS_PER_SEC << "seconds/image to detect points and features\n";
							std::cout << float(t6 - t5)*1000/ CLOCKS_PER_SEC << " mseconds to compute BF matching with crosscheck\n";
							std::cout << float(t8 - t7)*1000/ CLOCKS_PER_SEC << " mseconds for 2-NN matching\n\n";
						}
					}

				}

				vector< KeyPoint > keypoints2e = keypoints2;
				im2e = img2;
				if (scale[i] < 1) {
					for (int n =0; n< keypoints2e.size(); n++) {
						KeyPoint & p = keypoints2e[n];
						p.pt.x *= scale[i];
						p.pt.y *= scale[i];
					}
					resize(img2,im2e, Size(), scale[i], scale[i], INTER_NEAREST );
				}
				//		Scalar color = Scalar::all(-1);

				//		int flagsp =DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
				cv::Mat img_matchesnn2;
				cv::Mat img_matchesbf;

				if(matches12ratio[i].size()>20){
					quehay = quehay + " "+names[i];
					hayalgo =true;
					clock_t tiempo2 = clock();
					img2 = drawRectangle(im1e[i],img2,keypoints1e[i],keypoints2e,matches12ratio[i],scale[i],names[i]);
					clock_t tiempo3 = clock();
					objectsDetected.push_back(names[i]);
				}
				char c = cv::waitKey(1) ; //Wait 20 mseconds
				if( c == 27 )
					break;
				if( c == ' ' )
					cv::waitKey(0);
			}
			if(i==fsize-1){
				imshow("output_win", img2);
				if(hayalgo)
					std::cout << quehay <<"\n";
				if (! cameraMode) {
					cimg_library::CImg<unsigned char> cim_nn2( img2);
					cim_nn2.display("Matches",false);
					continue;
				}
			}
		}
		clock_t tiempo1 = clock();
		std::cout << float(tiempo1 - tiempo0)*1000 / CLOCKS_PER_SEC << " ms to compute SURF with DEFAULT_THRESH_SURF_OPENCV = "<<hessianThreshold <<" and DEFAULT_OCTAVES = " <<nOctaves<<"\n";
	}
	return 0;
}






