
#include <iostream>

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

#define cimg_plugin1 "cvMat.h"

#include <CImg.h>
#include "hough.h"
#include <math.h>
using namespace cv;
using namespace std;
#include "upv_defaults.h"
#include <vector>
#include <string>



void help()
{
	std::cout << "surf_opencv_two [options] imagefile1 imagefile2\n";
	std::cout << "surf_opencv_two [options] imagefile1 camera_number\n";
	std::cout << "-O #: number of octaves. Default=" << DEFAULT_OCTAVES_SURF_OPENCV <<"\n";

	std::cout << "-T #: blob response threshold.Default= "<< DEFAULT_THRESH_SURF_OPENCV <<"\n";
	std::cout << "-u 0/1: 0:Estimate point orientation. 1: Assume same orientation. Default = " << 0 <<"\n";;
}



vector<string> readfiles(){
	vector<string> files;
	FileStorage fs("ORB_BD.yml",FileStorage::READ);
	fs["files"] >> files;
	fs.release();
	if(files.size()<1){
		std::cerr <<"Can't find ORB Data Base\n";
		exit(0);
	}
	else
		return files;
}

void readDB(int fsize, vector<string> files, vector<KeyPoint> keypoints1[fsize],Mat descriptors1[fsize],Mat img1[fsize]){
	for(int i =0; i<fsize; ++i){
		String filename = files.at(i);
		std::cout << "Estoy leyendo:"<<filename<<"\n";
		FileStorage fs2(filename, FileStorage::READ);
		fs2["keypoints"] >> keypoints1[i];
		fs2["descriptors"] >> descriptors1[i];
		fs2["img"] >> img1[i];
		fs2.release();
	}
}



int main(int argc, char** argv)
{
	cimg_library::CImgDisplay cdisp;

	int flags = CV_WINDOW_AUTOSIZE;
	cv::namedWindow( "output_win", flags);
	int opt;

	double hessianThreshold = DEFAULT_THRESH_SURF_OPENCV; //OpenCV Doc recommends 300--500
	int nOctaves = DEFAULT_OCTAVES_SURF_OPENCV;
	int nOctaveLayers = 2;
	bool extended=false; // if false 64 elements descriptor. If true 128 elements descriptor
	bool upright=false ; // Orientation of descriptor is estimated if false

	float matchRatio = DEFAULT_MATCH_RATIO;

	bool cameraMode = false;

	int screen_width = cdisp.screen_width();
	//	int screen_height = cdisp.screen_height();


	// Process cmd line options
	while((opt=getopt(argc,argv,"hO:T:u:"))!=-1)
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

	vector<KeyPoint> keypoints2;
	vector<KeyPoint> keypoints1e[fsize];
	Mat im1e[fsize];
	Mat im2e;
	Mat img2;
	//vector<DMatch>   houghMatches[fsize];
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
		float capt_width = 800;
		float capt_height = 600;
		capture.set(CV_CAP_PROP_FRAME_WIDTH , capt_width);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, capt_height);
		capture >> img2;
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

	// detecting keypoints. Default OpenCV Values
	cv::Ptr< ORB > orb = cv::ORB::create(500);
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

	int distance_type = NORM_HAMMING;
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
		orb->detect(img2, keypoints2);
		orb->compute(img2, keypoints2, descriptors2);
		clock_t t4 = clock();

		vector<DMatch> matches12[fsize];
		vector<vector<DMatch> > matches12nn[fsize];
		vector<DMatch> matches12ratio[fsize];
		//vector<DMatch>  houghMatches[fsize];
		bool hayalgo = false;
		String quehay = "Lista de objetos:";

		for(int i =0; i<fsize;i++){
			clock_t tiempo0 = clock();
			if (keypoints1[i].size() > 20 && keypoints2.size() > 20 &&  !descriptors2.empty()) {
				//NN with crosscheck
				clock_t t5 = clock();
				try{
					bfmatcher.match( descriptors1[i], descriptors2, matches12[i] );
				}
				catch(Exception e){
					std::cout<< "exeption::::::"<<descriptors1[i].size()<<"D2:"<<descriptors2.size()<<"aa:"<< e.what()<<"\n";
					exit(0);
				}
				clock_t t6 = clock();
				clock_t t7 = clock();
				// 2-NN with ratio validation
				int knn = 2;
				t7 = clock();
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
				clock_t t9 = clock();
				//Hough
				int nscales = 5;
				std::vector< float > scales(nscales);
				int nrotations = 1;
				if ( !upright ) //if upright I know that there is no rotation
					nrotations = 12;
				std::vector < float > rotations(nrotations);
				for (int r = 0; r < rotations.size() ; r++)
					rotations[r] = -150 + 30.0 * r;

				if (nrotations == 1)
					rotations[0] = 0.0;

				int centrals = nscales / 2;
				for (int s = 0; s < nscales; s++){
					float ss = s - centrals;
					scales[s] = powf( float(2.0), float(ss / 2.0) );
				}

				int minMatchNumber = 7;
				int displayH = 0;
				if (cameraMode)
					displayH = 0;
				//HoughUPV houghupv(   rotations,  scales, img1[i].cols, img1[i].rows,  img2.rows, img2.cols,  minMatchNumber, displayH);

				clock_t t10 = clock();
				if(matches12ratio[i].size() < 5){
					std::cout << "2NN-Ratio Matches " << matches12ratio[i].size() << "\n";
					//matches12ratio[i].clear();
				}
				else
					//Generate output

					if (!cameraMode) {
						std::cout << "Descriptor Size =" << descriptors2.cols << " cols x " << descriptors2.rows << " rows\n";
						std::cout << "Detected in img1 " << keypoints1[i].size() << "\n";
						std::cout << "Detected in img2 " << keypoints2.size() << "\n";
						std::cout << "2NN-Ratio Matches " << matches12ratio[i].size() << "\n";
						//std::cout << "Hough Matches " << houghMatches[i].size() << "\n";


						std::cout << float(t2 - t1 + t4 - t3) /2.0 / CLOCKS_PER_SEC << "seconds/image to detect points and features\n";
						std::cout << float(t6 - t5) / CLOCKS_PER_SEC << " seconds to compute BF matching with crosscheck\n";
						std::cout << float(t8 - t7) / CLOCKS_PER_SEC << " seconds for 2-NN matching\n";
						std::cout << float(t10 - t9) / CLOCKS_PER_SEC << " seconds for Hough\n";
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
			imshow("output_win", img2);
			if (!cameraMode)
				drawMatches(im1e[i], keypoints1e[i], im2e, keypoints2e, matches12ratio[i], img_matchesnn2);
			if(matches12ratio[i].size()>10){

				size_t lastindex = files[i].find_last_of(".");
				string name = files[i].substr(0, lastindex);
				quehay = quehay + " "+name;
				//std::cout <<"Match::"<< quehay <<"i:"<<i <<"hayalgo:"<<hayalgo<<  "\n";
				hayalgo =true;
			}
			if(i==fsize-1 && hayalgo)
				std::cout << quehay <<"\n";
			if (! cameraMode) {
				cimg_library::CImg<unsigned char> cim_nn2( img_matchesnn2);
				cim_nn2.display("Matches_after_Hough",false);
				continue;
			}

			char c = cv::waitKey(20) ; //Wait 20 mseconds
			if( c == 27 )
				break;
			if( c == ' ' )
				cv::waitKey(0);

			clock_t tiempo1 = clock();
			std::cout << float(tiempo1 - tiempo0) / CLOCKS_PER_SEC << " segundos en iterar\n";
		}
	}
	return 0;
}






