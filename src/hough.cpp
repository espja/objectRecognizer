/*
 * hough.cpp
 *
 *  Created on: Nov 21, 2011
 *      Author: aalbiol
 */
#include <vector>
#include <stdio.h>
#include <iostream>



#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <CImg.h>
using namespace cv;
using namespace std;

#include "hough.h"
HoughUPV::HoughUPV( const std::vector<float> & rotations, const std::vector<float> & scales, int  img1Width,int img1Height,  int  img2Width,int img2Height, int minMatchNumber, int displayH)
{
	m_scales = scales;
	m_rotations = rotations;
	m_width1 = img1Width;
	m_height1 = img1Height;
	m_width2 = img2Width;
	m_height2 = img2Height;
	m_minMatchNumber = minMatchNumber;
	m_displayHough = displayH;
}


void HoughUPV::process(	const vector<KeyPoint> & keypoints1,  const vector<KeyPoint> & keypoints2, const vector<DMatch> & matches, vector<DMatch> & goodMatches)
{


	int nscales = m_scales.size();
	int nrotations = m_rotations.size();

	if (0 == nscales || 0 == nrotations)
	{
		std::cerr <<"Error in Hough. Zero rotations or scales\n";
		exit(0);
	}

	int nbins = 24;
	float nbins2 = nbins / 2.0;
	m_houghT.assign( nrotations, nscales, nbins, nbins).fill(0);

	goodMatches.clear();

	std::vector< int > center2x (nscales);
	std::vector< int > center2y (nscales);
	std::vector< int > width2 (nscales);
	std::vector< int > height2 (nscales);

	for( int k = 0 ; k < nscales ; k++)
	{
		width2[k] = m_width2;
		height2[k] = m_height2;
		center2x[k] = width2[k] / 2;
		center2y[k] = height2[k] / 2;
	}

	int center1x = m_width1 / 2;
	int center1y = m_height2 / 2;

	for( int r = 0 ; r < nrotations ; r++){
		float f = 3.1416/180.0;
		float co = std::cos ( float (f * m_rotations[r]) );
		float se = std::sin ( float (f * m_rotations[r]) );
		for (unsigned int i = 0; i < matches.size(); ++i)
		{
			float x1 = keypoints1.at(matches[i].queryIdx).pt.x ;
			float y1 = keypoints1.at(matches[i].queryIdx).pt.y ;
			float ddx = x1 - center1x;
			float ddy = y1 - center1y;

			float dx = co * ddx + se * ddy;
			float dy = -se * ddx + co * ddy;

			for( int k = 0 ; k < nscales ; k++){
				float x2 = keypoints2.at(matches[i].trainIdx).pt.x ;
				float y2 = keypoints2.at(matches[i].trainIdx).pt.y ;
				float p2x = x2 - m_scales [k] * dx - center2x[k];
				float p2y = y2 - m_scales [k] * dy - center2y[k];
				float binxf = nbins * p2x / width2[k] / 2.0 + nbins2;
				float binyf = nbins * p2y / height2[k] /2.0 + nbins2;
				int binx = floor( binxf );
				int biny = floor( binyf );
				int binx2 = ceil(binxf);
				int biny2 = ceil(binyf);
				if(binx >= 0 && biny >= 0 && binx < nbins && biny < nbins)
				{
					m_houghT( r,k, binx,biny) ++;
				}

				if(binx2 >= 0 && biny2 >= 0 && binx2 < nbins && biny2 < nbins)
				{
					m_houghT( r,k, binx2,biny2) ++;
				}

				if(binx >= 0 && biny2 >= 0 && binx < nbins && biny2 < nbins)
				{
					m_houghT( r,k, binx,biny2) ++;
				}

				if(binx2 >= 0 && biny >= 0 && binx2 < nbins && biny < nbins)
				{
					m_houghT( r,k, binx2,biny) ++;
				}
			}
		}
	}


	//Search for the maximum
	int scale_max, xcoord_max, ycoord_max, rotation_max;
	int houghMaximum = -1;
	for( int r = 0 ; r < nrotations ; r++){
		for (int s = 0; s < nscales; s ++ ) {

			for (int  x= 0; x < nbins; x++)
				for (int  y= 0; y < nbins; y++)
				{
					if (m_houghT (r, s, x, y) > houghMaximum)
					{
						houghMaximum = m_houghT (r, s, x, y);
						rotation_max = r;
						scale_max = s;
						xcoord_max = x;
						ycoord_max = y;
					}
				}
		}
	}
	//Print scale and position of maximum
	if ( m_displayHough ) {
		if (houghMaximum >= m_minMatchNumber ) {
			std::cout << "Hough Maximum: " << houghMaximum;
			std::cout <<  " Rotation = " << m_rotations[rotation_max];
			std::cout <<  " degrees. Scale = " << m_scales[scale_max];
			std::cout << " xoffset = " << xcoord_max - nbins/2 ;
			std::cout << " yoffset = " << ycoord_max - nbins /2;
			std::cout << "\n";
		}
		else
		{
			std::cout << "Hough Maximum: " << houghMaximum ;
			std::cout << " < " << m_minMatchNumber << "\n";
			return;
		}
	}
	//Select matches that have contributed to maximum
	float co = std::cos ( float(3.1416 /180.0 * m_rotations[rotation_max] ));
	float se = std::sin ( float(3.1416 /180.0 * m_rotations[rotation_max] ));
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		float x1 = keypoints1.at(matches[i].queryIdx).pt.x ;
		float y1 = keypoints1.at(matches[i].queryIdx).pt.y ;
		float ddx = x1 - center1x;
		float ddy = y1 - center1y;

		float dx = co * ddx + se * ddy;
		float dy = -se * ddx + co * ddy;

		float x2 = keypoints2.at(matches[i].trainIdx).pt.x ;
		float y2 = keypoints2.at(matches[i].trainIdx).pt.y ;
		float p2x = x2 - m_scales [scale_max] * dx - center2x[scale_max];
		float p2y = y2 - m_scales [scale_max] * dy - center2y[scale_max];

		float binxf = nbins * p2x / width2[scale_max] / 2.0 + nbins2;
		float binyf = nbins * p2y / height2[scale_max] /2.0 + nbins2;
		int binx = floor( binxf );
		int biny = floor( binyf );
		int binx2 = ceil(binxf);
		int biny2 = ceil(binyf);
		if(binx == xcoord_max && biny == ycoord_max && binx)
		{
			goodMatches.push_back(matches[i]);
			continue;
		}
		if(binx == xcoord_max && biny2 == ycoord_max && binx)
		{
			goodMatches.push_back(matches[i]);
			continue;
		}
		if(binx2 == xcoord_max && biny2 == ycoord_max && binx)
		{
			goodMatches.push_back(matches[i]);
			continue;
		}
		if(binx2 == xcoord_max && biny == ycoord_max && binx)
		{
			goodMatches.push_back(matches[i]);
			continue;
		}

	}
	if(m_displayHough) {
		cimg_library::CImg<int> d(nscales* nbins, nrotations *	nbins);
		for (int r = 0 ; r < nrotations ; r++){

			for(int s = 0; s < nscales; s++) {
				for(int x= 0; x < nbins; x++){
					for(int y = 0; y < nbins; y++){
						d( nbins * s + x, nbins * r + y) = m_houghT (r, s, x, y);
					}
				}
			}
		}


		d.display("Hough",false);
	}
}
