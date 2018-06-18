/*
 * hough.h
 *
 *  Created on: Nov 21, 2011
 *      Author: aalbiol
 */

#ifndef HOUGH_H_
#define HOUGH_H_


class HoughUPV {
public:
	HoughUPV(  const std::vector<float> & rotations,  const std::vector<float> & scales, int  img1Width,int img1Height,  int  img2Width,int img2Height, int minMatchNumber, int displayH);
	void process( const std::vector<cv::KeyPoint> & keypoints1, const std::vector<cv::KeyPoint> & keypoints2, const std::vector<cv::DMatch> & matches, std::vector<cv::DMatch> & goodMatches);
private:
	std::vector<float>  m_scales;
	std::vector<float>  m_rotations;
	int m_width1;
	int m_height1;
	int m_width2;
	int m_height2;
	int m_minMatchNumber;
	cimg_library::CImg<int>   m_houghT;
	int m_displayHough;
};


#endif /* HOUGH_H_ */
