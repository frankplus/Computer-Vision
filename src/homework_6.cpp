#include "homework_6.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

const float DISTANCE_RATIO_THRESHOLD = 3.0;
const double RANSAC_REPROJ_THRESHOLD = 3.0;
const String RESULT_WIN = "result";

void main_homework_6()
{
    // get input paths
    string images_path, video_path;
    cout << "Insert objects images path (e.g. 'data/lab6/objects/*.png'): ";
    cin >> images_path;
    cout << "Insert video path (e.g. 'data/lab6/video.mov'): ";
    cin >> video_path;

    // load template images
    vector<String> images_paths;
    glob(images_path, images_paths);
    vector<Mat> template_images;
    for (String imgpath: images_paths) 
	{
        cout << imgpath << endl;
        template_images.push_back(imread(imgpath));
    }

	// load video
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened())
		cout << "could not open video" << endl;

    // locate and track objects showing results
    locateAndTrack(template_images, cap);

    waitKey(0);
    cap.release();
}

/**
 * Locate and track objects showing keypoints and bounding rectangles at each frame.
 * @param template_images Vector of images of the objects to detect and track
 * @param video_capture The video to analyze and detect the given objects
 */
void locateAndTrack(vector<Mat> template_images, VideoCapture video_capture)
{
	Mat frame, result;
	vector<vector<Point2f>> all_corners, all_keypoints;
	vector<Scalar> colors;

	// Locate template objects in the first frame of the video
	video_capture >> frame;
	result = frame.clone();
	for (Mat image: template_images)
	{
		vector<Point2f> corners, keypoints;
		detectAndMatchKeypoints(image, frame, corners, keypoints);
		Scalar color(rand()%256, rand()%256, rand()%256);

		all_corners.push_back(corners);
		all_keypoints.push_back(keypoints);
		colors.push_back(color);
	}

	// draw and show result
	drawKeypointsAndRectangles(result, all_keypoints, all_corners, colors);
	imshow(RESULT_WIN, result);
	waitKey(30);

	// Track features of the objects using pyramid Lukas-Kanade tracker and show result
	Mat new_frame;
	while (true)
	{
		video_capture >> new_frame;
		if (new_frame.empty())
            break;

		// for every object recompute keypoints and corners based on the optical flow
		trackObjects(frame, new_frame, all_keypoints, all_corners);

		// draw and show result
		result = new_frame.clone();
		drawKeypointsAndRectangles(result, all_keypoints, all_corners, colors);
		imshow(RESULT_WIN, result);
		waitKey(1);

		frame = new_frame.clone();
	}
}

/**
 * Find image1 features inside image2 by detecting the keypoints in both images and matching them. 
 * Then return the matched keypoints found inside image2 and the corners of image1 projected onto image2.
 * @param image1 The source template image to search inside image2.
 * @param image2 The image inside which we search image1 features.
 * @param corner_points The output points in the image2 coordinates representing the projection of the image1 corners.
 * @param points The output points in the image2 coordinates representing the projection of the image1 keypoints.
 */
void detectAndMatchKeypoints(Mat image1, Mat image2, vector<Point2f> &corner_points, vector<Point2f> &points) 
{
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();

    // detect keypoints
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
    sift->detect(image1, keypoints1); 
	sift->detect(image2, keypoints2); 

    // compute descriptors
	Mat descriptors1;
    Mat descriptors2;
    sift->compute(image1, keypoints1, descriptors1);
	sift->compute(image2, keypoints2, descriptors2);

    // matching descriptors
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);
	
	// find min distance
	float min_distance = matches[0].distance;
	for (DMatch match: matches) 
		if (match.distance < min_distance)
			min_distance = match.distance;

	// refine by selecting the matches with distance less than ratio * min_distance
	vector<DMatch> refined_matches;
	for (DMatch match: matches) 
		if (match.distance < DISTANCE_RATIO_THRESHOLD * min_distance)
			refined_matches.push_back(match);

	// extract points from keypoints based on matches
    vector<Point2f> points_from;
    vector<Point2f> points_to;
    for (int j=0; j < refined_matches.size(); ++j) 
	{
        points_from.push_back(keypoints1.at(refined_matches.at(j).queryIdx).pt);
        points_to.push_back(keypoints2.at(refined_matches.at(j).trainIdx).pt);
    }

	// use find homography to find inliers
    Mat mask;
    Mat H = findHomography(points_from, points_to, CV_RANSAC, RANSAC_REPROJ_THRESHOLD, mask);
	for (int j=0; j<mask.rows; j++) 
		if (mask.at<uchar>(j, 0))
			points.push_back(points_to[j]);

	// project corners
	vector<Point2f> corner_points_from{Point2f(0,0), Point2f(image1.cols,0), Point2f(image1.cols,image1.rows), Point2f(0,image1.rows)};
	perspectiveTransform(corner_points_from, corner_points, H);
}

/**
 * Draw the keypoints and the bounding rectangles on the given image, each object with different colors.
 * @param result The image onto which we draw the keypoints and the rectangles.
 * @param keypoints A vector of keypoints coordinates for every object.
 * @param corners The coordinates of the corners of the rectangles in clockwise or counterclockwise order for each object.
 * @param color The drawing colors for each object
 */
void drawKeypointsAndRectangles(Mat result, vector<vector<Point2f>> &all_keypoints, vector<vector<Point2f>> &all_corners, vector<Scalar> colors)
{
	for (int i=0; i<all_keypoints.size(); i++)
	{
		vector<Point2f> keypoints = all_keypoints[i];
		vector<Point2f> corners = all_corners[i];
		Scalar color = colors[i];

		// draw keypoints
		for (int i=0; i<keypoints.size(); ++i)
			circle(result, keypoints[i], 3, color, -1, 8, 0 );

		// draw rectangle around matched object
		for (int i=0; i<corners.size()-1; ++i)
			line(result, corners[i], corners[i+1], color, 4);
		line(result, corners[0], corners[corners.size()-1], color, 4);
	}
}

/**
 * For every object recompute keypoints and corners based on the optical flow between to consecutive frames.
 * @param from_frame The first image frame
 * @param to_frame The next image frame
 * @param all_keypoints A vector of keypoints coordinates for every object, this will also store the output coordinates
 * @param all_corners A vector of corners coordinates for every object, this will also store the output coordinates
 */
void trackObjects(Mat from_frame, Mat to_frame, vector<vector<Point2f>> &all_keypoints, vector<vector<Point2f>> &all_corners)
{
	vector<Point2f> new_keypoints, new_corners;

	// for every object
	for (int i=0; i<all_keypoints.size(); i++)
	{
		vector<uchar> status;
		vector<float> errors;
		calcOpticalFlowPyrLK(from_frame, to_frame, all_keypoints[i], new_keypoints, status, errors, Size(7,7), 3);

		// estimate translation and rotation from the optical flow and reproject the corners accordingly
		Mat H = findHomography(all_keypoints[i], new_keypoints, CV_RANSAC, RANSAC_REPROJ_THRESHOLD, status, 300);
		perspectiveTransform(all_corners[i], new_corners, H);

		// remove the features whose flow has been lost
		vector<Point2f> good_new_points;
		for (int j=0; j<new_keypoints.size(); j++)
			if (status[j] == 1)
				good_new_points.push_back(new_keypoints[j]);

		all_keypoints[i] = good_new_points;
		all_corners[i] = new_corners;
	}
}