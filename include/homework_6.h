#ifndef __HOMEWORK6__
#define __HOMEWORK6__

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

void main_homework_6();

/**
 * Locate and track objects showing keypoints and bounding rectangles at each frame.
 * @param template_images Vector of images of the objects to detect and track
 * @param video_capture The video to analyze and detect the given objects
 */
void locateAndTrack(vector<Mat> template_images, VideoCapture video_capture);

/**
 * Find image1 features inside image2 by detecting the keypoints in both images and matching them. 
 * Then return the matched keypoints found inside image2 and the corners of image1 projected onto image2.
 * @param image1 The source template image to search inside image2.
 * @param image2 The image inside which we search image1 features.
 * @param corner_points The output points in the image2 coordinates representing the projection of the image1 corners.
 * @param points The output points in the image2 coordinates representing the projection of the image1 keypoints.
 */
void detectAndMatchKeypoints(Mat image1, Mat image2, vector<Point2f> &corner_points, vector<Point2f> &points);

/**
 * Draw the keypoints and the bounding rectangles on the given image, each object with different colors.
 * @param result The image onto which we draw the keypoints and the rectangles.
 * @param keypoints A vector of keypoints coordinates for every object.
 * @param corners The coordinates of the corners of the rectangles in clockwise or counterclockwise order for each object.
 * @param color The drawing colors for each object
 */
void drawKeypointsAndRectangles(Mat result, vector<vector<Point2f>> &all_keypoints, vector<vector<Point2f>> &all_corners, vector<Scalar> colors);

/**
 * For every object recompute keypoints and corners based on the optical flow between to consecutive frames.
 * @param from_frame The first image frame
 * @param to_frame The next image frame
 * @param all_keypoints A vector of keypoints coordinates for every object, this will also store the output coordinates
 * @param all_corners A vector of corners coordinates for every object, this will also store the output coordinates
 */
void trackObjects(Mat from_frame, Mat to_frame, vector<vector<Point2f>> &all_keypoints, vector<vector<Point2f>> &all_corners);

#endif