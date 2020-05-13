#include "homework_5.h"

#include "panoramic_utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

const string path = "data/lab5/data/*.bmp";

void main_homework_5() {

    // load images
    vector<String> images_paths;
    glob(path, images_paths);
    vector<Mat> images;
    for (String imgpath: images_paths) {
        cout << imgpath << endl;
        images.push_back(imread(imgpath));
    }

    // cylindrical projection
    vector<Mat> cylindrical_proj_images;
    for (Mat image: images) {
        Mat projected = PanoramicUtils::cylindricalProj(image, 33.0);
        cylindrical_proj_images.push_back(projected);
    }

    // extract features using SIFT feature detector
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    Feature2D feature_detector = *sift;

    // detect keypoints
    vector<vector<KeyPoint>> keypoints;
    feature_detector.detect(images, keypoints); // detect keypoints

    // compute descriptors
    vector<vector<Mat>> descriptors;
    // feature_detector.compute(images, keypoints, descriptors); // compute descriptors

    // matching descriptors
    Ptr<BFMatcher> bruteforce_matcher = BFMatcher::create();
    BFMatcher matcher = *bruteforce_matcher;
    vector<vector<DMatch>> images_matches;
    for (int i=0; i<images.size()-1; i++) {
        vector<DMatch> matches;
        matcher.match(descriptors[i+1], matches);
        images_matches.push_back(matches);
    }

    for (int i=0; i<images.size()-1; i++) {
        vector<Point2f> points_from;
        vector<Point2f> points_to;
        vector<DMatch> matches = images_matches[i];
        for(int i = 0; i < images_matches[i].size(); ++i)
        {
            // extract points from keypoints based on matches
            points_from.push_back(keypoints[i].at(matches.at(i).queryIdx).pt);
            points_to.push_back(keypoints[i+1].at(matches.at(i).trainIdx).pt);
        }
        // compute homography using RANSAC
        Mat mask;
        // Mat H = findHomography(points_from, points_to, cv::RANSAC, ransacThreshold, mask);
    }
    

    waitKey(0);
}