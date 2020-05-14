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
const float distance_ratio_threshold = 3.0;

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

    // detect keypoints
    vector<vector<KeyPoint>> keypoints;
    sift->detect(images, keypoints); // detect keypoints

    // compute descriptors
    vector<Mat> descriptors;
    sift->compute(images, keypoints, descriptors); // compute descriptors

    // matching descriptors
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
    vector<vector<DMatch>> images_matches;
    for (int i=0; i<images.size()-1; i++) {
        vector<DMatch> matches, refined_matches;
        matcher->match(descriptors[i], descriptors[i+1], matches);
        
        // find min distance
        float min_distance = matches[0].distance;
        for (DMatch match: matches) {
            min_distance = min(min_distance, match.distance);
        }

        // refine by selecting the matches with distance less than ratio * min_distance
        for (DMatch match: matches) {
            if (match.distance < distance_ratio_threshold * min_distance)
                refined_matches.push_back(match);
        }

        images_matches.push_back(refined_matches);
    }

    // find translation between consecutive images
    vector<Point2f> translations;
    for (int i=0; i<images.size()-1; i++) {
        // extract points from keypoints based on matches
        vector<Point2f> points_from;
        vector<Point2f> points_to;
        vector<DMatch> matches = images_matches[i];
        for (int j=0; j < matches.size(); ++j) {
            points_from.push_back(keypoints[i].at(matches.at(j).queryIdx).pt);
            points_to.push_back(keypoints[i+1].at(matches.at(j).trainIdx).pt);
        }

        // use find homography to find outliers
        Mat mask;
        findHomography(points_from, points_to, CV_RANSAC, 3, mask);

        // compute avarage translation between the matched keypoints
        Point2f avg_translation;
        int size = 0;
        for (int j=0; j<points_from.size(); j++) {
            if (mask.at<uchar>(j, 0)) {
                avg_translation += (points_from[j] - points_to[j]);
                size++;
            }
        }
        avg_translation /= (float) size;
        translations.push_back(avg_translation);
    }
    translations.push_back(Point2f(0,0));

    // create final panoramic image
    Mat result = Mat(450, 3750, CV_8UC3, Scalar(128,128,128));
    Point2f acc_translation = Point2f(0,0);
    for (int i=0; i<images.size(); i++) {
        Range range_x = Range(acc_translation.x, acc_translation.x + images[i].cols);
        Range range_y = Range(acc_translation.y, acc_translation.y + images[i].rows)+10;
        cout << "range x: " << range_x << " range_y: " << range_y << endl;
        Mat submat = result(range_y, range_x);
        images[i].copyTo(submat);
        // addWeighted( images[i], 0.5, submat, 0.5, 0.0, submat);

        acc_translation += translations[i];
    }
    imshow("res", result);

    waitKey(0);
}