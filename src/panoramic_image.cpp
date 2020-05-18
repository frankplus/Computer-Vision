#include "panoramic_image.h"

#include "panoramic_utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

const float DISTANCE_RATIO_THRESHOLD = 3.0;
const double RANSAC_REPROJ_THRESHOLD = 3.0;

/**
 * Stitch images togegher to make a panoramic image
 * @param images The input images to stitch togegher
 * @param FOV The field of view of the camera used to take the images
 * @param result The destination image in which to store the panoramic image.
 */
void PanoramicImage::stitchImages(vector<Mat> &images, float FOV, Mat &result) {
    vector<Mat> cylindrical_projected_images;
    cylindricalProjection(images, FOV, cylindrical_projected_images);

    vector<vector<DMatch>> images_matches;
    vector<vector<KeyPoint>> keypoints;
    detectAndMatchKeypoints(cylindrical_projected_images, keypoints, images_matches);

    vector<Point2f> translations;
    computeTranslations(keypoints, images_matches, translations);
    joinImages(cylindrical_projected_images, translations, result);
}

/**
 * Make cylindrical projection on the given images.
 * @param images The vector of input images
 * @param FOV The field of view of the camera used to take the images
 * @param out_images The destination vector of images
 */
void PanoramicImage::cylindricalProjection(vector<Mat> &images, float FOV, vector<Mat> &out_images) {
    for (Mat image: images) {
        Mat projected = PanoramicUtils::cylindricalProj(image, FOV/2);
        out_images.push_back(projected);
    }
}

/**
 * Extract features detecting keypoints and computing descriptors by using SIFT feature detector. 
 * Then match the keypoints between the pairs of consecutive images.
 * @param images The vector of input images
 * @param keypoints An output vector of keypoints for every given image
 * @param images_matches An output vector of keypoints matches for every pair of consecutive images with length one less the number of input images.
 */
void PanoramicImage::detectAndMatchKeypoints(vector<Mat> &images, vector<vector<KeyPoint>> &keypoints, vector<vector<DMatch>> &images_matches) {
    // extract features using SIFT feature detector
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();

    // detect keypoints
    sift->detect(images, keypoints); 

    // compute descriptors
    vector<Mat> descriptors;
    sift->compute(images, keypoints, descriptors);

    // matching descriptors
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
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
            if (match.distance < DISTANCE_RATIO_THRESHOLD * min_distance)
                refined_matches.push_back(match);
        }

        images_matches.push_back(refined_matches);
    }
}

/**
 * Compute the translations vectors for each image with respect to the first leftmost image.
 * @param keypoints A vector of keypoints for each input image 
 * @param matches A vector of keypoints matches for every pair of consecutive images with length one less the number of images.
 * @param translations The destination vector of translation vectors for each image
 */
void PanoramicImage::computeTranslations(vector<vector<KeyPoint>> &keypoints, vector<vector<DMatch>> &matches, vector<Point2f> &translations) {
    translations.push_back(Point2f(0,0));
    Point2f acc_translation = Point2f(0,0);
    for (int i=0; i<keypoints.size()-1; i++) {
        acc_translation += computeTranslation(keypoints[i], keypoints[i+1], matches[i]);
        translations.push_back(acc_translation);
    }
}

/**
 * Compute the relative translation vector between a pair of images.
 * @param keypoints1 and keypoints2 The keypoints of the two images respectively
 * @param matches The input vector of keypoint matches between the pair of images
 * @return The translation vector.
 */
Point2f PanoramicImage::computeTranslation(vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches) {
    // extract points from keypoints based on matches
    vector<Point2f> points_from;
    vector<Point2f> points_to;
    for (int j=0; j < matches.size(); ++j) {
        points_from.push_back(keypoints1.at(matches.at(j).queryIdx).pt);
        points_to.push_back(keypoints2.at(matches.at(j).trainIdx).pt);
    }

    // use find homography to find outliers
    Mat mask;
    findHomography(points_from, points_to, CV_RANSAC, RANSAC_REPROJ_THRESHOLD, mask);

    // compute avarage translation between the matched keypoints
    Point2f avg_translation;
    int size = 0;
    for (int j=0; j<points_from.size(); j++) {
        if (mask.at<uchar>(j, 0)) { // discard outliers
            avg_translation += (points_from[j] - points_to[j]);
            size++;
        }
    }
    avg_translation /= (float) size;

    return avg_translation;
}

/**
 * Merge the images using the precalculated set of translations producing the resulting panoramic image.
 * @param images The input vector of images.
 * @param translations The vector of translations vectors for each given image
 * @param result The destination image in which to store the panoramic image.
 */
void PanoramicImage::joinImages(vector<Mat> &images, vector<Point2f> &translations, Mat &result) {
    // calculate output image size
    float min_x, max_x, min_y, max_y;
    min_x = max_x = translations[0].x;
    min_y = max_y = translations[0].y;
    for (Point2f translation: translations) {
        min_x = min(min_x, translation.x);
        max_x = max(max_x, translation.x);
        min_y = min(min_y, translation.y);
        max_y = max(max_y, translation.y);
    }
    int len_images = images.size();
    int size_x = max_x - min_x + images[len_images-1].cols;
    int size_y = max_y - min_y + images[len_images-1].rows;

    // equalize histogram of images (I found a better result without equalization)
    // for (Mat image : images) {
    //     equalizeHist(image, image);
    // }

    // produce final image
    result = Mat(size_y, size_x, CV_8UC1, Scalar(128,128,128));
    for (int i=0; i<len_images; i++) {
        Rect roi = Rect(translations[i].x - min_x, translations[i].y - min_y, images[i].cols, images[i].rows);
        images[i].copyTo(result(roi));
    }
}