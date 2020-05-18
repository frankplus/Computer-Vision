#ifndef __PANORAMICIMAGE__
#define __PANORAMICIMAGE__

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class PanoramicImage {
public: 
    /**
     * Stitch images togegher to make a panoramic image
     * @param images The input images to stitch togegher
     * @param FOV The field of view of the camera used to take the images
     * @param result The destination image in which to store the panoramic image.
     */
    static void stitchImages(vector<Mat> &images, float FOV, Mat &result);

    /**
     * Make cylindrical projection on the given images.
     * @param images The vector of input images
     * @param FOV The field of view of the camera used to take the images
     * @param out_images The destination vector of images
     */
    static void cylindricalProjection(vector<Mat> &images, float FOV, vector<Mat> &out_images);

    /**
     * Extract features detecting keypoints and computing descriptors by using SIFT feature detector. 
     * Then match the keypoints between the pairs of consecutive images.
     * @param images The vector of input images
     * @param keypoints An output vector of keypoints for every given image
     * @param images_matches An output vector of keypoints matches for every pair of consecutive images with length one less the number of input images.
     */
    static void detectAndMatchKeypoints(vector<Mat> &images, vector<vector<KeyPoint>> &keypoints, vector<vector<DMatch>> &matches);

    /**
     * Compute the translations vectors for each image with respect to the first leftmost image.
     * @param keypoints A vector of keypoints for each input image 
     * @param matches A vector of keypoints matches for every pair of consecutive images with length one less the number of images.
     * @param translations The destination vector of translation vectors for each image
     */
    static void computeTranslations(vector<vector<KeyPoint>> &keypoints, vector<vector<DMatch>> &images_matches, vector<Point2f> &translations);

    /**
     * Compute the relative translation vector between a pair of images.
     * @param keypoints1 and keypoints2 The keypoints of the two images respectively
     * @param matches The input vector of keypoint matches between the pair of images
     * @return The translation vector.
     */
    static Point2f computeTranslation(vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches);

    /**
     * Merge the images using the precalculated set of translations producing the resulting panoramic image.
     * @param images The input vector of images.
     * @param translations The vector of translations vectors for each given image
     * @param result The destination image in which to store the panoramic image.
     */
    static void joinImages(vector<Mat> &images, vector<Point2f> &translations, Mat &result);
};

#endif