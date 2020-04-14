#include "homework_2.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>
#include <numeric>

#include "utils.h"

using namespace std;
using namespace cv;

// #define USE_MY_DATASET

#ifdef USE_MY_DATASET
const Size patternsize(7,5); 
const float square_width = 0.03f;
const float square_height = 0.03f;
const string images_path = "data/lab2/my_checkerboard_images/*.jpg";
const string test_image_path = "data/lab2/my_test_image.jpg";
#else
const Size patternsize(6,5); 
const float square_width = 0.11f;
const float square_height = 0.11f;
const string images_path = "data/lab2/checkerboard_images/*.png";
const string test_image_path = "data/lab2/test_image.png";
#endif

void main_homework_2() {

    // find images
    vector<String> images;
    glob(images_path, images);

    // compute 3D coordinates of the corners (in the chessboard reference system)
    vector<Point3f> corners3d;
    for (int i=0; i<patternsize.height; i++)
        for (int j=0; j<patternsize.width; j++) {
            Point3f point = Point3f(j*square_width, i*square_height, 0);
            corners3d.push_back(point);
        }

    vector<vector<Point3f>> vector_corners3d;
    vector<vector<Point2f>> vector_corners2d; 

    Mat image, gray;
    vector<Point2f> corners2d;
    bool patternfound;

    for (String imgpath: images) {
        // load image
        cout << imgpath << endl;
        image = imread(imgpath);
        #ifdef USE_MY_DATASET
            resize(image, image, Size(image.cols / 2.0, image.rows / 2.0));
        #endif

        // find chessboard corners on the image
        patternfound = findChessboardCorners(image, patternsize, corners2d,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (patternfound) {
            // refine pixel coordinates
            cvtColor(image, gray, COLOR_BGR2GRAY);
            TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.001);
            cornerSubPix(gray, corners2d, Size(17,17), Size(-1,-1), criteria);
            
            vector_corners2d.push_back(corners2d);
            vector_corners3d.push_back(corners3d);
        } else {
            cout << "Pattern not found" << endl;
        }
    }

    // showing an image with detected corners for debugging purposes
    drawChessboardCorners(image, patternsize, corners2d, patternfound);
    imshow("Image",image);

    // performing camera calibration
    Mat camera_matrix, dist_coeffs, R, T;
    Size img_size = image.size();
    double error = calibrateCamera(vector_corners3d, vector_corners2d, img_size, camera_matrix, dist_coeffs, R, T);

    cout << "Camera matrix : " << camera_matrix << endl;
    cout << "Distortion coefficients : " << dist_coeffs << endl;
    cout << "Rotation vector : " << R << endl;
    cout << "Translation vector : " << T << endl;
    cout << "Calibrate camera error: " << error << endl;

    // compute mean reprojection error
    vector<Point2f> reprojected_points;
    vector<double> img_mean_errors;
    for (int image=0; image<images.size(); image++) {
        vector<Point2f> extracted_corners = vector_corners2d[image];
        projectPoints(corners3d, R.row(image), T.row(image), camera_matrix, dist_coeffs, reprojected_points);
        double sum_errors = 0;
        for (int corner=0; corner<corners3d.size(); corner++) {
            double distance = norm(reprojected_points[corner] - extracted_corners[corner]);
            sum_errors += distance;
        }
        double mean_error = sum_errors / corners3d.size();
        img_mean_errors.push_back(mean_error);
    }

    // find best and worst image
    auto bestworst_image = minmax_element(begin(img_mean_errors), end(img_mean_errors));
    int best_image_index = bestworst_image.first - begin(img_mean_errors);
    int worst_image_index = bestworst_image.second - begin(img_mean_errors);
    double best_error = *bestworst_image.first;
    double worst_error = *bestworst_image.second;
    double mean_error = accumulate(begin(img_mean_errors), end(img_mean_errors), 0.0) / images.size(); 

    cout << "best mean error: " << best_error << endl;
    cout << "worst mean error: " << worst_error << endl;
    cout << "mean of mean errors: " << mean_error << endl;

    Mat best_image = imread(images[best_image_index]);
    Mat worst_image = imread(images[worst_image_index]);
    imshow("Best image", best_image);
    imshow("Worst image", worst_image);

    // undistort and rectify a test image 
    Mat test_image = imread(test_image_path);
    #ifdef USE_MY_DATASET
        resize(test_image, test_image, Size(test_image.cols / 2.0, test_image.rows / 2.0));
    #endif
    img_size = test_image.size();
    Mat output_image, rect_mat, mapx, mapy;
    initUndistortRectifyMap(camera_matrix, dist_coeffs, rect_mat, camera_matrix, img_size, CV_32FC1, mapx, mapy);
    remap(test_image, output_image, mapx, mapy, INTER_LINEAR);

    imshow("Test image original", test_image);
    imshow("Test image undistorted", output_image);

    waitKey(0);
}

