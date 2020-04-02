#include "homework_2.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <filesystem>
#include <vector>

#include "utils.h"

using namespace std;
using namespace cv;

void main_homework_2() {

    string path = "data/lab2/checkerboard_images/";
    vector<Mat> images;
    for (const auto & entry : filesystem::directory_iterator(path))
        images.push_back(imread(entry.path()));

    show_collage(images);

    Size patternsize(6,5); 
    const float square_width = 30.0f;
    const float square_height = 30.0f;
    vector<Point3f> corners3d;

    // compute 3D coordinates of the corners (in the chessboard reference system)
    for (int i=0; i<patternsize.height; i++)
        for (int j=0; j<patternsize.width; j++) {
            Point3f point = Point3f(j*square_width, i*square_height, 0);
            corners3d.push_back(point);
        }

    vector<vector<Point3f>> vector_corners3d;
    vector<vector<Point2f>> vector_corners2d; 

    for (const auto & image : images) {
        // find chessboard corners on the image
        vector<Point2f> corners2d;
        bool patternfound = findChessboardCorners(image, patternsize, corners2d,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (patternfound) {
            vector_corners2d.push_back(corners2d);
            vector_corners3d.push_back(corners3d);
        }
    }

    // showing an image with detected corners for debugging purposes
    int image_index = 0;
    drawChessboardCorners(images[image_index], patternsize, Mat(vector_corners2d[image_index]), true);
    imshow("image",images[image_index]);

    // performing camera calibration
    cv::Mat camera_matrix,dist_coeffs,R,T;
    Size img_size = Size(images[0].rows, images[0].cols);
    calibrateCamera(vector_corners3d, vector_corners2d, img_size, camera_matrix, dist_coeffs, R, T);

    cout << "Camera matrix : " << camera_matrix << std::endl;
    cout << "Distortion coefficients : " << dist_coeffs << std::endl;
    cout << "Rotation vector : " << R << std::endl;
    cout << "Translation vector : " << T << std::endl;

    // compute mean reprojection error
    vector<Point2f> reprojected_points;
    for (int image=0; image<images.size(); image++) {
        vector<Point2f> extracted_corners = vector_corners2d[image];
        projectPoints(corners3d, R.row(image), T.row(image), camera_matrix, dist_coeffs, reprojected_points);
        double sum_errors = 0;
        for (int corner=0; corner<corners3d.size(); corner++) {
            double distance = norm(reprojected_points[corner] - extracted_corners[corner]);
            sum_errors += distance;
        }
        double mean_errors = sum_errors / corners3d.size();
        cout << "Image " << image << " mean error: " << mean_errors << endl;
    }

    waitKey(0);
}

