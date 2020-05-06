#include "homework_4.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

const char* canny_window_name = "homework 4 - canny edge detection";
const char* hough_window_name = "homework 4 - hough lines detection";

struct HoughParams {
    Mat detected_edges;
    Mat orig_img;
    double rho_resolution = 1.0;
    double theta_resolution = 0.05;
    int lines_threshold = 120;
    int circle_acc_threshold = 30;
    int circle_max_radius = 10;
};

struct CannyParams {
    Mat src;
    Mat detected_edges;
    int low_threshold = 283;
    int threshold_ratio = 3;
    int kernel_size = 3;
    HoughParams *hough_params;
};

static void houghTransform(int, void *params) {
    HoughParams *hough_params = static_cast<HoughParams*>(params); 

    vector<Vec2f> lines;
    Mat detected_edges = hough_params->detected_edges;
    HoughLines(detected_edges, lines, hough_params->rho_resolution, 
                hough_params->theta_resolution, hough_params->lines_threshold, 0, 0 ); 
    
    vector<Vec3f> circles;
    HoughCircles(detected_edges, circles, HOUGH_GRADIENT, 2, detected_edges.rows/4, 
                    200, hough_params->circle_acc_threshold, 2, hough_params->circle_max_radius );

    // Draw the lines and circles and show result
    Mat dst = hough_params->orig_img.clone();
    for ( size_t i = 0; i < lines.size(); i++ ) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( dst, pt1, pt2, Scalar(0,0,255), 2, LINE_AA);
    }

    for( size_t i = 0; i < circles.size(); i++ )
    {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         circle( dst, center, radius, Scalar(0,255,0), -1, 8, 0 );
    }

    imshow( hough_window_name, dst );
}

static void cannyThreshold(int, void *params) {
    CannyParams *canny_params = static_cast<CannyParams*>(params); 
    Canny( canny_params->src, canny_params->detected_edges, canny_params->low_threshold, 
            canny_params->low_threshold * canny_params->threshold_ratio, canny_params->kernel_size );

    // show result
    imshow( canny_window_name, canny_params->detected_edges );

    // run hough transform
    canny_params->hough_params->detected_edges = canny_params->detected_edges;
    houghTransform(0, canny_params->hough_params);
}

void main_homework_4() {

    Mat input_img = imread("data/lab4/input.png");
    namedWindow( canny_window_name );
    namedWindow( hough_window_name );

    Mat src_gray;
    cvtColor( input_img, src_gray, COLOR_BGR2GRAY );

    HoughParams hough_params;
    hough_params.orig_img = input_img;

    CannyParams canny_params;
    canny_params.src = src_gray;
    canny_params.hough_params = &hough_params;

    const int max_low_threshold = 400;
    const int max_hough_lines_threshold = 300;
    const int max_hough_circle_threshold = 50;

    createTrackbar( "Min canny threshold:", canny_window_name, &canny_params.low_threshold, 
                    max_low_threshold, cannyThreshold, &canny_params );

    createTrackbar( "hough lines threshold:", hough_window_name, &hough_params.lines_threshold, 
                    max_hough_lines_threshold, houghTransform, &hough_params );

    createTrackbar( "hough circle threshold:", hough_window_name, &hough_params.circle_acc_threshold, 
                    max_hough_circle_threshold, houghTransform, &hough_params );

    cannyThreshold(0, &canny_params);
    houghTransform(0, &hough_params);

    waitKey(0);
}