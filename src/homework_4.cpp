#include "homework_4.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

const char* canny_window_name = "homework 4 - canny edge detection";
const char* hough_window_name = "homework 4 - hough lines detection";
const char* result_window_name = "homework 4 - result";

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

struct Line {
    Point a;
    Point b;
};

/**
 * Convert lines in polar coordinates which is returned by hough transform into pairs 
 * of points in x/y coordinates which delimit the lines.
 * @param input_lines The vector of input lines, each line is a Vec2f containing rho and theta
 * @param output_lines The output vector of lines, each line is a Line object containing a pair of Point.
 */
void polar_lines_to_cartesian(vector<Vec2f> &input_lines, vector<Line> &output_lines) {
    for ( size_t i = 0; i < input_lines.size(); i++ ) {
        float rho = input_lines[i][0], theta = input_lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        Line line;
        line.a.x = cvRound(x0 + 1000*(-b));
        line.a.y = cvRound(y0 + 1000*(a));
        line.b.x = cvRound(x0 - 1000*(-b));
        line.b.y = cvRound(y0 - 1000*(a));
        output_lines.push_back(line);
    }
}

/**
 * Finds the intersection of two lines, or returns false.
 * The lines are defined by (o1, p1) and (o2, p2).
 * The intersection point will be stored in Point2f &r
 */
bool lines_intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                      Point2f &r) {
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

/**
 * Draw a circle on given image
 * @param img The image onto which the circle is to be drawn
 * @param circle_to_draw The circle described as a Vec3f contaning center coordinates and circle radius.
 */
void draw_circle(Mat img, Vec3f circle_to_draw) {
    Point center(cvRound(circle_to_draw[0]), cvRound(circle_to_draw[1]));
    int radius = cvRound(circle_to_draw[2]);
    circle( img, center, radius, Scalar(0,255,0), -1, 8, 0 );
}

/**
 * Fill the lower triangle between two given intersecting lines and fill given circle, then show final result.
 * @param img The source image
 * @param line1 and line2 The pair of lines delimiting the triangle
 * @param circle_to_draw The circle to draw described as a Vec3f contaning center coordinates and circle radius.
 */
void show_result(Mat img, Line line1, Line line2, Vec3f circle_to_draw) {

    Point2f pt_intersection, pt1, pt2;

    // find lower triangle vertices
    if ( lines_intersection(line1.a, line1.b, line2.a, line2.b, pt_intersection) ) {
        pt1 = line1.a.y > line1.b.y ? line1.a : line1.b;
        pt2 = line2.a.y > line2.b.y ? line2.a : line2.b;
        Point triangle[] = {pt1, pt2, pt_intersection};
        const Point* polygons[] = {triangle};
        int npt[] = { 3 };

        // draw triangle
        fillPoly(img, polygons, npt, 1, Scalar(0,0,255), LINE_8);
    }

    // draw circle
    draw_circle(img, circle_to_draw);

    // show result
    imshow( result_window_name, img );
}

/**
 * Callback function which runs hough transform to detect lines and circles in an image, then it finally 
 * show the final image coloring the space between two detected lines and a detected circle.
 * @param params An HoughParams object contaning the source images, and parameters
 */
static void hough_transform(int, void *params) {
    HoughParams *hough_params = static_cast<HoughParams*>(params); 

    // detect lines
    vector<Vec2f> lines;
    Mat detected_edges = hough_params->detected_edges;
    HoughLines(detected_edges, lines, hough_params->rho_resolution, 
                hough_params->theta_resolution, hough_params->lines_threshold, 0, 0 ); 
    
    // draw lines
    vector<Line> lines_cartesian;
    polar_lines_to_cartesian(lines, lines_cartesian);
    Mat dst = hough_params->orig_img.clone();
    for ( Line detected_line : lines_cartesian ) {
        line( dst, detected_line.a, detected_line.b, Scalar(0,0,255), 2, LINE_AA);
    }

    // detect circles
    vector<Vec3f> circles;
    HoughCircles(detected_edges, circles, HOUGH_GRADIENT, 2, detected_edges.rows/4, 
                    200, hough_params->circle_acc_threshold, 2, hough_params->circle_max_radius );

    // draw circles
    for (Vec3f circle : circles) {
        draw_circle(dst, circle);
    } 

    // show detected lines and circles
    imshow( hough_window_name, dst );

    // show final result
    if (lines_cartesian.size() >= 2 && circles.size() > 0) {
        show_result(hough_params->orig_img.clone(), lines_cartesian[0], lines_cartesian[1], circles[0]);
    }
}

/**
 * Callback function which runs canny edge detector on an image to detect edges and show the result.
 * Then it finally calls the hough_transform callback on the detected edges to detect lines and circles.
 * @param params An CannyParams object contaning the source images, and parameters
 */
static void canny_threshold(int, void *params) {
    CannyParams *canny_params = static_cast<CannyParams*>(params); 
    Canny( canny_params->src, canny_params->detected_edges, canny_params->low_threshold, 
            canny_params->low_threshold * canny_params->threshold_ratio, canny_params->kernel_size );

    // show result
    imshow( canny_window_name, canny_params->detected_edges );

    // run hough transform
    canny_params->hough_params->detected_edges = canny_params->detected_edges;
    hough_transform(0, canny_params->hough_params);
}

void main_homework_4() {

    // loads an image
    string path;
    cout << "Type input image path (empty input loads 'data/lab4/input.png'): ";
    getline(cin, path);
    if (path.empty()) {
        path = "data/lab4/input.png";
    }
    Mat input_img = imread(path);

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
                    max_low_threshold, canny_threshold, &canny_params );

    createTrackbar( "hough lines threshold:", hough_window_name, &hough_params.lines_threshold, 
                    max_hough_lines_threshold, hough_transform, &hough_params );

    createTrackbar( "hough circle threshold:", hough_window_name, &hough_params.circle_acc_threshold, 
                    max_hough_circle_threshold, hough_transform, &hough_params );

    canny_threshold(0, &canny_params);
    hough_transform(0, &hough_params);

    waitKey(0);
}