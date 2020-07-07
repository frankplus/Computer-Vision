#include "finalproject.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

const String cascade_path = "data/final_project/cascade.xml";
const string images_path = "data/final_project/*.jpg";
const String detection_winname = "Detection";
const String result_winname = "Result";


static void on_trackbar_change(int, void *params)
{
    DetectionParams *filter_params = static_cast<DetectionParams*>(params);

    preprocess_image(filter_params->input_image, filter_params->filtered_image, 
                    filter_params->filter_sigma, filter_params->hue_range);

    detect_and_display(filter_params);
}


void main_finalproject() 
{
    namedWindow(detection_winname);
    namedWindow(result_winname);

    DetectionParams params;
    if( !params.tree_cascade.load( cascade_path ) )
    {
        cout << "Error loading cascade\n";
        return;
    }

    createTrackbar("Hue lower bound", detection_winname, &params.hue_range.start, 128, on_trackbar_change, (void*)&params);
    createTrackbar("Hue upper bound", detection_winname, &params.hue_range.end, 128, on_trackbar_change, (void*)&params);
    createTrackbar("Filter sigma range/space", detection_winname, &params.filter_sigma, 200, on_trackbar_change, (void*)&params);
    createTrackbar("Detection min neighbors", detection_winname, &params.min_neighbors, 40, on_trackbar_change, (void*)&params);
    createTrackbar("Detection min size", detection_winname, &params.min_size, 100, on_trackbar_change, (void*)&params);
    createTrackbar("Group threshold", detection_winname, &params.group_thresh, 20, on_trackbar_change, (void*)&params);
    createTrackbar("Group eps", detection_winname, &params.group_eps, 40, on_trackbar_change, (void*)&params);

    // load images
    vector<String> images_paths;
    glob(images_path, images_paths);

    for (String imgpath: images_paths) 
    {
        cout << imgpath << endl;
        Mat image = imread(imgpath);
        resize(image, image, Size(500, 500));
        params.input_image = image;
        on_trackbar_change(0, &params);
        waitKey(0);
    }
}

void detect_and_display( DetectionParams *params )
{
    Mat result = params->input_image.clone();

    vector<Rect> trees;
    Size min_size{params->min_size, params->min_size};
    params->tree_cascade.detectMultiScale( params->filtered_image, trees, 1.05, params->min_neighbors, 0, min_size);
    groupRectangles(trees, params->group_thresh, params->group_eps/100.0);

    for ( size_t i = 0; i < trees.size(); i++ )
    {
        Point pt1(trees[i].x, trees[i].y);
        Point pt2(trees[i].x + trees[i].width, trees[i].y + trees[i].height);
        rectangle(params->filtered_image, pt1, pt2, Scalar(128,0,0), 2);
        rectangle(result, pt1, pt2, Scalar(128,0,0), 2);
    }

    imshow( detection_winname, params->filtered_image );
    imshow( result_winname, result );
}

void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range)
{
    Mat filtered;
    bilateralFilter(input, filtered, 11, sigma, sigma);

    Mat image_HSV;
    cvtColor(filtered, image_HSV, COLOR_BGR2HSV);

    Mat mask;
    Scalar lower_bound(hue_range.start, 0, 0);
    Scalar upper_bound(hue_range.end, 255, 255);
    inRange(image_HSV, lower_bound, upper_bound, mask);

    cvtColor( filtered, result, COLOR_BGR2GRAY );
    result.setTo(Scalar(128,128,128), mask);
}