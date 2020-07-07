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

    print_parameters(filter_params);

    preprocess_image(filter_params->input_image, filter_params->filtered_image, 
                    filter_params->filter_sigma, filter_params->hue_range, filter_params->value_range);

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

    createTrackbar("Hue lower bound", detection_winname, &params.hue_range.start, 256, on_trackbar_change, (void*)&params);
    createTrackbar("Hue upper bound", detection_winname, &params.hue_range.end, 256, on_trackbar_change, (void*)&params);
    createTrackbar("Value lower bound", detection_winname, &params.value_range.start, 256, on_trackbar_change, (void*)&params);
    createTrackbar("Value upper bound", detection_winname, &params.value_range.end, 256, on_trackbar_change, (void*)&params);
    createTrackbar("Filter sigma range/space", detection_winname, &params.filter_sigma, 200, on_trackbar_change, (void*)&params);
    createTrackbar("Detection min neighbors", detection_winname, &params.min_neighbors, 40, on_trackbar_change, (void*)&params);
    createTrackbar("Detection min size", detection_winname, &params.min_size, 100, on_trackbar_change, (void*)&params);

    // load images
    vector<String> images_paths;
    glob(images_path, images_paths);

    while(true)
    {
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
}

void detect_and_display( DetectionParams *params )
{
    Mat result = params->input_image.clone();

    vector<Rect> trees;
    Size min_size{params->min_size, params->min_size};
    params->tree_cascade.detectMultiScale( params->filtered_image, trees, 1.05, params->min_neighbors, 0, min_size);

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

void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range, Range value_range)
{
    vector<Mat> img_channels;
    split(input, img_channels);

    equalizeHist(img_channels[0], img_channels[0]);
    equalizeHist(img_channels[1], img_channels[1]);
    equalizeHist(img_channels[2], img_channels[2]);

    Mat equalized_img;
    merge(img_channels, equalized_img);

    Mat filtered;
    bilateralFilter(equalized_img, filtered, 11, sigma, sigma);

    Mat image_HSV;
    cvtColor(filtered, image_HSV, COLOR_BGR2HSV);

    Mat mask;
    Scalar lower_bound(hue_range.start, 0, value_range.start);
    Scalar upper_bound(hue_range.end, 255, value_range.end);
    inRange(image_HSV, lower_bound, upper_bound, mask);

    cvtColor( filtered, result, COLOR_BGR2GRAY );
    add(result, Scalar(40), result, mask);
}

void print_parameters(DetectionParams *filter_params)
{
    cout << "hue range: " << filter_params->hue_range << endl;
    cout << "value range: " << filter_params->value_range << endl;
    cout << "filter_sigma: " << filter_params->filter_sigma << endl;
    cout << "min_neighbors: " << filter_params->min_neighbors << endl;
    cout << "min_size: " << filter_params->min_size << endl;
    cout << endl;
}