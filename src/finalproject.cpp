#include "finalproject.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

const string path = "data/final_project/*.jpg";
const String winname = "Detection";
CascadeClassifier tree_cascade;

struct FilterParams
{
    Mat image;
    Range hue_range;
};

void detect_and_display( Mat frame_gray, Mat frame_orig )
{
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> trees;
    Mat result = frame_orig.clone();

    tree_cascade.detectMultiScale( frame_gray, trees, 1.05, 10, 0, Size(100,100));
    // groupRectangles(trees, 1, 0.85);
    for ( size_t i = 0; i < trees.size(); i++ )
    {
        Point pt1(trees[i].x, trees[i].y);
        Point pt2(trees[i].x + trees[i].width, trees[i].y + trees[i].height);
        rectangle(frame_gray, pt1, pt2, Scalar(128,0,0), 2);
        rectangle(result, pt1, pt2, Scalar(128,0,0), 2);
    }

    imshow( winname, frame_gray );
    imshow( "result", result );
}

static void on_trackbar_change(int, void *params)
{
    FilterParams *filter_params = static_cast<FilterParams*>(params);

    Mat image_HSV, mask, masked_image, image_gray;

    Mat image = filter_params->image;
    // bilateralFilter(filter_params->image, image, 3, 50,50);

    cvtColor(image, image_HSV, COLOR_BGR2HSV);
    Scalar lower_bound(filter_params->hue_range.start-1, 0, 0);
    Scalar upper_bound(filter_params->hue_range.end, 255, 255);
    inRange(image_HSV, lower_bound, upper_bound, mask);

    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    bitwise_not(mask, mask);
    bitwise_or(image_gray, mask, masked_image);

    detect_and_display(masked_image, filter_params->image);

    // cvtColor( filter_params->image, image_gray, COLOR_BGR2GRAY );
    // detect_and_display(image_gray, filter_params->image);
}

void main_finalproject() 
{

    if( !tree_cascade.load( "data/final_project/cascade.xml" ) )
    {
        cout << "Error loading cascade\n";
        return;
    }

    // load images
    vector<String> images_paths;
    glob(path, images_paths);

    namedWindow(winname);


    FilterParams params;
    params.hue_range.start = 0;
    params.hue_range.end = 128;
    createTrackbar("hue lower bound", winname, &params.hue_range.start, 128, on_trackbar_change, (void*)&params);
    createTrackbar("hue upper bound", winname, &params.hue_range.end, 129, on_trackbar_change, (void*)&params);

    for (String imgpath: images_paths) 
    {
        cout << imgpath << endl;
        Mat image = imread(imgpath);
        resize(image, image, Size(400,500));
        params.image = image;
        on_trackbar_change(0, &params);
        waitKey(0);
    }
}