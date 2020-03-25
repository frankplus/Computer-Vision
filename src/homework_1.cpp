#include "homework_1.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#define NEIGHBORHOOD_X 9
#define NEIGHBORHOOD_Y 9
#define MAX_R_CHANNEL 70
#define MAX_G_CHANNEL 100
#define MAX_B_CHANNEL 70

using namespace std;
using namespace cv;

void main_homework_1() {
    Mat input_img = imread("assets/robocup.jpg");

    resize(input_img, input_img, Size(input_img.cols / 2.0, input_img.rows / 2.0));
    imshow("img", input_img);
    setMouseCallback("img", onMouse, (void*)&input_img);

    waitKey(0);
}

void onMouse( int event, int x, int y, int f, void *userdata ) {
    if ( event == EVENT_LBUTTONDOWN ) {
        Mat *image = (Mat*) userdata;
        Mat image_out = image->clone();

        if (y + NEIGHBORHOOD_Y > image_out.rows
            || x + NEIGHBORHOOD_X > image_out.cols)
            return;

        Rect rect(x, y, NEIGHBORHOOD_X, NEIGHBORHOOD_Y);
        Scalar mean = cv::mean(image_out(rect));
        cout << "Mean: " << mean << endl;
    }
}