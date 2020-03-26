#include "homework_1.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#define NEIGHBORHOOD_X 9
#define NEIGHBORHOOD_Y 9
#define THRESHOLD 7

using namespace std;
using namespace cv;

void main_homework_1() {
    Mat input_img = imread("data/robocup.jpg");

    resize(input_img, input_img, Size(input_img.cols / 2.0, input_img.rows / 2.0));
    imshow("img", input_img);
    setMouseCallback("img", onMouse, (void*)&input_img);

    waitKey(0);
}

void onMouse( int event, int x, int y, int f, void *userdata ) {
    if ( event == EVENT_LBUTTONDOWN ) {
        Mat *image = (Mat*) userdata;
        Mat image_out = image->clone();

        Mat image_hsv;
        cvtColor(*image, image_hsv, COLOR_RGB2HSV);
        GaussianBlur(image_hsv, image_hsv, Size(5, 5), 0);

        if (y + NEIGHBORHOOD_Y > image_hsv.rows
            || x + NEIGHBORHOOD_X > image_hsv.cols)
            return;

        Rect rect(x, y, NEIGHBORHOOD_X, NEIGHBORHOOD_Y);
        Scalar mean = cv::mean(image_hsv(rect));

        for (int i = 0; i < image_hsv.rows; ++i)
            for (int j = 0; j < image_hsv.cols; ++j) {
                if ( abs(image_hsv.at<Vec3b> (i, j)[0] - mean[0]) < THRESHOLD) {

                    image_out.at<Vec3b> (i, j) = Vec3b(92,37,201);

                }
            }

        imshow("final_result", image_out);
    }
}