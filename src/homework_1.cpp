#include "homework_1.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#define NEIGHBORHOOD_X 9
#define NEIGHBORHOOD_Y 9
#define MAX_R_CHANNEL 70
#define MAX_G_CHANNEL 70
#define MAX_B_CHANNEL 70

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

        if (y + NEIGHBORHOOD_Y > image_out.rows
            || x + NEIGHBORHOOD_X > image_out.cols)
            return;

        Rect rect(x, y, NEIGHBORHOOD_X, NEIGHBORHOOD_Y);
        Scalar mean = cv::mean(image_out(rect));

        for (int i = 0; i < image_out.rows; ++i)
            for (int j = 0; j < image_out.cols; ++j) {
                if ( 
                    (abs(image_out.at<Vec3b> (i, j)[0] - mean[0]) < MAX_B_CHANNEL)
                    && (abs(image_out.at<Vec3b> (i, j)[1] - mean[1]) < MAX_G_CHANNEL)
                    && (abs(image_out.at<Vec3b> (i, j)[2] - mean[2]) < MAX_R_CHANNEL) 
                ) {

                    image_out.at<Vec3b> (i, j) = Vec3b(92,37,201);

                }
            }

        imshow("final_result", image_out);
    }
}