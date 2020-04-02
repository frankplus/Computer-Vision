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


    waitKey(0);
}

