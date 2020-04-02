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

    waitKey(0);
}

