#include "homework_5.h"

#include "panoramic_utils.h"
#include "panoramic_image.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

// const string path = "data/lab5/data/*.bmp";
// const float FOV = 66.0;

// const string path = "data/lab5/dolomites/*.png";
// const float FOV = 54.0;

const string path = "data/lab5/kitchen/*.bmp";
const float FOV = 66.0;

// const string path = "data/lab5/dataset_lab_19_automatic/*.png";
// const float FOV = 66.0;

// const string path = "data/lab5/dataset_lab_19_manual/*.png";
// const float FOV = 66.0;

void main_homework_5() {

    // load images
    vector<String> images_paths;
    glob(path, images_paths);
    vector<Mat> images;
    for (String imgpath: images_paths) {
        cout << imgpath << endl;
        images.push_back(imread(imgpath));
    }

    Mat result;
    PanoramicImage::stitchImages(images, FOV, result);
    imshow("result", result);

    waitKey(0);
}