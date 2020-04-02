#include "utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

void show_collage(const std::vector<cv::Mat> &images) {
    Mat collage_img = Mat(480,640,CV_8UC3);
    int rows = sqrt(images.size()) + 1;
    tile(images, collage_img, rows, rows);
    imshow("collage",collage_img);
}

void tile(const vector<Mat> &src, Mat &dst, int grid_x, int grid_y) {
    // patch size
    int width  = dst.cols/grid_x;
    int height = dst.rows/grid_y;
    // iterate through grid
    int k = 0;
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            if (k < src.size()) {
                Mat s = src[k++];
                resize(s,s,Size(width,height));
                s.copyTo(dst(Rect(j*width,i*height,width,height)));
            } else {
                break;
            }
        }
    }
}