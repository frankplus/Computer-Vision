#include "utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

void show_collage(const std::vector<cv::Mat> &images, string winname) {
    Mat collage_img = Mat(480,640,CV_8UC3);
    int rows = sqrt(images.size()) + 1;
    tile(images, collage_img, rows, rows);
    imshow(winname,collage_img);
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

void show_histogram(std::vector<cv::Mat> &hists, std::string winname) {
	// Min/Max computation
	double hmax[3] = {0, 0, 0};
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = {"blue", "green", "red"};
	cv::Scalar colors[3] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
							cv::Scalar(0, 0, 255)};

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++) {
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++) {
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0);
		}

		// cv::imshow(hists.size() == 1 ? "value" : winname + " " + wname[i], canvas[i]);
	}
    show_collage(canvas, winname);
}