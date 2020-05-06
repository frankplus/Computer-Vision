#include "homework_3.h"
#include "utils.h"
#include "filter.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

enum FilterType { MEDIAN, GAUSSIAN, BILATERAL };

struct FilterParams
{
    FilterType type;
    Filter *filter;
    string window_name;
    int filter_size;
    int sigma1;
    int sigma2;
};

void onTrackbarChange(int pos, void *userdata) {
    FilterParams *filter_params = static_cast<FilterParams*>(userdata);
    Filter *filter = filter_params->filter;

    switch (filter_params->type) {
        case MEDIAN: {
            filter->setSize(filter_params->filter_size);
            break;
        }
        case GAUSSIAN: {
            GaussianFilter *gaussian_filter = dynamic_cast<GaussianFilter*>(filter);
            gaussian_filter->setSize(filter_params->filter_size);
            gaussian_filter->setSigma(filter_params->sigma1);
            break;
        }
        case BILATERAL: {
            BilateralFilter *bilateral_filter = dynamic_cast<BilateralFilter*>(filter);
            bilateral_filter->setSigmaRange(filter_params->sigma1);
            bilateral_filter->setSigmaSpace(filter_params->sigma2);
            break;
        }
    }

    // apply filter and show result
    filter->doFilter();
    imshow(filter_params->window_name, filter->getResult());
}

void main_homework_3() {
    // 1. Loads an image
    Mat input_img = imread("data/lab3/image.jpg");
    imshow("input image", input_img);
    
    // 2. Prints the histograms of the image
    vector<Mat> img_channels;
    split(input_img, img_channels);
    generate_show_histograms(img_channels, "input");

    // 3. Equalizes the R,G and B channels
    equalizeHist(img_channels[0], img_channels[0]);
    equalizeHist(img_channels[1], img_channels[1]);
    equalizeHist(img_channels[2], img_channels[2]);

    // 4. Shows the equalized image and the histogram of its channels.
    generate_show_histograms(img_channels, "equalized");
    Mat equalized_img;
    merge(img_channels, equalized_img);
    imshow("equalized image", equalized_img);

    // 5. Convert to HSV color space
    Mat hsv;
    cvtColor(input_img, hsv, COLOR_BGR2HSV);
    split(hsv, img_channels);

    // Equalizing only on Value channel which seems to be the best choice. And show histogram.
    int equalize_channel = 2;
    equalizeHist(img_channels[equalize_channel], img_channels[equalize_channel]);
    generate_show_histograms(img_channels, "hsv equalized");

    // Switch back to the RGB color space and visualize the resulting image
    merge(img_channels, equalized_img);
    cvtColor(equalized_img, equalized_img, COLOR_HSV2BGR);
    imshow("equalized hsv image", equalized_img);

    /*************** PART 2 ***************/

    // Create windows
    namedWindow("median filter");
    namedWindow("gaussian filter");
    namedWindow("bilateral filter");

    imshow("median filter", equalized_img);
    imshow("gaussian filter", equalized_img);
    imshow("bilateral filter", equalized_img);

    // Initialize filters
    MedianFilter median_filter(equalized_img, 1);
    FilterParams median_params = { MEDIAN, &median_filter, "median filter", 8};

    GaussianFilter gaussian_filter(equalized_img, 1);
    FilterParams gaussian_params = { GAUSSIAN, &gaussian_filter, "gaussian filter", 8, 25};

    BilateralFilter bilateral_filter(equalized_img, 1);
    FilterParams bilateral_params = { BILATERAL, &bilateral_filter, "bilateral filter", 8, 75, 75};

    // Add parameters trackbars to windows
    createTrackbar("kernel size", "median filter", &median_params.filter_size, 16, onTrackbarChange, (void*)&median_params);

    createTrackbar("kernel size", "gaussian filter", &gaussian_params.filter_size, 16, onTrackbarChange, (void*)&gaussian_params);
    createTrackbar("sigma", "gaussian filter", &gaussian_params.sigma1, 50, onTrackbarChange, (void*)&gaussian_params);

    createTrackbar("sigma range", "bilateral filter", &bilateral_params.sigma1, 150, onTrackbarChange, (void*)&bilateral_params);
    createTrackbar("sigma space", "bilateral filter", &bilateral_params.sigma2, 150, onTrackbarChange, (void*)&bilateral_params);
    
    waitKey(0);
}

/**
 * Generate histograms for the three channels of an image
 * @param img_channels Vector of 3 image channels
 * @param winname Name of the window
 */
void generate_show_histograms(vector<Mat> img_channels, string winname) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    Mat b_hist, g_hist, r_hist;
    calcHist( &img_channels[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist( &img_channels[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist( &img_channels[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    vector<Mat> hists{b_hist, g_hist, r_hist};
    show_histogram(hists, winname);
}