#ifndef __FINALPROJECT__
#define __FINALPROJECT__

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;

struct DetectionParams
{
    CascadeClassifier tree_cascade;
    Mat input_image;
    Mat filtered_image;
    Range hue_range{0, 195};
    Range value_range{92, 256};
    int filter_sigma = 134;
    int min_neighbors = 14;
    int min_size = 75;
    int delta_brightness = 48;
};

void main_finalproject();

/**
 * Preprocess image before the detection phase. The operations are histogram equalization, 
 * bilateral filter and brighten up pixels in a given hue and value range.
 * @param input The input image
 * @param result The output image
 * @param sigma Sigma color and sigma space for the bilateral filter
 * @param hue_range The pixels inside this hue range are brighten up
 * @param value_range The pixels inside this value range are brighten up
 */
void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range, Range value_range, int delta_brightness);

/**
 * Detect the trees using cascade classifier at multi scale. Draw rectangles around the 
 * detected trees and show the result both on the preprocessed image and on the original image.
 * @param filter_params Pointer to DetectionParams struct containing original and preprocessed 
 *                      image and detection parameters
 */
void detect_and_display( DetectionParams *params );

/**
 * Print the parameters for debugging purposes.
 * @param filter_params pointer to DetectionParams struct containing preprocessing and detection parameters
 */
void print_parameters(DetectionParams *filter_params);

#endif