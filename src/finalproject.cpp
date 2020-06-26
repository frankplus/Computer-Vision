#include "homework_5.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

void detect_and_display( Mat frame );

const string path = "data/final_project/*.jpg";

CascadeClassifier tree_cascade;

void main_finalproject() 
{

    if( !tree_cascade.load( "data/final_project/cascade.xml" ) )
    {
        cout << "Error loading cascade\n";
        return;
    }

    // load images
    vector<String> images_paths;
    glob(path, images_paths);

    for (String imgpath: images_paths) {
        cout << imgpath << endl;
        Mat image = imread(imgpath);
        resize(image, image, Size(400,500));
        detect_and_display(image);
    }
}

void detect_and_display( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> trees;
    tree_cascade.detectMultiScale( frame_gray, trees, 1.1, 15);
    for ( size_t i = 0; i < trees.size(); i++ )
    {
        Point pt1(trees[i].x, trees[i].y);
        Point pt2(trees[i].x + trees[i].width, trees[i].y + trees[i].height);
        rectangle(frame, pt1, pt2, Scalar(0,0,128), 2);
        // Point center( trees[i].x + trees[i].width/2, trees[i].y + trees[i].height/2 );
        // ellipse( frame, center, Size( trees[i].width/2, trees[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );

    }

    imshow( "Capture - Face detection", frame );
    waitKey(0);
}