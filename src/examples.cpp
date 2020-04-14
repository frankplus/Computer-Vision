#include "examples.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void run_example(int example) {
    string test_image = "data/example/bg.png";

    switch (example) {
        case 1:
            ex_1_show_img(test_image);
            break;
        case 2:
            ex_2_show_gen_img();
            break;
        case 3:
            ex_3_show_gen_colored_img();
            break;
        case 4:
            ex_4_affine_transform(test_image);
            break;
    }

    waitKey(0);
}

void ex_1_show_img(string path) {
    Mat img = imread(path);
    namedWindow("Example 1");
    imshow("Example 1", img);
}

void ex_2_show_gen_img() {
    Mat img(200, 200, CV_8U);
    for (int i = 0; i < 200; ++i)
        for (int j = 0; j < 200; ++j)
            img.at<unsigned char> (i, j) = std::min(i+j, 255);
    namedWindow("Example 2");
    imshow ("Example 2", img);
}

void ex_3_show_gen_colored_img() {
    Mat img(200, 200, CV_8UC3);
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            img.at<Vec3b>(i, j)[0] = i;
            img.at<Vec3b>(i, j)[1] = j;
            img.at<Vec3b>(i, j)[2] = 0;
        }
    }
    namedWindow("Example 3");
    imshow ("Example 3", img);
}

void ex_4_affine_transform(string imgpath) {
    Mat src = imread(imgpath);
    Point2f srcTri[3];
    srcTri[0] = Point2f( 0.f, 0.f );
    srcTri[1] = Point2f( src.cols - 1.f, 0.f );
    srcTri[2] = Point2f( 0.f, src.rows - 1.f );

    Point2f dstTri[3];
    dstTri[0] = Point2f( 0.f, src.rows*0.33f );
    dstTri[1] = Point2f( src.cols*0.85f, src.rows*0.25f );
    dstTri[2] = Point2f( src.cols*0.15f, src.rows*0.7f );

    Mat warp_mat = getAffineTransform( srcTri, dstTri );
    Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
    warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

    Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
    double angle = -50.0;
    double scale = 0.6;
    Mat rot_mat = getRotationMatrix2D( center, angle, scale );
    Mat warp_rotate_dst;
    warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );

    imshow( "Source image", src );
    imshow( "Warp", warp_dst );
    imshow( "Warp + Rotate", warp_rotate_dst );
}