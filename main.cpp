#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define EXAMPLE 4 // set this to choose which example to execute

using namespace cv;
using namespace std;

void ex_1_show_img(char *path) {
    cv::Mat img = cv::imread(path);
    cv::namedWindow("Example 1");
    cv::imshow("Example 1", img);
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

void ex_4_affine_transform(char *imgpath) {
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

int main(int argc, char** argv) {

    switch(EXAMPLE) {
        case 1:
            ex_1_show_img(argv[1]);
            break;
        case 2:
            ex_2_show_gen_img();
            break;
        case 3:
            ex_3_show_gen_colored_img();
            break;
        case 4:
            ex_4_affine_transform(argv[1]);
            break;
    }

    cv::waitKey(0);
    return 0;
}