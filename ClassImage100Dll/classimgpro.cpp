#include"classimgpro.h"
#include <QDebug>

//1. 交换通道 方法1：直接使用循环 将BGR 转成 RGB
cv::Mat ImagePro::SwitchByManual(cv::Mat &SrcImg)
{
    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),SrcImg.type());

    for(auto y = 0;y < SrcImg.rows;++y)
    {
        for(auto x = 0;x < SrcImg.cols;++x)
        {
            //BGR -> RGB
            dst.at<cv::Vec3b>(y,x)[0] = SrcImg.at<cv::Vec3b>(y,x)[2];
            dst.at<cv::Vec3b>(y,x)[1] = SrcImg.at<cv::Vec3b>(y,x)[1];
            dst.at<cv::Vec3b>(y,x)[2] = SrcImg.at<cv::Vec3b>(y,x)[0];
        }
    }
    return dst;
}

Mat ImagePro::SwitchBySplitAndMerge(const Mat &SrcImg)
{
    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),SrcImg.type());

    std::vector<cv::Mat> channels;
    cv::split(SrcImg,channels);

    std::swap(channels[0],channels[2]);

    cv::merge(channels,dst);

    return dst;
}

Mat ImagePro::SwitchByMixChannels(const Mat &SrcImag)
{
    cv::Mat dst(SrcImag.size(),SrcImag.type());

    std::vector<int> fromTo = {0, 2, 1, 1, 2, 0};

    cv::mixChannels(SrcImag,dst,fromTo);

    return dst;
}

//2 灰度化
Mat ImagePro::ScaleByManual(const Mat &SrcImg)
{
    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),CV_8UC1);

    for(auto y = 0;y < SrcImg.rows;++y)
    {
        for(auto x = 0;x < SrcImg.cols;++x)
        {
            auto points = SrcImg.at<cv::Vec3b>(y,x);
            dst.at<uchar>(y,x) = (uchar)(0.2126 * points[2] + 0.7152 * points[1] + 0.0722 * points[0]);
        }
    }

    return dst;
}

Mat ImagePro::SCaleBycvtColor(const Mat &SrcImg)
{
    cv::Mat dst;
    cv::cvtColor(SrcImg,dst,cv::COLOR_BGR2GRAY);
    return dst;
}

Mat ImagePro::ThresholdingByManual(const Mat &SrcImg)
{
    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),CV_8UC1);

    //三通道变成单通道
    Mat tmp;
    cv::cvtColor(SrcImg,tmp,cv::COLOR_BGR2GRAY);

    for(auto y = 0;y < tmp.rows;++y)
    {
        for(auto x = 0;x < tmp.cols;++x)
        {
            if(tmp.at<uchar>(y,x) > 128)
            {
                dst.at<uchar>(y,x) = 255;
            }
            else
            {
                dst.at<uchar>(y,x) = 0;
            }
        }
    }

    return dst;
}

Mat ImagePro::ThresholdingByThreshold(const Mat &SrcImg,Mat &Gray, int threshold)
{
    cv::Mat dst;
    if(SrcImg.channels() == 3)
    {
        //三通道变成单通道
        cv::cvtColor(SrcImg,Gray,cv::COLOR_BGR2GRAY);
    }

    cv::threshold(Gray,dst,threshold,255,cv::THRESH_BINARY);
    return dst;
}

Mat ImagePro::OtsuThresholdingByManual(Mat &SrcImg)
{

    //三通道变成单通道
    Mat tmp;
    if(SrcImg.channels() == 3)
    {
        cv::cvtColor(SrcImg,tmp,cv::COLOR_BGR2GRAY);
    }

    //大津阈值
    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),CV_8UC1);

    double maxVariance = 0.0;
    uchar bestThresh = 0;

    //计算出最佳阈值
    for(int thresh = 0;thresh < 255;++thresh)
    {
        std::vector<uchar> overtopVec,underVec;

        for(auto y = 0;y < tmp.rows;++y)
        {
            for(auto x = 0;x < tmp.cols;++x)
            {
                uchar point = tmp.at<uchar>(y,x);
                if(point > thresh)
                {
                    overtopVec.emplace_back(point);
                }
                else
                {
                    underVec.emplace_back(point);
                }

            }
        }

        double p0 = 1.0 * overtopVec.size() / (tmp.rows * tmp.cols);
        double p1 = 1.0 * underVec.size() / (tmp.rows * tmp.cols);

        double x0 = 0.0;
        if(!underVec.empty())
        {
            for_each(overtopVec.begin(), overtopVec.end(), [&x0](double item){x0 += item;});
            x0 = x0 / overtopVec.size();
        }
        double x1 = 0.0;
        if(!underVec.empty())
        {
            for_each(underVec.begin(),underVec.end(),[&x1](double item){x1 += item;});
            x1 = x1 / underVec.size();
        }

        double var = 2 * p0 * p1 * std::pow((x0 - x1),2);

        if(var > maxVariance)
        {
            maxVariance = var;
            bestThresh = thresh;
        }

    }

    //根据最佳阈值进行二值化
    for(auto y = 0;y < tmp.rows;y++)
    {
        for(auto x = 0;x < tmp.cols;x++)
        {
            if(tmp.at<uchar>(y,x) > bestThresh)
            {
                dst.at<uchar>(y,x) = 255;
            }
        }
    }

    qDebug() << "bestThresh:" <<  bestThresh << endl;
    return dst;
}

Mat ImagePro::MaxPooling(const Mat &SrcImg)
{
    //最大池化
    cv::Mat dst = SrcImg.clone();

    int ksize = 3;

    for(int y = 0;y < SrcImg.rows;y+=ksize)
    {
        for(int x = 0;x < SrcImg.cols;x+=ksize)
        {
            for(int c = 0;c < SrcImg.channels();c++)
            {
                uchar maxValue = 0;
                for(int dy = 0;dy < ksize;dy++)
                {
                    for(int dx = 0;dx < ksize;dx++)
                    {
                        if(SrcImg.at<cv::Vec3b>(y+dy,x+dx)[c] > maxValue)
                        {
                            maxValue = SrcImg.at<cv::Vec3b>(y+dy,x+dx)[c];
                        }
                    }
                }

                for(int dy = 0;dy < ksize;dy++)
                {
                    for(int dx = 0;dx < ksize;dx++)
                    {
                        dst.at<cv::Vec3b>(y+dy,x+dx)[c] = maxValue;
                    }
                }
            }
        }
    }

    return dst;
}

Mat ImagePro::WordProcessing(const Mat &SrcImg, Mat & Gray)
{
    //简单处理文字信息
    Mat dst = cv::Mat::zeros(SrcImg.size(),CV_8UC1);
    int kernel_size = 3;
       int scale = 5;
       int delta = 2;
       int ddepth = CV_8UC1;
    //Mat tmp;
    cv::cvtColor(SrcImg,Gray,CV_RGB2GRAY);
    //Gray = ThresholdingByThreshold(Gray,Gray);
    Laplacian(Gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

    //dst = OtsuThresholdingByManual(dst);
    //dst = ThresholdingByThreshold(dst,Gray,threshold);
//    for(int y = 0;y < SrcImg.rows;y++)
//    {
//        for(int x = 0;x < SrcImg.cols;x++)
//        {
//            if(dst.at<uchar>(y,x) == 0)
//            {
//                dst.at<uchar>(y,x) = 255;
//            }

//        }
//    }
    //dst = MaxPooling(dst);

    return dst;
}

Mat ImagePro::GaussianFilterManual(const Mat &SrcImg, int ksize, float sigma)
{
    assert((ksize > 0)&&(1 == (ksize % 2)) && (!SrcImg.empty()));

    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),SrcImg.type());

    //产生高斯核
    int origin = floor(ksize / 2);
    float kernel[ksize][ksize];

    float sum = 0;
    for(int y = 0;y < ksize;y++)
    {
        for(int x = 0;x < ksize;x++)
        {
            int u = y - origin;
            int v = x - origin;

            kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp(-(u * u + v * v)) / (2 * sigma * sigma);
            sum += kernel[y][x];
        }
    }

    //核归一化
    for(int y = 0;y < ksize;y++)
    {
        for(int x = 0;x < ksize;x++)
        {
            kernel[y][x] /= sum;
        }
    }

    //滤波操作
    for(int y = 0;y < SrcImg.rows;y++)
    {
        for(int x = 0;x < SrcImg.cols;x++)
        {
            for(int c = 0;c < SrcImg.channels();c++)
            {
                float value = 0;
                for(int dy = -origin;dy < origin + 1;dy++)
                {
                    for(int dx = -origin;dx < origin + 1;dx++)
                    {
                        if(((dx + x) >= 0) && ((dy + y) >= 0))
                        {
                            //对应坐标相乘
                            value += (float)SrcImg.at<cv::Vec3b>(y + dy,x +dx)[c] * kernel[dy + origin][dx + origin];
                        }
                    }
                }
                dst.at<cv::Vec3b>(y,x)[c] = (uchar)value;
            }
        }
    }



    return dst;

}

Mat ImagePro::GaussianFilterOpenCV(const Mat &SrcImg, int ksize, float sigma)
{
    assert((ksize > 0) && ( 1 == (ksize % 2)) && (!SrcImg.empty()));

    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),SrcImg.type());
    cv::GaussianBlur(SrcImg,dst,cv::Size(ksize,ksize),sigma);

    return dst;
}

Mat ImagePro::MedianFilterManual(const Mat &SrcImg, int ksize)
{
    assert((ksize > 0)&& (1 == (ksize % 2)) && (!SrcImg.empty()));

    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),SrcImg.type());

    //确定核中心点
    int origin = floor(ksize / 2);

    //滤波操作
    for(int y = 0;y < SrcImg.rows;y++)
    {
        for(int x = 0;x < SrcImg.cols;x++)
        {
            for(int c = 0;c < SrcImg.channels();c++)
            {
                std::vector<uchar> values(ksize * ksize);
                int index = 0;

                for(int dy = -origin;dy < origin + 1;dy++)
                {
                    for(int dx = -origin;dx < origin +1;dx++)
                    {
                        if(((dx + x) >= 0) && ((dy + y) >= 0))
                        {
                            values[index] = SrcImg.at<cv::Vec3b>(y+dy,x+dx)[c];
                            ++index;
                        }
                    }
                }
                std::sort(values.begin(),values.end());
                dst.at<cv::Vec3b>(y,x)[c] = values[std::floor(values.size() / 2)+1];
            }
        }
    }

    return dst;
}

Mat ImagePro::MedianFilterOpenCV(const Mat &SrcImg, int ksize)
{
    assert((ksize > 0) && (1 == (ksize % 2)) && (!SrcImg.empty()));

    cv::Mat dst = cv::Mat::zeros(SrcImg.size(),SrcImg.type());

    cv::medianBlur(SrcImg,dst,ksize);
    return dst;
}

Mat ImagePro::MeanFilterManual(const Mat &src, int ksize)
{
    assert((ksize > 0) && ( 1 == (ksize % 2)) && ( !src.empty()));

    cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

    //确定核中心
    int origin = floor(ksize / 2);

    //滤波操作
    for(int y = 0;y < src.rows;y++)
    {
        for(int x = 0;x < src.cols;x++)
        {
            for(int c = 0;c < src.channels();c++)
            {
                int sum = 0;

                for(int dy = -origin;dy < origin + 1;dy++)
                {
                    for(int dx = -origin;dx < origin + 1;dx++)
                    {
                        if(((y +dy) >= 0) && ((x +dx) >= 0))
                        {
                            sum += src.at<cv::Vec3b>(y +dy,x+dx)[c];
                        }

                    }
                }

                dst.at<cv::Vec3b>(y,x)[c] = (uchar)(sum / (ksize * ksize));
            }
        }
    }

    return dst;
}

Mat ImagePro::MeanFilterOpenCV(const Mat &src, int ksize)
{
    assert((ksize > 0) && (1 == (ksize % 2)) && (!src.empty()));

    cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

    cv::blur(src,dst,cv::Size(ksize,ksize));

    return dst;
}

Mat ImagePro::MotionFilterManual(const Mat &src, int ksize)
{
    assert( (ksize > 0) && (1 == ((ksize % 2))) && (!src.empty()));

    cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

    //产生滤波核
    std::vector<float> kernel(ksize * ksize);
    for(int i = 0;i < ksize;i++)
    {
        for(int j = 0;j < ksize;j++)
        {
            if(i == j)
            {
                kernel[i * ksize + j] = 1.0 / ksize;
            }
        }
    }


    //滤波核中心
    int origin = floor(ksize / 2);

    //滤波操作
    for(int y = 0;y < src.rows;y++)
    {
        for(int x = 0;x < src.cols;x++)
        {
            for(int c = 0;c < src.channels();c++)
            {
                float value = 0;

                for(int dy = -origin;dy < origin + 1;dy++)
                {
                    for(int dx = -origin;dx < origin + 1;dx++)
                    {
                        if(((dx + x) >= 0) && ((y+dy) >= 0))
                        {
                            value += src.at<cv::Vec3b>(y+dy,x+dx)[c] * kernel[(dy+origin)*ksize + dx + origin];
                        }
                    }
                }

            dst.at<cv::Vec3b>(y,x)[c] = (uchar)value;

            }
        }
    }



    return dst;
}

Mat ImagePro::MotionFilterOpenCV(const Mat &src, int ksize)
{
    assert((ksize > 0) && ( 1== (ksize % 2)) && (!src.empty()));

    cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

    cv::Mat kernel = cv::Mat::zeros(ksize,ksize,CV_32F);
    for(int i = 0;i < ksize;i++)
    {
        kernel.at<float>(i,i) = 1.0 /ksize;
    }

    cv::filter2D(src,dst,src.depth(),kernel);
    cv::blur(src,dst,cv::Size(ksize,ksize));

    return dst;
}

Mat ImagePro::MaxMinFilterManual(const Mat &src, int ksize)
{
    assert( (ksize > 0) && ( 1== (ksize % 2)) && (!src.empty()));

    Mat tmp;
    //因为最大值和最小值是在灰度图像上
    if(src.channels() == 3)
    {
        tmp = SCaleBycvtColor(src);
    }

    cv::Mat dst = cv::Mat::zeros(tmp.size(),tmp.type());

    //滤波核中心
    int origin = floor(ksize / 2);

    //滤波操作
    for(int y = 0;y < tmp.rows;y++)
    {
        for(int x = 0;x < tmp.cols;x++)
        {
            uchar maxVal = 0;
            uchar minVal = 255;

            for(int dy = -origin;dy < origin + 1;dy++)
            {
                for(int dx = -origin;dx < origin + 1;dx++)
                {
                    if(((dx +x)>=0) && ((dy +y) >= 0))
                    {
                        uchar val = tmp.at<uchar>(y+dy,x+dx);
                        if(val > maxVal)
                            maxVal = val;
                        if(val < minVal)
                            minVal = val;
                    }
                }
            }

            dst.at<uchar>(y,x) = maxVal - minVal;
        }
    }
    return dst;
}


Mat ImagePro::DifferentialFilterManual(const Mat &src, bool bHorizontial)
{


    Mat tmp;
    //因为最大值和最小值是在灰度图像上
    if(src.channels() == 3)
    {
        tmp = SCaleBycvtColor(src);
    }
    cv::Mat dst = cv::Mat::zeros(tmp.size(),tmp.type());


    //产生差分核
    float kernel[3][3] = {{0,-1,0},{0,1,0},{0,0,0}};

    if(bHorizontial)
    {
        kernel[0][1] = 0;
        kernel[1][0] = -1;
    }

    int origin = 1;

    //滤波操作
    for(int y = 0;y < tmp.rows;y++)
    {
        for(int x = 0;x < tmp.cols;x++)
        {
            float value = 0;
            for(int dy = -origin;dy < origin + 1;dy++)
            {
                for(int dx = -origin;dx < origin + 1;dx++)
                {
                    if(((dx + x) >= 0) && ((dy +y)>=0))
                    {
                        //对应坐标相乘
                        value += (float)tmp.at<uchar>(y+dy,x+dx) * kernel[dy + origin][dx +origin];
                    }
                }
            }

            value = fmax(value,0);
            dst.at<uchar>(y,x) = (uchar)value;
        }
    }
    return dst;
}

Mat ImagePro::SobelFilterManual(const Mat &src, bool bHorizontial)
{
    //灰度图
    Mat tmp;
    tmp = SCaleBycvtColor(src);

    cv::Mat dst = cv::Mat::zeros(tmp.size(),tmp.type());

    //sobel 算子
    float kernel[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    if(bHorizontial)
    {
        kernel[0][1] = 0;
        kernel[0][2] = 1;
        kernel[1][0] = -2;
        kernel[1][2] = 2;
        kernel[2][0] = -1;
        kernel[2][1] = 0;
    }

    int origin = 1;

    //滤波操作
    for(int y = 0;y < tmp.rows;y++)
    {
        for(int x = 0;x < tmp.cols;x++)
        {
            float value = 0;
            for(int dy = -origin;dy < origin + 1;dy++)
            {
                for(int dx = -origin;dx < origin + 1;dx++)
                {
                    if(((x + dx) >= 0) && ((y+dy)>= 0))
                    {
                        value += (float)tmp.at<uchar>(dy+y,dx+x)*kernel[dy+origin][dx+origin];
                    }
                }
            }
            value = fmax(value,0);
            dst.at<uchar>(y,x) = (uchar)value;
        }
    }
    return dst;
}

Mat ImagePro::PrewittFilterManual(const Mat &src, bool bHorizontial)
{
    cv::Mat tmp = SCaleBycvtColor(src);

    cv::Mat dst = cv::Mat::zeros(tmp.size(),tmp.type());
    //Prewtt算子
    float kernel[3][3] = {{1,1,1},{0,0,0},{-1,-1,-1}};

    if(bHorizontial)
    {
        kernel[0][1] = 0;
        kernel[0][2] = -1;
        kernel[1][0] = 1;
        kernel[1][2] = -1;
        kernel[2][0] = 1;
        kernel[2][1] = 0;
    }


    int origin = 1;

    //滤波操作
    for(int y = 0;y < tmp.rows;y++)
    {
        for(int x = 0;x < tmp.cols;x++)
        {
            float value = 0;
            for(int dy = -origin;dy < origin + 1;dy++)
            {
                for(int dx = -origin;dx < origin + 1;dx++)
                {
                    if(((dx + x) >= 0) && ((dy + y) >= 0))
                    {
                        //对应坐标相乘
                        value += (float)tmp.at<uchar>(y+dy,dx+x)*kernel[dy+origin][dx+origin];
                    }
                }
            }
            value = fmax(value,0);
            dst.at<uchar>(y,x) = (uchar)value;
        }
    }

    return dst;
}

Mat ImagePro::LaplacianFilterManual(const Mat &src)
{
    cv::Mat tmp = SCaleBycvtColor(src);

    cv::Mat dst = cv::Mat::zeros(tmp.size(),tmp.type());

    cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,1,0,1,-4,1,0,1,0);

    int origin = 1;

    //滤波操作
    for(int y = 0;y < tmp.rows;y++)
    {
        for(int x = 0;x < tmp.cols;x++)
        {
            float value = 0;
            for(int dy = -origin;dy < origin + 1;dy++)
            {
                for(int dx = -origin;dx < origin + 1;dx++)
                {
                    if( ((dx + x) >= 0)&&((dy +y) >= 0))
                    {
                        //对应坐标相乘
                        value += (float)tmp.at<uchar>(dy+y,dx+x) * kernel.at<float>(dy+origin,dx+origin);
                    }
                }
            }

            value = fmax(value,0);
            value = fmin(value,255);

            dst.at<uchar>(y,x) = (uchar)value;
        }
    }

    return dst;
}

Mat ImagePro::LaplacianFilterOpenCV(const Mat &src)
{
    cv::Mat dst;
    cv::Laplacian(src,dst,-1,1);

    return dst;
}

Mat ImagePro::EmbossFilterManual(const Mat &src)
{
    cv::Mat tmp = SCaleBycvtColor(src);

    cv::Mat dst = cv::Mat::zeros(tmp.size(),tmp.type());

    cv::Mat kernel = (cv::Mat_<float>(3,3) << -2,-1,0,-1,1,1,0,1,2);

    int original = 1;

    //滤波操作
    for(int y = 0;y < tmp.rows;y++)
    {
        for(int x = 0;x < tmp.cols;x++)
        {
            float value = 0;
            for(int dy = -original;dy < original + 1;dy++)
            {
                for(int dx = -original;dx < original + 1;dx++)
                {
                    if(((dx +x)>=0) && ((dy+y) >= 0))
                    {
                        //对应坐标相乘
                        value += (float)tmp.at<uchar>(y+dy,x+dx)*kernel.at<float>(dy+original,dx+original);
                    }

                }
            }

            value = fmax(value,0);
            value = fmin(value,255);
            dst.at<uchar>(y,x) = (uchar)value;
        }
    }
    return dst;
}

Mat ImagePro::EmbossFilterOpenCV(const Mat &src)
{
    cv::Mat dst;
    cv::Mat kernel = (cv::Mat_<float>(3,3) << -2,-1,0,-1,1,1,0,1,2);
    cv::filter2D(src,dst,-1,kernel);
    return dst;
}

Mat ImagePro::LoGFilterManual(const Mat &src, int ksize, float sigma)
{
    assert((ksize > 0) && (1 == (ksize % 2)) && (!src.empty()));

    cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

    //产生高斯核
    int origin = floor(ksize / 2);
    float kernel[ksize][ksize];

    float sum = 0;
    for(int y = 0;y < ksize;y++)
    {
        for(int x = 0;x < ksize;x++)
        {
            int u = y - origin;
            int v = x - origin;

            kernel[y][x] = (u*u + v*v -sigma*sigma) / (2 * M_PI * pow(sigma,6))
                          * (exp(-(u*u +v*v) / (2 * sigma * sigma)));
            sum += kernel[y][x];
        }
    }

    //核归一化
    for(int y = 0;y < ksize;y++)
    {
        for(int x = 0;x < ksize;x++)
        {
            kernel[y][x] /= sum;
        }
    }

    //滤波操作
    for(int y = 0;y < src.rows;y++)
    {
        for(int x= 0;x < src.cols;x++)
        {
            for(int c = 0;c < src.channels();c++)
            {
                float value = 0;
                for(int dy = -origin;dy < origin + 1;dy++)
                {
                    for(int dx = -origin;dx < origin + 1;dx++)
                    {
                        if(((x + dx) >= 0) && ((dy + y) >= 0))
                        {
                            value += (float)src.at<cv::Vec3b>(y + dy,x + dx)[c] * kernel[dy + origin][dx + origin];
                        }
                    }
                }
                value = fmax(value,0);
                value = fmin(value,255);
                dst.at<cv::Vec3b>(y,x)[c] = (uchar)value;
            }
        }
    }
    return dst;
}

Mat ImagePro::CalGrayHist(const Mat &src)
{
    cv::Mat tmp = SCaleBycvtColor(src);

    cv::Mat histogram = cv::Mat::zeros(Size(256,256),CV_8UC3);

    int rows = tmp.rows;
    int cols = tmp.cols;

    for(int r = 0;r < rows;r++)
    {
        for(int c = 0;c < cols;c++)
        {
            int index = int(tmp.at<uchar>(r,c));
            histogram.at<int>(0,index) += 1;
        }
    }


    return histogram;
}

Mat ImagePro::HistNormalization(const Mat &img,int a,int b)
{
    // get height and width
      int width = img.cols;
      int height = img.rows;
      int channel = img.channels();

      int c, d;
      int val;

      // prepare output
      cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);

      // get [c, d]
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          for (int _c = 0; _c < channel; _c++){
            val = (float)img.at<cv::Vec3b>(y, x)[_c];
            c = fmin(c, val);
            d = fmax(d, val);
          }
        }
      }

      // histogram transformation
      for (int y = 0; y < height; y++){
        for ( int x = 0; x < width; x++){
          for ( int _c = 0; _c < 3; _c++){
            val = img.at<cv::Vec3b>(y, x)[_c];

            if (val < a){
              out.at<cv::Vec3b>(y, x)[_c] = (uchar)a;
            }
            else if (val <= b){
              out.at<cv::Vec3b>(y, x)[_c] = (uchar)((b - a) / (d - c) * (val - c) + a);
            }
            else {
              out.at<cv::Vec3b>(y, x)[_c] = (uchar)b;
            }
          }
        }
      }

      return out;
}

Mat ImagePro::HistEqualization(Mat &img)
{
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    cv::Mat out = cv::Mat::zeros(height,width,CV_8UC3);

    //histogram equalization hyper-parameters
    double Zmax = 255;
    double hist[255];
    double S = height * width * channels;

    int val;
    double hist_sum = 0;

    //hist init
    for(int i = 0;i < 255;i++)
    {
        hist[i] = 0;
    }

    //get hist sum
    for(int y = 0;y < height;y++)
    {
        for(int x = 0;x < width;x++)
        {
            for(int c = 0;c < channels;c++)
            {
                val = (int)img.at<cv::Vec3b>(y,x)[c];
                hist[val]++;
            }
        }
    }

    //hist equalization
    for(int y = 0;y < height;y++)
    {
        for(int x = 0;x < width;x++)
        {
            for(int c = 0;c < channels;c++)
            {
                val = (int)img.at<cv::Vec3b>(y,x)[c];

                //get hist sum <= current pixel value
                hist_sum = 0;
                for(int l = 0;l < val;l++)
                {
                    hist_sum += hist[l];
                }

                out.at<cv::Vec3b>(y,x)[c] =(uchar)(Zmax / S * hist_sum);
            }
        }
    }


    return out;
}

Mat ImagePro::GammaCorrection(Mat img, double gamma_c, double gamma_g)
{
    //get height and width
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    cv::Mat dst = cv::Mat::zeros(height,width,CV_8UC3);

    double val;

    for(int y = 0;y < height;y++)
    {
        for(int x = 0;x < width;x++)
        {
            for(int c = 0;c < channel;c++)
            {
                val = (double)img.at<cv::Vec3b>(y,x)[c]/255;

                dst.at<cv::Vec3b>(y,x)[c] = (uchar)(pow(val / gamma_c,1 / gamma_g) * 255);
            }
        }
    }


    return dst;
}

Mat ImagePro::NearestNeighbor(Mat img, double rx, double ry)
{
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    int resized_width = (int)(width * rx);
    int resized_height = (int)(height * ry);
    int x_before,y_before;

    //output image
    cv::Mat out = cv::Mat::zeros(resized_height,resized_width,CV_8UC3);

    for(int y = 0;y < resized_height;y++)
    {
        y_before = (int)round(y / ry);
        y_before = fmin(y_before,width - 1);

        for(int x = 0;x < resized_width;x++)
        {
            x_before = (int)round(x / rx);
            x_before = fmin(x_before,width -1);

            for(int c = 0;c < channel;c++)
            {
                out.at<cv::Vec3b>(y,x)[c] = img.at<cv::Vec3b>(y_before,x_before)[c];
            }
        }
    }
    return out;
}

Mat ImagePro::Bilinear(Mat img, double rx, double ry)
{
    //get height and width
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    //get resized shape
    int resized_width = (int)(width * rx);
    int resized_height = (int)(height * ry);
    int x_before,y_before;
    double dx,dy;
    double val;

    //output image
    cv::Mat out = cv::Mat::zeros(resized_height,resized_width,CV_8UC3);

    //bi-linear interpolation
    for(int y = 0;y < resized_height;y++)
    {
        y_before = (int)floor(y / ry);
        y_before = fmin(y_before,height - 1);
        dy = y / ry - y_before;

        for(int x = 0;x < resized_width;x++)
        {
            x_before = (int)floor(x / rx);
            x_before = fmin(x_before,width - 1);
            dx = x /rx -x_before;

            //compute bi-linear
            for(int c = 0;c < channels;c++)
            {
                val = (1. - dx)*(1. - dy) * img.at<cv::Vec3b>(y_before,x_before)[c] +
                        dx * (1. - dy) *img.at<cv::Vec3b>(y_before,x_before+1)[c] +
                        (1. - dx)*dy*img.at<cv::Vec3b>(y_before+1,x_before)[c]+
                        dx*dy*img.at<cv::Vec3b>(y_before+1,x_before+1)[c];

            out.at<cv::Vec3b>(y,x)[c] = (uchar)val;
            }
        }
    }

    return out;
}


Mat ImagePro::Bicubic(Mat img, double rx, double ry)
{
    //get height and width
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    //get resized shape
    int resized_width =(int)(width * rx);
    int resized_height = (int)(height * ry);
    int x_before,y_before;
    double dx,dy,wx,wy,w_sum;
    double val;
    int _x,_y;

    cv::Mat out = cv::Mat::zeros(resized_height,resized_width,CV_8UC3);


    for(int y = 0;y < resized_height;y++)
    {
        dy = y / ry;
        y_before = (int)floor(dy);

        for(int x = 0;x < resized_width;x++)
        {
            dx = x / rx;
            x_before = (int)floor(dx);


            for(int c = 0;c < channels;c++)
            {
                w_sum = 0;
                val = 0;

                //bi-cubic computation
                for(int j = -1;j < 3;j++)
                {
                    _y = val_clip(y_before+j,0,height-1);
                    wy = h(fabs(dy - _y));

                    for(int i = -1;i < 3;i++)
                    {
                        _x = val_clip(x_before+i,0,width - 1);
                        wx = h(fabs(dx -_x));
                        w_sum += wy * wx;
                        val += (double)img.at<cv::Vec3b>(_y,_x)[c]*wx*wy;
                    }
                }

                val /= w_sum;
                val = val_clip(val,0,255);

                out.at<cv::Vec3b>(y,x)[c] = (uchar)val;
            }
        }
    }
    return out;
}

Mat ImagePro::Affine(Mat img, double a, double b, double c, double d, double tx, double ty)
{
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    double det = a * d - b * c;

    int resized_width = (int)(width * a);
    int resized_height = (int)(height * d);

    int x_before,y_before;
    //double dx,dy,wx,wy,w_sum;
    //double val;
    //int _x,_y;

    cv::Mat out = cv::Mat::zeros(resized_height,resized_width,CV_8UC3);

    for(int y = 0;y < resized_height;y++)
    {
        for(int x = 0;x < resized_width;x++)
        {
            x_before = (int)((d * x - b * y)/det -tx);
            if((x_before < 0) || (x_before >= width))
            {
                continue;
            }

            //get original position y
            y_before = (int)((-c * x + a * y) / det -ty);
            if((y_before < 0) || (y_before >= height)){
                continue;
            }

            //assign pixel to new position
            for(int c = 0;c < channel;c++)
            {
                out.at<cv::Vec3b>(y,x)[c] = img.at<cv::Vec3b>(y_before,x_before)[c];
            }
        }
    }
    return out;
}

Mat ImagePro::AffineTheta(Mat img, double a, double b, double c, double d, double tx, double ty, double theta)
{
    //get width and height
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();
    double det;

    if(theta != 0)
    {
        //Affine parameters
        double rad = theta / 180. * M_PI;
        a = std::cos(rad);
        b = -std::sin(rad);
        c = std::sin(rad);
        d = std::cos(rad);
        tx = 0;
        ty = 0;

        det = a * d - b * c;

        //center transition
        double cx = width / 2;
        double cy = height / 2;
        double new_cx = (d * cx - b - cy) / det;
        double new_cy = (-c * cx + a * cy) / det;

        tx = new_cx - cx;
        ty = new_cy - cy;
    }

    //Resize width and height
    int resize_width = (int)(width * a);
    int resize_height = (int)(height * d);

    if(theta != 0)
    {
        resize_width = (int)width;
        resize_height = (int)height;
    }

    int x_before,y_before;

    cv::Mat out = cv::Mat::zeros(resize_height,resize_width,CV_8UC3);

    //Affine transformation
    for(int y = 0;y < resize_height;y++)
    {
        for(int x = 0;x < resize_width;x++)
        {
            x_before = (int)((d * x - b * y) / det - tx);

            if((x_before < 0) || (x_before >= width))
            {
                continue;
            }

            y_before = (int)((-c*x + a*y) / det -ty);

            if((y_before < 0) || (y_before >= height))
            {
                continue;
            }

            for(int c = 0;c < channel;c++)
            {
                out.at<cv::Vec3b>(y,x)[c] = img.at<cv::Vec3b>(y_before,x_before)[c];
            }

        }
    }
    return out;
}

Mat ImagePro::FourierTransform(cv::Mat img)
{
    fourier_str fourier_s;
    int height = img.rows;
    int width = img.cols;

    cv::Mat out = cv::Mat::zeros(height,width,CV_8UC1);

    cv::Mat gray = SCaleBycvtColor(img);

    fourier_s = DFT(gray,fourier_s);

    out = IDFT(out,fourier_s);

    return out;

}

Mat ImagePro::FTLFilter(Mat img)
{
    int height = img.cols;
    int width = img.rows;

    //创建 结构体
    fourier_str fourier_s;

    //输出图像初始化
    cv::Mat out = cv::Mat::zeros(height,width,CV_8UC1);

    //灰度图像
    cv::Mat gray = SCaleBycvtColor(img);

    //DFT
    fourier_s = DFT(gray,fourier_s);

    //lpf
    fourier_s = lpf(fourier_s,0.5);

    out = IDFT(out,fourier_s);

    return out;
}

double ImagePro::h(double t)
{
    double a = -1;
    if(fabs(t) <= 1)
    {
        return (a+2) * pow(fabs(t),3) - (a + 3)*pow(fabs(t),2) + 1;
    }
    else if(fabs(t) <= 2)
    {
        return a * pow(fabs(t),3)-5*a*pow(fabs(t),2)+8*a*fabs(t)-4*a;
    }
    return 0;
}

int ImagePro::val_clip(int x, int min, int max)
{
    return fmin(fmax(x,min),max);
}

fourier_str ImagePro::DFT(Mat img,fourier_str fourier_s)
{
    double I;
    double theta;
    std::complex<double> val;

    for(int l = 0;l < fheight;l++)
    {
        for(int k = 0;k < fwidth;k++)
        {
            val.real(0);
            val.imag(0);
            for(int y = 0;y < fheight;y++)
            {
                for(int x = 0;x < fwidth;x++)
                {
                    I = (double)img.at<uchar>(y,x);
                    theta = -2 * M_PI * ((double)k* (double)x / (double)fwidth + (double)l *(double)y / (double)fheight);
                    val += std::complex<double>(cos(theta),sin(theta)) * I;
                }
            }
            val /= sqrt(fheight*fwidth);
            fourier_s.coef[l][k] = val;

        }
    }
    return fourier_s;
}

//Inverse Discrete Fourier transformation
Mat ImagePro::IDFT(Mat out, fourier_str fourier_s)
{
    double theta;
    double g;

    std::complex<double> G;
    std::complex<double> val;

    for(int y = 0;y < fheight;y++)
    {
        for(int x = 0;x < fwidth;x++)
        {
            val.real(0);
            val.imag(0);
            for(int l = 0;l < fheight;l++)
            {
                for(int k = 0;k < fwidth;k++)
                {
                    G = fourier_s.coef[l][k];
                    theta = 2 * M_PI * ((double)k * (double)x/(double)fwidth +(double)l * (double)y / (double)fheight);
                    val += std::complex<double>(cos(theta),sin(theta) )*G;
                }
            }
            g = std::abs(val) / sqrt(fheight * fwidth);
            out.at<uchar>(y,x) = (uchar)g;

        }
    }

    return out;
}

fourier_str ImagePro::lpf(fourier_str fourier_s, double pass_r)
{
    // filtering
     int r = fheight / 2;
     int filter_d = (int)((double)r * pass_r);
     for ( int j = 0; j < fheight / 2; j++)
     {
       for ( int i = 0; i < fwidth / 2; i++)
       {
         if (sqrt(i * i + j * j) >= filter_d)
         {
           fourier_s.coef[j][i] = 0;
           fourier_s.coef[j][fwidth - i] = 0;
           fourier_s.coef[fheight - i][i] = 0;
           fourier_s.coef[fheight - i][fwidth - i] = 0;
         }
       }
       return fourier_s;
}




}




