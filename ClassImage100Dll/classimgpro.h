#ifndef CLASSIMGPRO_H
#define CLASSIMGPRO_H
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

const int fheight = 128,fwidth = 128;
//复数 结构体
struct fourier_str{
    std::complex<double> coef[fheight][fwidth];
};

class ImagePro
{
public:
    //1. 通道替换
    cv::Mat SwitchByManual(cv::Mat &SrcImg);
    cv::Mat SwitchBySplitAndMerge(const cv::Mat &SrcImg);
    cv::Mat SwitchByMixChannels(const cv::Mat &SrcImag);

    //2. 灰度化
    //Y = 0.2126R + 0.7152G+ 0.0722B
    cv::Mat ScaleByManual(const cv::Mat &SrcImg);
    cv::Mat SCaleBycvtColor(const cv::Mat &SrcImg);

    //3. 二值化
    cv::Mat ThresholdingByManual(const cv::Mat &SrcImg);
    cv::Mat ThresholdingByThreshold(const cv::Mat &SrcImg,Mat &Gray, int threshold = 210);

    //4. 大津阈值Otsu
    cv::Mat OtsuThresholdingByManual(cv::Mat & SrcImg);

    //8. 最大池化
    cv::Mat MaxPooling(const cv::Mat &SrcImg);

    //文字处理
    cv::Mat WordProcessing(const cv::Mat & SrcImg,Mat & Gray);

    //9. 高斯滤波
    cv::Mat GaussianFilterManual(const cv::Mat&SrcImg,int ksize = 3,float sigma = 1.0);
    cv::Mat GaussianFilterOpenCV(const cv::Mat &SrcImg,int ksize = 3,float sigma = 1.0);

    //10. 中值滤波
    cv::Mat MedianFilterManual(const cv::Mat &SrcImg,int ksize = 3);
    cv::Mat MedianFilterOpenCV(const cv::Mat &SrcImg,int ksize = 3);

    //11. 均值滤波
    cv::Mat MeanFilterManual(const cv::Mat & src,int ksize = 3);
    cv::Mat MeanFilterOpenCV(const cv::Mat &src,int ksize =3);

    //12. 对角滤波
    cv::Mat MotionFilterManual(const cv::Mat &src,int ksize = 3);
    cv::Mat MotionFilterOpenCV(const cv::Mat& src,int ksize = 3);

    //13. 最大值最小值模糊
    cv::Mat MaxMinFilterManual(const cv::Mat &src,int ksize = 3);

    //14. 差分滤波
    cv::Mat DifferentialFilterManual(const cv::Mat &src,bool bHorizontial = false);

    //15. sobel滤波器
    cv::Mat SobelFilterManual(const cv::Mat &src,bool bHorizontial = false);

    //16. Prewitt滤波器
    cv::Mat PrewittFilterManual(const cv::Mat & src,bool bHorizontial = false);

    //17. Laplacian滤波器
    cv::Mat LaplacianFilterManual(const cv::Mat &src);
    cv::Mat LaplacianFilterOpenCV(const cv::Mat &src);

    //18. Emboss滤波器
    cv::Mat EmbossFilterManual(const cv::Mat &src);
    cv::Mat EmbossFilterOpenCV(const cv::Mat &src);

    //19. Laplacian of Gaussian
    cv::Mat LoGFilterManual(const cv::Mat &src,int ksize = 3,float sigma = 1.0);

    //20. 直方图
    cv::Mat CalGrayHist(const cv::Mat & src);

    //21. 直方图归一化
    cv::Mat HistNormalization(const cv::Mat &img,int a,int b);

    //22. 直方图均衡化
    cv::Mat HistEqualization(cv::Mat &img);

    //23. gamma映射校正
    cv::Mat GammaCorrection(cv::Mat img,double gamma_c = 1,double gamma_g = 2.2);

    //24. 最近邻插值
    cv::Mat NearestNeighbor(cv::Mat img,double rx = 1.5,double ry = 1.5);

    //25. 双线性插值
    cv::Mat Bilinear(cv::Mat img,double rx = 1.5,double ry = 1.5);

    //26. 双三次插值
    cv::Mat Bicubic(cv::Mat img,double rx,double ry);

    //28.仿射变换-平移
    cv::Mat Affine(cv::Mat img,double a,double b,double c,double d,double tx,double ty);

    //29.仿射变换-旋转
    cv::Mat AffineTheta(cv::Mat img,double a,double b,double c,double d,double tx,double ty,double theta);

    //32.傅里叶变换
    cv::Mat FourierTransform(cv::Mat img);

    //33.傅里叶变换 低通滤波
    cv::Mat FTLFilter(cv::Mat img);


private:
    //weight function
    double h(double t);

    //clip value
    int val_clip(int x,int min,int max);

    /************** 傅里叶变换相关 *****************/
    //Discrete Fourier transformation 离散傅里叶变换
    fourier_str DFT(cv::Mat img,fourier_str fourier_s);

    cv::Mat IDFT(cv::Mat out,fourier_str fourier_s);


    //Low Pass Filter
    fourier_str lpf(fourier_str fourier_s,double pass_r);

protected:


};





#endif // CLASSIMGPRO_H
