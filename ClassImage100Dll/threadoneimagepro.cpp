#include "threadoneimagepro.h"


ThreadOneImagePro::ThreadOneImagePro(QObject *parent) : QObject(parent)
{
    isStop = false;
}

void ThreadOneImagePro::SetFlag(bool flag)
{
    isStop = flag;
}

void ThreadOneImagePro::ImageProcessThread()
{
    while(isStop == false)
    {

        //运行图像处理函数
        //并计算耗时
        ImagePro m_ImgPro;


        //计算耗时
        struct timeval tpstart,tpend;
        float timeuse;

        //判断是否读取SrcImg成功
        if(srcImg.data != NULL)
        {
           gettimeofday(&tpstart,NULL);

           //dstImg = m_ImgPro.SwitchByManual(srcImg);
           //dstImg = m_ImgPro.SwitchBySplitAndMerge(srcImg);
           //dstImg = m_ImgPro.SwitchByMixChannels(srcImg);
           //dstImg = m_ImgPro.ScaleByManual(srcImg);
           dstImg = m_ImgPro.ThresholdingByManual(srcImg);
           //qDebug() << " " <<  dstImg.at<uchar>(4,6) << endl;
           //dstImg = m_ImgPro.FTLFilter(srcImg);
           gettimeofday(&tpend,NULL);
           timeuse = (1000000*(tpend.tv_sec-tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec)/ 1000000.0;
           qDebug() << timeuse << " s";


        }
        else
        {
            qDebug() << "输入的原始图像有误！" << endl;
            return ;
        }
        emit ThreadOneDone();

        if(isStop == true)
        {
            break;
        }
    }

}
