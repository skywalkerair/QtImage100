#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <sys/time.h>
//#include "global.h"
//#include "threadoneimagepro.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //开辟线程内存空间
    threads = new QThread;
    ThreadOneImage = new ThreadOneImagePro;
    ThreadOneImage->moveToThread(threads);

    connect(this,&MainWindow::ThreadOneStart,ThreadOneImage,&ThreadOneImagePro::ImageProcessThread);
    connect(ThreadOneImage,&ThreadOneImagePro::ThreadOneDone,this,&MainWindow::showImg);
    connect(this,&MainWindow::destroyed,this,&MainWindow::CloseALL);

    connect(this,&MainWindow::SendShow,this,&MainWindow::showImg);

    ui->chk_ContinueShow->setChecked(true);


    //初始化 定时器1 用于实时显示图像
    timerShowImg = new QTimer(this);
    timerShowImg->stop();
    timerShowImg->setInterval(1000); //设置定时周期，单位：毫秒
    //connect(timerShowImg,SIGNAL(timeout()),this,SLOT(showImg()));

    //判断图像是否经过处理
    Processed = false;

}

MainWindow::~MainWindow()
{

    delete ui;
}

//显示Mat类型数据在QImage上
void MainWindow::displayMat(Mat image)
{
    Mat rgb;
    QImage img;
    QImage imgScaled;

    if(image.channels() == 3){
        //cvt Mat BGR 2 QImage RGB
        cvtColor(image,rgb,CV_BGR2RGB);
        img = QImage((const unsigned char*)(rgb.data),
                     rgb.cols,rgb.rows,
                     rgb.cols*rgb.channels(),
                     QImage::Format_RGB888);
    }
    else if(image.channels() == 1)
    {

        img = QImage((const unsigned char*)(image.data),
                     image.cols,image.rows,
                     image.cols*1,
                     QImage::Format_Grayscale8);
        //imshow("1",image);
        //qDebug() <<" ###### !" << endl;
    }
    else
    {
        qDebug() <<" the image channels is wrong !" << endl;
    }
    imgScaled = img.scaled(ui->QShowImageLabel->size(),Qt::KeepAspectRatio);
    ui->QShowImageLabel->setPixmap(QPixmap::fromImage(imgScaled));
   // ui->QShowImageLabel->resize(ui->QShowImageLabel->pixmap()->size());



}

//判断是处理前的图像还是处理后的图像
//SLOT
void MainWindow::showImg()
{

    if(Processed)
    {
        //经过处理之后的图像，显示出来
         displayMat(dstImg);

    }
    else{
        //处理之前的图像
        displayMat(srcImg);
        //ThreadOneImage->SetFlag(true);
       // qDebug() << "显示 处理之前的图像 " << endl;
    }
    if(threads->isRunning() == true)
    {
        ThreadOneImage->SetFlag(true);
        threads->quit();
        threads->wait();
    }


}


void MainWindow::on_btn_OpenFile_clicked()
{
    //按下“打开文件”按钮，将需要处理的原图像读入
    QString fileName = QFileDialog::getOpenFileName(
                        this,
                        "Please select an image","",
                        "Image File(*.jpg *.png *.bmp *.pgm *.pbm *.jpeg);;All(*.*)");

    //srcImg = cv::imread(fileName.toLatin1().data());
    srcImg = cv::imread(fileName.toStdString());
    if(srcImg.data == NULL)
    {
        qDebug() << "Open Image File failed using OpenCV! "<< endl;
    }
    else{
        //cv::imshow("SrcImg",SrcImg);
       // displayMat(SrcImg);
        qDebug()<<"Open the SrcImage success!"<<endl;
        Processed = false;
        if(timerShowImg->isActive() == false)
        {
            timerShowImg->start(1);
        }

    }


}

//处理图像按钮
void MainWindow::on_btn_ImgPorcess_clicked()
{
    if(threads->isRunning() == true)
    {
        return;
    }

    threads->start();
    ThreadOneImage->SetFlag(false);
    Processed = true;

    emit ThreadOneStart();


}


//鼠标左键按下
void MainWindow::mousePressEvent(QMouseEvent* event)
{
    if(event->button() == Qt::LeftButton)
    {
        //如果鼠标左键点下
        Processed = !Processed;

    }
    emit SendShow();
    //qDebug() << "checkbox state is " << ui->chk_ContinueShow->checkState() << endl;
}
//鼠标左键释放
void MainWindow::mouseReleaseEvent(QMouseEvent* event)
{
    if(event->button() == Qt::LeftButton)
        Processed = !Processed;

    //ThreadOneImage->SetFlag(false);
    emit SendShow();
}

void MainWindow::CloseALL()
{
    if(threads->isRunning() == false)
    {
        return;
    }

    ThreadOneImage->SetFlag(true);
    threads->quit();
    threads->wait();
    delete threads;
}
//checkbox
void MainWindow::on_chk_ContinueShow_clicked()
{
     if(ui->chk_ContinueShow->checkState() == Qt::Checked)
     {

          timerShowImg->stop();
     }
     else
     {
          timerShowImg->start(1);
     }
}
