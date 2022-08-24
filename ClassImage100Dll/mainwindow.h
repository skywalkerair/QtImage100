#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QImage>
#include <QTimer>
#include <QMouseEvent>
#include <QThread>

#include "threadoneimagepro.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

   void displayMat(cv::Mat image);
   void mousePressEvent(QMouseEvent* event);
   void mouseReleaseEvent(QMouseEvent* event);
   void CloseALL();

signals:
   /******************** 线程相关 *****************/
   void ThreadOneStart();
   void SendShow();

//自动生成 槽函数
private slots:
   //打开文件按钮
    void on_btn_OpenFile_clicked();
    //显示图像函数
    void showImg();

    void on_btn_ImgPorcess_clicked();

    void on_chk_ContinueShow_clicked();



private:
    Ui::MainWindow *ui;

    //定时器 用于显示图像
    QTimer *timerShowImg;
    //判断图像是否经过处理
    bool Processed;

    //线程
    QThread *threads;
    ThreadOneImagePro *ThreadOneImage;

};

#endif // MAINWINDOW_H
