#ifndef THREADONEIMAGEPRO_H
#define THREADONEIMAGEPRO_H

#include <QObject>
#include <QTimer>
#include <QDebug>
#include <sys/time.h>
#include "classimgpro.h"
#include "global.h"


class ThreadOneImagePro : public QObject
{
    Q_OBJECT
public:
    explicit ThreadOneImagePro(QObject *parent = nullptr);


    void SetFlag(bool flag = false);
private:
    bool isStop;

signals:
    void ThreadOneDone();
public slots:
    void ImageProcessThread();

};

#endif // THREADONEIMAGEPRO_H
