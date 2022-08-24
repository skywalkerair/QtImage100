/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QLabel *QShowImageLabel;
    QPushButton *btn_OpenFile;
    QPushButton *btn_ImgPorcess;
    QCheckBox *chk_ContinueShow;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1011, 719);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 0, 811, 521));
        QShowImageLabel = new QLabel(groupBox);
        QShowImageLabel->setObjectName(QStringLiteral("QShowImageLabel"));
        QShowImageLabel->setGeometry(QRect(10, 0, 781, 511));
        QFont font;
        font.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        font.setBold(false);
        font.setWeight(50);
        QShowImageLabel->setFont(font);
        QShowImageLabel->setFrameShape(QFrame::Box);
        QShowImageLabel->setLineWidth(2);
        QShowImageLabel->setMidLineWidth(0);
        QShowImageLabel->setTextFormat(Qt::AutoText);
        QShowImageLabel->setAlignment(Qt::AlignCenter);
        QShowImageLabel->setMargin(1);
        btn_OpenFile = new QPushButton(centralWidget);
        btn_OpenFile->setObjectName(QStringLiteral("btn_OpenFile"));
        btn_OpenFile->setGeometry(QRect(850, 20, 121, 41));
        QFont font1;
        font1.setFamily(QString::fromUtf8("\346\245\267\344\275\223"));
        font1.setPointSize(12);
        font1.setBold(true);
        font1.setWeight(75);
        btn_OpenFile->setFont(font1);
        btn_ImgPorcess = new QPushButton(centralWidget);
        btn_ImgPorcess->setObjectName(QStringLiteral("btn_ImgPorcess"));
        btn_ImgPorcess->setGeometry(QRect(850, 90, 121, 41));
        btn_ImgPorcess->setFont(font1);
        chk_ContinueShow = new QCheckBox(centralWidget);
        chk_ContinueShow->setObjectName(QStringLiteral("chk_ContinueShow"));
        chk_ContinueShow->setGeometry(QRect(840, 160, 151, 19));
        chk_ContinueShow->setFont(font1);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1011, 26));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        groupBox->setTitle(QString());
        QShowImageLabel->setText(QApplication::translate("MainWindow", "\345\233\276\345\203\217\346\230\276\347\244\272\345\214\272", Q_NULLPTR));
        btn_OpenFile->setText(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\226\207\344\273\266", Q_NULLPTR));
        btn_ImgPorcess->setText(QApplication::translate("MainWindow", "\345\244\204\347\220\206\345\233\276\345\203\217", Q_NULLPTR));
        chk_ContinueShow->setText(QApplication::translate("MainWindow", "\345\256\236\346\227\266\346\233\264\346\226\260\345\233\276\345\203\217", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
