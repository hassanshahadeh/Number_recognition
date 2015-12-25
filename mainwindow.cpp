#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <QFile>
#include <QDir>
#include<dir.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/ml.h>
#include <cstdlib>
#include<QFile>
#include <ctype.h>
#include <cstdlib>
#include <QTextStream>
#include <QtCore>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <strings.h>
#include <string.h>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
//Mat training_labels,hogfeat;
QString filename;
using namespace cv;


using namespace std;
Mat training_labels,hogfeat;
int  countNu=0;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_HOG_clicked()
{
    for (int  k= 0;k  < 10; k++) {

        filename=QString("C:\\Qt\\Qt5.1.1\\Tools\\QtCreator\\bin\\Nmber_re\\project_number\\train\\%1").arg(k);
        QString p=filename;
      //  QStringList filters("*.jpg");
       // QStringList filters1("*.bmp");
       QDir dir(p);
       //list=dir.entryInfoList();
       dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
           dir.setSorting(QDir::Size | QDir::Reversed);
           QFileInfoList list = dir.entryInfoList();
        QString fpath;
        Mat temp(list.size(),1,CV_16U);
        Mat feat;
         for (int i = 0; i < list.size(); i++){
             temp.at<LONG>(i,0)=k;
             QFileInfo finfo=list.at(i);
             fpath =finfo.absoluteFilePath();

             Mat m=imread(fpath.toStdString(),1);
             Mat ImageInput=m.clone();
            cv::resize(m,ImageInput,Size(16,16));
             // cvtColor(ImageInput,ImageInput,CV_RGB2GRAY);
             try{
            // imshow("hog",ImageInput);

             waitKey(1 )
                     catch(...){}

             HOGDescriptor hog;
             vector<float> ders;
             vector<Point> locs;
             hog.blockSize=Size(16,16);
             hog.blockStride=Size(16,16);
             hog.cellSize=Size(8,8);
             hog.winSize=Size(16,16);
//             hog.blockSize=Size(16,16);
//             hog.blockStride=Size(16,16);
//             hog.cellSize=Size(8,8);
//             hog.winSize=Size(48,32);
//             std::vector<Mat> rgb;
//             split(ImageInput,rgb);


//             equalizeHist(rgb.at(0),rgb.at(0));
//             equalizeHist(rgb.at(1),rgb.at(1));
//             equalizeHist(rgb.at(2),rgb.at(2));
//             merge(rgb,ImageInput);
             hog.compute(ImageInput,ders,Size(0,0),Size(0,0),locs);
             feat.create(1,ders.size(),CV_32FC1);
             for (int j = 0; j < ders.size(); j++)
                 feat.at<float>(0,j)=ders.at(j);
              hogfeat.push_back(feat);
                 feat.release();
          //  cout<<"data samples are:("<<feat.rows<<","<<feat.cols<<")"<<endl;
            //     waitKey();
         }
         training_labels.push_back(temp);

    }try{
    cout<<"data samples are:("<<hogfeat.rows<<","<<hogfeat.cols<<")"<<endl;
    cout<<"data samples are:("<<training_labels.rows<<","<<training_labels.cols<<")"<<endl;
    FileStorage file1("svm_fe.xml",FileStorage::WRITE);
    file1<<"hogfeat"<<hogfeat;
    file1.release();
    FileStorage file2("svm_la.xml",FileStorage::WRITE);
    file2<<"labels"<<training_labels;
    file2.release();
    }
    catch(...){}

}

void MainWindow::on_SVM_clicked()
{
    FileStorage file1("svm_fe.xml",FileStorage::READ);
     file1["hogfeat"]>>hogfeat;
     file1.release();
     FileStorage file2("svm_la.xml",FileStorage::READ);
     file2["labels"]>>training_labels;
     file2.release();
     Mat tr;
     training_labels.convertTo(tr,CV_32FC1);
     CvSVMParams params;
     params.svm_type=CvSVM::C_SVC;
   //  params.degree=1;
      params.gamma=3;
      params.kernel_type=CvSVM::C;

    // params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    CvSVM svm;

    cout<<"initialize svm done datd samples:"<<hogfeat.rows<<"\n"<<"number of features each sample:"<<hogfeat.cols<<"\n"<<"training_lables:("<<training_labels.rows<<","<<training_labels.cols<<")"<<endl;
   // svm.train(hogfeat,tr,Mat(),Mat(),params);
    svm.train(hogfeat,tr,Mat(),Mat(),params);
     cout<<"training...."<<endl;
     svm.save("svm_orb.xml");
     cout<<"svm rady...."<<endl;

}

void MainWindow::on_Test_clicked()
{
    CvSVMParams params;
    params.svm_type=CvSVM::C_SVC;
    params.kernel_type=CvSVM::C;
    params.gamma=3;
    CvSVM svm;
    svm.load("svm_orb.xml");Mat m;
    int i=0;
    for (int ii = 0; ii < 10; ++ii) {
        filename =QString("C:\\Qt\\Qt5.1.1\\Tools\\QtCreator\\bin\\Nmber_re\\project_number\\test\\%1").arg(ii);
//        QString p=filename;
//        QStringList filters("*.jpg");
//        QDir dir(p);
//        QFileInfoList list =dir.entryInfoList(filters);
        QString p=filename;
      //  QStringList filters("*.jpg");
       // QStringList filters1("*.bmp");
       QDir dir(p);
       //list=dir.entryInfoList();
       dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
           dir.setSorting(QDir::Size | QDir::Reversed);
           QFileInfoList list = dir.entryInfoList();
        QString fpath;

        for (int jj = 0; jj < list.size(); ++jj) {
            QFileInfo fInfo=list.at(jj);
            fpath =fInfo.absoluteFilePath();
             m=imread(fpath.toStdString(),1);
            Mat ImageInput=m.clone();
           cv::resize(m,ImageInput,Size(16,16));


            imshow("test",ImageInput);
            waitKey(1);

            Mat feat;
            HOGDescriptor hog;
            vector<float> ders;
            vector<Point> locs;
            hog.blockSize=Size(16,16);
            hog.blockStride=Size(16,16);
            hog.cellSize=Size(8,8);
            hog.winSize=Size(16,16);
            hog.compute(ImageInput,ders,Size(0,0),Size(0,0),locs);
            feat.create(1,ders.size(),CV_32FC1);
            for(int j=0;j<ders.size();j++){
             feat.at<float>(0,j)=ders.at(j);
            }
             i=svm.predict(feat);

             cout<<i<<endl;
             if(i==ii){
                 countNu++;
             }

        }

    }
    cout<< countNu<<endl;

}
