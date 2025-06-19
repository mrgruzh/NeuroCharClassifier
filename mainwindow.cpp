#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTextStream>
#include <QFile>
#include <QDebug>
#include "core.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //this->setStyleSheet();
    symbol = new ImageWidget(this);
    symbol->setTopLeft(200,120);
    ui->recognizeButton->setEnabled(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_addSymbol_clicked()
{
    QFile file_d(datafile);
    file_d.open(QIODevice::Append);
    QDataStream stream1(&file_d);
    QString data = symbol->getData();
    if(file_d.isOpen()){
        stream1<<data;
    }
    file_d.close();
    qDebug()<<data;
    QFile file_l(labelsfile);
    file_l.open(QIODevice::Append);
    QDataStream stream2(&file_l);
    if(file_l.isOpen()){
        stream2<<type;
    }
    file_l.close();
    qDebug()<<type;
    symbol->clear();
}

void MainWindow::on_radioButton_clicked()
{
    type = 0;
}

void MainWindow::on_radioButton_4_clicked()
{
    type = 1;
}

void MainWindow::on_radioButton_3_clicked()
{
    type = 2;
}

void MainWindow::on_recognizeButton_clicked()
{
    std::map<int,QString> map;
    map[0]="Р";map[1]="2";map[2]="Б";map[-1]="Не распознано";
    QString data_x = symbol->getData();
    Featurizer featurizer;
    int y = nn->predict(featurizer.create_features(parseString(data_x)));
    ui->linePredict->setText(map[y]);
    qDebug()<<y;

}

Matrix MainWindow::parseString(QString str)
{
    Matrix result;
    vector<double> row;
    for(int i = 0; i < 8;i++)result.matrix.push_back(row);
    int j = 0;
    int s = 0;
    for(int i = 0; i < str.size();i++){
        if(isdigit(str.at(i).toLatin1())){
            result.matrix[j].push_back(str.at(i).toLatin1()-'0');
            s++;
            if(s % 8 == 0)j++;
        }
    }
    qDebug()<<str;
    qDebug()<<result.matrix;
    return result;
}

std::vector<double> MainWindow::parseLabel(int label)
{
    std::vector<double> result;
    for(int i = 0; i <3;i++){
        if(i != label)
            result.push_back(0.0);
        else {
            result.push_back(1.0);
        }
    }
    return result;
}

void MainWindow::on_fitButton_clicked()
{
    if(nn != Q_NULLPTR)
        delete nn;
    vector<vector<double>> X;
    vector<vector<double>> Y;
    QFile file_d(datafile);
    file_d.open(QIODevice::ReadOnly);
    QDataStream stream1(&file_d);
    QString data;
    Featurizer featurizer;
    if(file_d.isOpen()){
        while(!file_d.atEnd()){
            stream1>>data;
            Matrix matrix = parseString(data);
            vector<double> x = featurizer.create_features(matrix);
            X.push_back(x);
        }
    }
    file_d.close();
    QFile file_l(labelsfile);
    file_l.open(QIODevice::ReadOnly);
    QDataStream stream2(&file_l);
    int type;
    int i = 0;
    std::map<int,QString> map;
    map[0]="Р";map[1]="2";map[2]="Б";map[-1]="Не распознано";
    if(file_l.isOpen()){
        while(!file_l.atEnd()){
            stream2>>type;
            Y.push_back(parseLabel(type));
            QString str = Q_NULLPTR;
            for(int j = 0; j < 3;j++){
                if(Y[i][j] == 1.0){
                    str = map[j];
                }
            }
            qDebug()<<"symbol: "<<str<<+" , features: "<<X[i];
            i++;
        }
    }
    file_l.close();
    nn = new NN();
    uint size = X.size();

    for(uint i=0;i<size;i++){
        X.push_back(X[i]);
        Y.push_back(Y[i]);
    }

    Matrix m_X;
    m_X.matrix = X;
    Matrix m_Y;
    m_Y.matrix = Y;
     nn->fit(m_X,m_Y,110);
       qDebug()<<"cross_entropy test: "<<nn->cross_entropy(m_X,m_Y);
    qDebug()<<"accuracy test: "<<nn->accuracy(m_X,m_Y);
    ui->recognizeButton->setEnabled(true);
}
