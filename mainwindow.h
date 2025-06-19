#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include "gui.h"
#include "core.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_addSymbol_clicked();

    void on_radioButton_clicked();

    void on_radioButton_4_clicked();

    void on_radioButton_3_clicked();

    void on_recognizeButton_clicked();

    void on_fitButton_clicked();

private:
    Matrix parseString(QString str);
    vector<double> parseLabel(int label);
    Ui::MainWindow *ui;
    QString datafile = "train_data.txt";
    QString labelsfile = "train_labels.txt";
    ImageWidget* symbol;
    int type = -1;
    NN *nn = Q_NULLPTR;
};
#endif // MAINWINDOW_H
