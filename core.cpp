#include "core.h"
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <QDebug>
#include <QFile>

#include <QtCharts/QLineSeries>
#include <QtCharts/QChartView>
#include <QtCharts/QValueAxis>

#include <algorithm>
#include <random>
#include <chrono>
//using namespace QtCharts;


Neuron::Neuron(int size)
{
     weights.resize(size);
}

void Neuron::initialize()
{
    w0 = -0.1 + rand()/double(RAND_MAX)*0.2;
    for(size_t i = 0; i < weights.size();i++)
        weights[i] = -0.1 + rand()/double(RAND_MAX)*0.2;
}

void Neuron::calculate(std::vector<double> x)
{
    value = 0;
    value += w0;
    for(size_t i = 0; i < x.size();i++)
        value += x[i]*weights[i];
}

void Neuron::step(double err,std::vector<double> x)
{
    w0 -= learning_rate*err;
    for(int i = 0; i < weights.size();i++){
        weights[i] -=  learning_rate*err*x[i];
    }
}

Layer::Layer(int size, QString activate,int input_dim)
{
    for(int i = 0; i < size;i++)
        neurons.push_back(new Neuron(input_dim));
    this->activation = activate;
}

void Layer::initialize()
{
    for(auto neuron: neurons)
        neuron->initialize();
}

std::vector<double> Layer::calculate(std::vector<double> x)
{
    std::vector<double> result;
    for(auto neuron: neurons){
        neuron->calculate(x);
        result.push_back(neuron->value);
    }
    return result;
}

std::vector<double> Layer::activate(std::vector<double> x)
{
    std::vector<double> y_;
    if(activation == "softmax")
        y_ = softmax(calculate(x));
    if(activation == "Relu")
        y_ = Relu(calculate(x));
    if(activation == "tanh")
        y_ = tanh(calculate(x));
    return y_;
}

std::vector<double> Layer::softmax(std::vector<double> x)
{
    std::vector<double> result;
    double sum = 0;
    double max = x[0];
    for(auto e:x){
       if(max < e) max = e;
    }
    for(auto e:x){
       sum+=exp(e-max);
    }
    for(auto e:x){
        double value = exp(e-max)/sum;
        result.push_back(value > 0.001 ? value : 0.001);
    }

    return result;
}

std::vector<double> Layer::Relu(std::vector<double> x)
{
    std::vector<double> result;
    for(auto e:x)
        result.push_back(e > 0.0 ? e : 0.0);
    return result;
}

std::vector<double> Layer::tanh(std::vector<double> x)
{
    std::vector<double> result;
    for(auto e:x)
        result.push_back(1.0/(1+exp(-e)));
    return result;
}


NN::NN()
{
    layers.push_back(new Layer(16,"tanh",4));
    layers.push_back(new Layer(4,"softmax",16));
}


auto NN::split(Matrix X, Matrix y, double test_size)
{
    Matrix X_train, X_test, y_train, y_test;

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indexes;
    for (size_t i = 0; i < X.matrix.size(); ++i) {
        indexes.push_back(i);
    }
    std::random_shuffle(indexes.begin(), indexes.end());

    size_t borderIndex = X.matrix.size() * (1.0 - test_size);
    for(size_t i = 0; i < indexes.size(); ++i) {
        size_t index = indexes[i];
        if (index < borderIndex) {
            X_train.matrix.push_back(X.matrix[index]);
            y_train.matrix.push_back(y.matrix[index]);
        }
        else {
            X_test.matrix.push_back(X.matrix[index]);
            y_test.matrix.push_back(y.matrix[index]);
        }
    }

    return std::tuple(X_train, X_test, y_train, y_test);
}
void NN::fit(Matrix X, Matrix Y, unsigned epochs)
{
    std::vector<std::pair<int, double>> test_accuracies;
    std::vector<std::pair<int, double>> train_accuracies;

    QFile file_1("accuracy.txt");
    file_1.open(QIODevice::WriteOnly | QFile::Text);
    QTextStream  stream1(&file_1);
    srand(579);
    // инициализация весов
    for(size_t i = 0; i < layers.size();i++){
        layers[i]->initialize();
    }
    // обучение по эпохам

    auto [X_train, X_test, y_train, y_test] = split(X, Y, 0.3);
//    qDebug() << "split:" << X_train.matrix << endl << X_test.matrix  << endl <<  y_train.matrix << endl << y_test.matrix;
    for(size_t i = 0; i < epochs;i++){
        for(size_t j = 0; j < X_train.matrix.size();j++){
            // выбрать элемент x из X случайно
            vector<double> x = X_train.matrix[j];
            vector<double> y = y_train.matrix[j];
            // прямой ход
            vector<double> a = layers[0]->activate(x);
            vector<double> y_ = layers[1]->activate(a);
            // обратный ход
            vector<double> sigma2;
            for(size_t k = 0; k < layers[1]->neurons.size(); k ++){
                sigma2.push_back(y_[k]-y[k]);
            }
            vector<double> sigma1;
            for(size_t k = 0; k < layers[0]->neurons.size(); k ++){
                double sum = 0;
                for(size_t s = 0; s < layers[1]->neurons.size(); s ++)
                    sum += sigma2[s]*layers[1]->neurons[s]->weights[k];
                sigma1.push_back(sum);
            }
            // градиентный шаг

            for(size_t k = 0;k < layers[1]->neurons.size();k++)
                layers[1]->neurons[k]->step(sigma2[k],a);
            for(size_t k = 0;k < layers[0]->neurons.size();k++)
               layers[0]->neurons[k]->step(sigma1[k]*(a[k]*(1-a[k])),x);

            qDebug() << x;
        }
        // значение метрики после эпохи

        qDebug() << "cross_entropy: " << cross_entropy(X,Y);
        qDebug() << "accuracy train: " << accuracy(X_train, y_train);
        qDebug() << "accuracy test: " << accuracy(X_test, y_test);

        train_accuracies.emplace_back(i, accuracy(X_train, y_train));
        test_accuracies.emplace_back(i, accuracy(X_test, y_test));

    }
    file_1.close();

    show_accuracy_plot("Тестовая выборка", test_accuracies);
    show_accuracy_plot("Тренировочная выборка", train_accuracies);
}

void NN::show_accuracy_plot(const QString title, std::vector<std::pair<int, double>> accuracies)
{
    QLineSeries* series = new QLineSeries();
    for(auto [epoch, accuracy]: accuracies) {
        series->append(epoch, accuracy);
    }

    QChart* chart = new QChart();

    QValueAxis* axisY = new QValueAxis(chart);
    axisY->setRange(0, 1);
    axisY->setTitleText("Точность");
    axisY->setLabelFormat("%.2lf");
    chart->addAxis(axisY, Qt::AlignLeft);

    int epochs = accuracies.back().first;
    QValueAxis* axisX = new QValueAxis(chart);
    axisX->setTitleText("Эпоха");
    axisX->setLabelFormat("%d");
    axisX->setRange(0, epochs);
    chart->addAxis(axisX, Qt::AlignBottom);

    chart->addSeries(series);
    series->attachAxis(axisX);
    series->attachAxis(axisY);
    chart->setTitle(title);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->show();
    chartView->resize(640, 480);
    chartView->setWindowTitle(title);

}

double NN::cross_entropy(Matrix X,Matrix Y)
{
    double score = 0.0;
    for(uint k = 0; k < X.matrix.size();k++){
        double sum = 0;
        vector<double> y_ = predict_proba(X.matrix[k]);
        for(uint l = 0; l < 4;l++)
            sum -= Y.matrix[k][l]*log(y_[l]);
        score += sum;
    }
    return score/((double) X.matrix.size());
}

double NN::accuracy(Matrix X,Matrix Y)
{
    double score = 0.0;
    for(uint k = 0; k < X.matrix.size();k++){
        int y_ = predict(X.matrix[k]);
        int y = 0;
        for(uint l = 0; l < 4;l++)
            if(Y.matrix[k][l] == 1.0) y = l;
        if(y == y_) score++;
    }
    return score/((double) X.matrix.size());
}

pair<pair<Matrix,Matrix>,pair<Matrix,Matrix>> NN::split(const Matrix X,const Matrix Y)
{
    srand(static_cast<uint>(time(nullptr)));
    Matrix copy_X;
    copy_X.matrix = X.matrix;
    Matrix copy_Y;
    copy_Y.matrix = Y.matrix;
    Matrix train_X, test_X, train_Y, test_Y;
    int train_size = 0.66*X.matrix.size();
    for(uint i=0;i<train_size;i++){
        uint index = rand()%copy_X.matrix.size();
        train_X.matrix.push_back(copy_X.matrix[index]);
        train_Y.matrix.push_back(copy_Y.matrix[index]);
        copy_X.matrix.erase(copy_X.matrix.begin()+index);
        copy_Y.matrix.erase(copy_Y.matrix.begin()+index);
    }
    for(uint i=0;i<copy_X.matrix.size();i++){
        test_X.matrix.push_back(copy_X.matrix[i]);
        test_Y.matrix.push_back(copy_Y.matrix[i]);
    }
    pair<Matrix,Matrix> train = pair<Matrix,Matrix>(train_X,train_Y);
    pair<Matrix,Matrix> test = pair<Matrix,Matrix>(test_X,test_Y);
    return pair<pair<Matrix,Matrix>,pair<Matrix,Matrix>>(train,test);
}

std::vector<double> NN::predict_proba(vector<double> x)
{
    vector<double> a = layers[0]->activate(x);
    vector<double> y_ = layers[1]->activate(a);
    return y_;
}

int NN::predict(vector<double> x)
{
    vector<double> a = layers[0]->activate(x);
    vector<double> y_ = layers[1]->activate(a);
    int argmax = 0;
    double max = -1.0;
    double prob = 1.0;
    for(auto e:y_)
        prob*=(1-e);
    y_.push_back(prob);
    for(uint i = 0; i < y_.size();i++)
        if(y_[i] > max){
            max = y_[i];
            argmax = i;
        }
    return argmax;
}



vector<double> Featurizer::create_features(Matrix X){
    vector<double> result;
    result.push_back(vertical_feature(X));
    result.push_back(horizontal_feature(X));
    result.push_back(right_slope_feature(X));
    result.push_back(left_slope_feature(X));
    return result;
}

int Featurizer::vertical_feature(Matrix X){
    vector<vector<double>> matrix = X.matrix;
    int result = 0;
    uint size = matrix.size();
    for(uint i = 0; i < size;i++){
        int flag = 0;
        for(uint j = 0; j < size;j++){
            if(matrix[i][j] == 1.0)flag++;
            else{
                if(flag > 1) result++;
                flag = 0;
            }
        }
        if(flag > 1) result++;
    }
    return result;
}

int Featurizer::horizontal_feature(Matrix X){
    vector<vector<double>> matrix = X.matrix;
    int result = 0;
    uint size = matrix.size();
    for(uint i = 0; i < size;i++){
        int flag = 0;
        for(uint j = 0; j < size;j++){
            if(matrix[j][i] == 1.0)flag++;
            else{
                if(flag > 1) result++;
                flag = 0;
            }
        }
        if(flag > 1) result++;
    }
    return result;
}

int Featurizer::right_slope_feature(Matrix X){
    vector<vector<double>> matrix = X.matrix;
    int result = 0;
    uint size = matrix.size();
    for(uint i = 0; i < 2*size-3;i++){
        int flag = 0;
        uint x, y;
        y = i < size ? i : size-1;
        x = i < size ? 0 : i-size+1;
        while(x < size && y < size){
            if(matrix[y][x] == 1.0)flag++;
            else{
                if(flag > 1) result++;
                flag = 0;
            }
            y--;
            x++;
        }
        if(flag > 1) result++;
    }
    return result;
}

int Featurizer::left_slope_feature(Matrix X){
    vector<vector<double>> matrix = X.matrix;
    int result = 0;
    uint size = matrix.size();
    for(uint i = 0; i < 2*size-3;i++){
        int flag = 0;
        uint x, y;
        y = i < size ? i : size-1;
        x = i < size ? size-1 : 2*size-1-i;
        while(x < size && y < size){
            if(matrix[y][x] == 1.0)flag++;
            else{
                if(flag > 1) result++;
                flag = 0;
            }
            y--;
            x--;
        }
        if(flag > 1) result++;
    }
    return result;
}
