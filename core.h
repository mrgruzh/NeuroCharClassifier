#ifndef CORE_H
#define CORE_H
#include <vector>
#include <QString>

using namespace std;

class Matrix
{
public:
    vector<vector<double>> matrix;
};

class Neuron
{
public:
    Neuron(int size = 65);
    void initialize();
    void calculate(vector<double> x);
    void step(double err, vector<double> x);
    vector<double> weights;
    double w0;
    double value;
    double learning_rate = 0.06;
};

class Layer
{
public:
    Layer(int n_neurons,QString activation,int input_dim);
    void initialize();
    vector<double> calculate(vector<double> x);
    vector<double> activate(vector<double> x);
    vector<Neuron*> neurons;
    vector<double> softmax(vector<double> x);
    vector<double> Relu(vector<double> x);
    vector<double> tanh(vector<double> x);
    QString activation;
};


class NN
{
public:
    NN();
    ~NN(){}

    auto split(Matrix X, Matrix y, double test_size);
    void fit(Matrix X,Matrix Y, unsigned epochs = 100);
    int predict(vector<double> x);
    double cross_entropy(Matrix X,Matrix Y);
    double accuracy(Matrix X, Matrix Y);
    vector<double> predict_proba(vector<double> x);
    pair<pair<Matrix,Matrix>,pair<Matrix,Matrix>> split(Matrix X, Matrix Y);

    void show_accuracy_plot(const QString title, std::vector<std::pair<int, double>> accuracies);
private:
    vector<Layer*> layers;
    double learning_rate;
};

class Featurizer{
public:
    vector<double> create_features(Matrix X);
private:
    int vertical_feature(Matrix X);
    int horizontal_feature(Matrix X);
    int right_slope_feature(Matrix X);
    int left_slope_feature(Matrix X);
};



#endif // CORE_H
