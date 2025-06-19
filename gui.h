#ifndef GUI_H
#define GUI_H

#include <QWidget>
#include <vector>

class PixelWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PixelWidget(QWidget* parent = Q_NULLPTR);
    ~PixelWidget(){}
    static int size(){return pixel_size;}
    int getValue(){return flag == false ? 0 : 1;}
    void clear();
protected slots:
protected:
    void mousePressEvent(QMouseEvent *event);
    void paintEvent(QPaintEvent* event);
private:
    bool flag = false;
    static const uint pixel_size = 40;
    QRect rect = QRect(0,0,pixel_size,pixel_size);
};

class ImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ImageWidget(QWidget* parent = Q_NULLPTR);
    ~ImageWidget();
    void setTopLeft(int left,int top);
    QString getData();
    void clear();
protected slots:
protected:
    void paintEvent(QPaintEvent* event);
private:
    uint n_pixels = 8;
    std::vector<std::vector<PixelWidget*>> symbol_matrix;
};

#endif // GUI_H
