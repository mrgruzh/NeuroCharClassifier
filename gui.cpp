#include "gui.h"
#include <QPainter>

PixelWidget::PixelWidget(QWidget* parent)
{
    this->setParent(parent);
}

void PixelWidget::clear()
{
    flag = false;
    repaint();
}

void PixelWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHints(QPainter::Antialiasing);
    QPen pen(Qt::black);
    int pen_width = 2;
    pen.setWidth(pen_width);
    painter.setPen(pen);
    painter.drawRect(rect);
    QRect fillingRect = QRect(rect.left()+pen_width,
                              rect.top()+pen_width,
                              rect.width()-2*pen_width,
                              rect.height()-2*pen_width);
    if(flag)
        painter.fillRect(fillingRect,QBrush(Qt::black));
    else {
        painter.fillRect(fillingRect,QBrush(Qt::white));
    }
}

void PixelWidget::mousePressEvent(QMouseEvent *event)
{
    flag = !flag;
    repaint();
}

ImageWidget::ImageWidget(QWidget* parent)
{
    this->setParent(parent);
    uint margin = 10;
    int pixel_size = PixelWidget::size();
    this->setGeometry(0,0,(n_pixels+1)*margin+n_pixels*pixel_size,
                      (n_pixels+1)*margin+n_pixels*pixel_size);
    for(uint i = 0; i < n_pixels;i++){
        std::vector<PixelWidget*> row;
        for(uint j = 0; j < n_pixels;j++){
            PixelWidget* pixel = new PixelWidget(this);
            pixel->setGeometry(margin*(j+1)+pixel_size*j,
                               margin*(i+1)+pixel_size*i,
                               pixel_size,
                               pixel_size);
            row.push_back(pixel);
        }
        symbol_matrix.push_back(row);
    }
}

void ImageWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHints(QPainter::Antialiasing);
    QPen pen(Qt::black);
    int pen_width = 2;
    pen.setWidth(pen_width);
    painter.setPen(pen);
    painter.drawRect(QRect(0,0,this->geometry().width(),
                           this->geometry().height()));
}

void ImageWidget::setTopLeft(int left,int top)
{
    this->setGeometry(left,top,this->geometry().width(),
                      this->geometry().height());
}

QString ImageWidget::getData()
{
    QString result;
    for(uint i = 0; i < n_pixels;i++)
        for(uint j = 0; j < n_pixels;j++)
            result+=QString::number(symbol_matrix[i][j]->getValue())+" ";
    return result;
}

ImageWidget::~ImageWidget()
{
    for(uint i = 0; i < n_pixels;i++)
        for(uint j = 0; j < n_pixels;j++)
            delete symbol_matrix[i][j];
}

void ImageWidget::clear()
{
    for(uint i = 0; i < n_pixels;i++)
        for(uint j = 0; j < n_pixels;j++)
            symbol_matrix[i][j]->clear();
}
