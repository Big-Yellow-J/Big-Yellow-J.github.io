// geometry.cpp
#include <iostream>
#include <cmath>
using namespace std;

const double PI = 3.14159;

// 圆计算
class Circle {
private:
    double r;  // 半径（私有）

public:
    // 构造函数
    Circle(double radius = 0) {
        r = radius;
        if (r < 0) r = 0;
    }

    // 设置半径
    void setRadius(double radius) {
        if (radius >= 0) r = radius;
    }

    // 获取半径
    double getRadius() {
        return r;
    }

    // 周长
    double perimeter() {
        return 2 * PI * r;
    }

    // 面积
    double area() {
        return PI * r * r;
    }

    // 打印
    void print() {
        cout << "圆: 半径 = " << r 
             << ", 周长 = " << perimeter() 
             << ", 面积 = " << area() << endl;
    }
};


int main() {
    Circle c1(5);
    c1.print();

    c1.setRadius(10);
    c1.print();

    return 0;
}