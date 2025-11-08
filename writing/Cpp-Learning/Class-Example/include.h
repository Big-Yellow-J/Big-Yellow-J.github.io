// include.h
#ifndef INCLUDE_H
#define INCLUDE_H

#include <iostream>
#include <string>
#include <cmath>

using namespace std;

class BaseNum{
    public:
        double length=2, width=2, height=2;
        double radius=10;
};
class NormalNum{
    public:
        double PI=3.14;
};

class Geometry: public BaseNum, public NormalNum{
    public:
        // 定义 静态成员 数据 只要使用这个类就会 彼此之间各项这个 静态成员
        static int com_num; // 静态成员 统计计算次数
        // 使用和类相同的名称的函数 ==> 构造函数
        Geometry(double l, double w, double h, double r){
            length= l; width= w; height= h; radius= r;
        };
        Geometry(){length=2; width=2; height=2; radius=2;};
        // 拷贝构造函数
        Geometry(const Geometry& other){
            length= other.length; height= other.height;
            width= other.width; radius= other.radius;}
        ~Geometry();                                  // 析构函数
        static void cout_fun();
        void setRadius(double r);                     // 修改 private 的值
        double CircleArea(double r= -1);              // 计算圆的面积
        double CirclePerimeter(double r= -1);         // 计算圆的周长
        double CylinderVolume(double r= -1);          // 计算圆柱体积
        void print_normal();
        // void print_normal() const; 如果使用 const 那么在这个函数里面也只能使用 const的函数
};
#endif // CONCAT_ACTION_H