/*
C++中类的基本使用（可以通过python角度去对类进行了解），主要是设计到3大特性：
1、封装；2、继承；3、多态
类的基本定义：
class 类名{
public:
    类成员变量;
    类成员函数;
private:
    私有成员变量;
    私有成员函数;
};

值得注意的是：
1、public 是可以修改访问的，反之 private 是不可以修改访问的
2、public 内部成员函数可以直接在class外部定义
⭐3、在使用类过程中最好是使用 构造函数（Constructor）（类似python中的 __init__）相当于
最开始就给类的变量都给初始值（写代码先去写 类 中函数 然后再去定义函数）。对于构造函数分为：
有参/无参构造函数；普通构造和拷贝构造函数
4、与之对应的另外一个函数 析构函数 析构函数的名称与类的名称是完全相同的，只是在前面加了个波浪号（~）作为前缀，
它不会返回任何值，也不能带有任何参数。
5、如果在头文件里面定义了函数的初始值，那么后续具体函数实现里面就不要在使用
*/
#include "include.h"
#include <iostream>
using namespace std;

int main(){
    Geometry geometry_com1(10, 10, 10, 10);
    Geometry geometry_com2;
    Geometry geometry_com1_copy= geometry_com1;

    cout<< "圆的面积值为："<< geometry_com1.CircleArea(100)
        << "\t圆的面积值为："<< geometry_com1.CircleArea()
        << "\t圆的周长值为："<< geometry_com1.CirclePerimeter()
        << "\t圆柱体积为："<< geometry_com2.CylinderVolume()
        << "\t类的使用次数："<< geometry_com1.com_num<<endl;

    cout<< "圆的面积值为："<< geometry_com2.CircleArea()
        << "\t圆的周长值为："<< geometry_com2.CirclePerimeter()
        << "\t圆柱体积为："<< geometry_com2.CylinderVolume()
        << "\t圆的半径为："<< geometry_com2.radius
        << "\t类的使用次数："<< geometry_com1.com_num<< endl;

    double cylinder_volume = geometry_com1.CylinderVolume();

    geometry_com1.print_normal();
    geometry_com1.setRadius(10000);
    geometry_com1.print_normal();
    geometry_com1_copy.print_normal();
    return 0;
}