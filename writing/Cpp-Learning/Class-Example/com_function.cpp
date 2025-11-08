#include "include.h"
#include <cmath>

int Geometry::com_num = 0;
Geometry::~Geometry(){cout<< "释放资源"<<endl;}
void Geometry::setRadius(double r){if (r> 0) radius= r;}
double Geometry::CircleArea(double r){ // 这里r不要用额外初始值
    com_num++;
    double rad= (r< 0)? radius: r;
    return PI*pow(rad, 2);}
double Geometry::CirclePerimeter(double r){
    com_num++;
    double rad= (r< 0)? radius: r;
    return 2*PI*radius;}
double Geometry::CylinderVolume(double r){
    com_num++;
    double rad= (r< 0)? radius: r;
    return CircleArea(rad)* height;
}
void Geometry::print_normal(){
    cout << "圆面积 = " << CircleArea()
         << "\t圆周长 = " << CirclePerimeter() 
         << "\t计算次数："<< com_num
         << endl;
}
void Geometry::cout_fun(){cout<< "\n随便输出！\n"<< endl;}