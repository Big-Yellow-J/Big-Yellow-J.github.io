/*
C++中变量使用
一、变量基本使用
变量基本声明方式为：变量类型 变量名称 变量值（这个不一定需要）
变量分为两种：1、变量（后续可以被修改）；2、常量（不可以被修改），可以直接通过 #define 或者 const
补充：
1、如果需要用字符串，需要使用 #include <string>
2、引号内容，在单引号只能放1个字符 但是双引号 需要是字符串
3、值得注意的是，上面操作只是对变量声明（告诉编译器 这个变量是上面类型的），所以这样会报错：tmp_a = float a;
4、可以使用 unsigned int/short 等 来去除负数让正数的范围翻倍

二、C++ 存储类
1、static
函数内部局部变量--> 保持值不丢失，只要程序运行就一直存在
文件顶层全局变量--> 只在本文件中可以访问
2、auto
对于变量一般定义：int count = 5 可以直接用 auto count = 5自动知道 count是一个整数类型变量
TODO: 学习C++函数使用介绍如下内容以及全局变量等；
3、extern（对比运行 code-2.cpp 和 code-2-1.cpp）
简单理解方式为：我在别处定义，你先别急着分配内存，我只是声明
*/

#include <iostream>
#include <string>
using namespace std;

// 存储类使用
void print();

static int count = 5;
#define NUM 50 // 定义常量 NUM 数值为50 注意没有分号
int num= 50;   // 定义变量 num 初始化数值为0

int main(){
    // 数值数据使用
    const int NUM1 = 50; // 定义常量 
    int a= 65;           // 对应 ASCII 码的 A
    float float1= 1.1;

    // 字符串使用
    const char* string1 = "aabb";
    char string3 = 'a';
    string string2 = "ccdd";

    // 存储类
    auto auto_a= a;
    auto auto_string1 = "aabb";

    // 输出结果
    num = a* num;
    cout<< a+ NUM<< "\n";
    cout<< num<< "\n";
    cout<<"浮点数和整数相加：" <<float1+ a<< "\n";
    cout<<"除法运算：" <<a/ float1<< "\t取余运算："<< a% NUM<< "\n";
    cout<<"数值类型变换：" <<int(float1)<< "\n";
    cout<<"数值类型变换：" <<char(a+ 1)<< "\n";

    cout<<"输出字符串：" <<string1<< "\n";
    cout<<"字符串相加：" <<string2+ string1<< "\n";
    cout<<"字符串大小：" <<sizeof(string2)<< "\n";
    cout<<"字符串长度：" <<string2.length()<< "\n";

    while(count--)
    {
        print();
    }
    return 0;
}

void print(){
    int i = 1;
    static int static_i = 0; // 局部静态变量
    static_i++;
    i++;
    cout<< "变量 i= "<< i<< "\t静态变量 static_i="<< static_i<< "\t变量count= "<< count<< "\n";
}