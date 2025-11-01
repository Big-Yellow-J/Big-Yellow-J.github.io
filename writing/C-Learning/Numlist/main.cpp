/*
C++ 中循环、判断、函数、数组
如果定义函数需要进行跨文件进行处理，那么可以按照下面方式处理（假设两个c++代码）
1、code-3.cpp；2、find_numlist.cpp；3、sort_numlist.cpp。那么处理步骤就是（假设要在1中使用2,3）
两种方式实现目标：1、使用头文件；2、使用makelist文件
第一种方式：
首先 定义头文件 numlist_function.h。在头文件中 把我需要使用的函数都加到头文件里面即可
而后 在所有需要”协同“的文件里面都添加 我的头文件 #include "numlist_function.h"
最后 编译。
上面过程中格外需要注意的是：⭐如果变量用auto处理的话要么将其函数放到 头文件中，要么将auto换成具体的类型
第二种方式：
直接写 CMakeLists.txt即可，然后不需要在每个文件里面定义头文件 #include "numlist_function.h"，不过
需要在最后的main.cpp 里面去使用 vector<int> insert_sort_function(vector<int> num_list);
编译方式的话：
cmake .
cmake --build .
然后运行 ./cmake_run

TODO: 去学习如何使用 cmake规则语法
TODO: 去学习如何使用template模板定义函数
TODO: 去处理更加复杂情况
*/

// #include "numlist_function.h"
#include <iostream>
#include <vector>            // 数组
#include <initializer_list>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

// 排序函数声明
vector<int> quick_sort_function(vector<int> num_list);
vector<int> insert_sort_function(vector<int> num_list);
vector<int> bubble_sort_function(vector<int> num_list);
void vector_print(vector<int> num_list, auto way);

// 查找函数声明
int sequence_search(vector<int> num_list, int target);
int binary_search(vector<int> num_list, int target);
int hash_search(vector<int> num_list, int target);
void search_print(int value);

// 使用 auto 参数编译过程中需要加参数 -std=c++20 
// g++ -std=c++20 code-3.cpp -o test
auto max_function(auto a, auto b) {
    if (a > b) return a;
    else return b;
}

int main() {
    // 定义数组
    auto num_list = vector<int>{2, 3, 5, 1, 10, 0};

    cout <<"输出 max_function(10, 20):\t" << max_function(10, 20) << "\n";
    cout <<"数组长度为："<< num_list.size()<< "\n";

    // 插入排序处理
    auto s_time = high_resolution_clock::now();
    insert_sort_function(num_list);
    auto e_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(e_time - s_time);
    cout << fixed << setprecision(3)<< "插入排序耗时: " << duration.count() << " 微秒\n";

    // 冒泡排序处理
    s_time = high_resolution_clock::now();
    bubble_sort_function(num_list);
    e_time = high_resolution_clock::now();
    duration = duration_cast<microseconds>(e_time - s_time);
    cout << fixed << setprecision(3)<< "冒泡排序处理耗时: " << duration.count() << " 微秒\n";

    // 快速排序
    s_time = high_resolution_clock::now();
    auto tmp_num_list = quick_sort_function(num_list);
    vector_print(tmp_num_list, "快速排序");
    e_time = high_resolution_clock::now();
    duration = duration_cast<microseconds>(e_time - s_time);
    cout << fixed << setprecision(3)<< "快速排序耗时: " << duration.count() << " 微秒\n";

    // 暴力查找
    int index= sequence_search(tmp_num_list, 10);
    search_print(index);

    // 二分查找
    // 第一次使用 int 已经定义了后续只需要 使用即可
    index= binary_search(tmp_num_list, 10);
    search_print(index);

    // 哈希查找
    index= hash_search(tmp_num_list, 10);
    search_print(index);

    return 0;
}

